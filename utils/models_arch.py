"""
Sparse Convolutional Neural Network Architecture for Event Classification.

This module implements a hierarchical sparse convolutional autoencoder pipeline
for particle physics event classification. The architecture uses rulebook-based
sparse convolutions (VSC3x3) which operate only on active (non-zero) spatial
locations, enabling efficient computation on sparse data.

Key Components:
- VSC3x3Rulebook: 3x3 kernel sparse convolution using pre-computed rulebook
- SparseVSCBlockRulebook: VSC conv + BatchNorm + ReLU block
- SparseMaxPool2x2: Max pooling adapted for sparse coordinates
- SparseVGGStage: Multiple VSC blocks followed by pooling (hierarchical feature extraction)
- AutoEncoder1: First-stage encoder-decoder for initial representation learning
- AutoEncoder2: Second-stage encoder-decoder for latent space refinement
- SparseEventClassifier: Classifier combining both encoders with a linear classification head

Pipeline Stages:
1. Encoder1: 8 -> 16 -> 32 channels (2 stages with 3x2 blocks)
2. Encoder2: 32 -> 64 channels (1 stage with 3 blocks)
3. Classifier: Global mean pooling + 2-layer MLP head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utility import *


class VSC3x3Rulebook(nn.Module):
    """
    Rulebook-based 3x3 sparse convolution (VSC: Volumetric Sparse Convolution).

    Uses pre-computed rulebooks to perform convolution only on pairs of active
    (non-zero) coordinates, avoiding computation on zero-valued regions.

    Args:
        in_ch (int): Number of input channels
        out_ch (int): Number of output channels
        bias (bool): Whether to use bias term (default: True)

    Input:
        coords: Sparse coordinates, shape (N, 3) [batch, y, x]
        feats: Features at coordinates, shape (N, in_ch)
        rules: Rulebook lists - 9 tuples of (in_rows, out_rows) for 3x3 kernel offsets

    Output:
        coords: Same as input coordinates
        out: Convolved features, shape (N, out_ch)
    """
    def __init__(self, in_ch, out_ch, bias=True):
        super().__init__()
        # Weight: (9, in_ch, out_ch) - one kernel position per 3x3 offset
        self.weight = nn.Parameter(
            torch.randn(9, in_ch, out_ch) * (1.0 / (in_ch ** 0.5))
        )
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None

    def forward(self, coords, feats, rules):
        # Apply sparse convolution using pre-computed rulebook
        out = vsc_forward_rulebook(feats, rules, self.weight, self.bias)
        return coords, out
    
    
class SparseVSCBlockRulebook(nn.Module):
    """
    Basic building block: VSC convolution + BatchNorm + ReLU.

    Processes sparse coordinates and features through:
    1. Sparse 3x3 convolution (VSC)
    2. Batch normalization
    3. ReLU activation

    Args:
        in_ch (int): Number of input channels
        out_ch (int): Number of output channels

    Input:
        coords: Sparse coordinates, shape (N, 3)
        feats: Features at coordinates, shape (N, in_ch)
        rules: Rulebook for convolution

    Output:
        coords: Same coordinates (unchanged)
        feats: Activated features, shape (N, out_ch)
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.vsc = VSC3x3Rulebook(in_ch, out_ch)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()

    def forward(self, coords, feats, rules):
        coords, feats = self.vsc(coords, feats, rules)
        feats = self.bn(feats)
        feats = self.act(feats)
        return coords, feats
    
    
class SparseMaxPool2x2(nn.Module):
    """
    Max pooling for sparse data with stride 2.

    Downsamples sparse coordinates by integer division and aggregates features
    at duplicate downsampled locations using max operation.

    Process:
    1. Downsample coordinates by dividing y,x by 2
    2. Find unique coordinates after downsampling
    3. Apply max reduction on features for each unique location

    Input:
        coords: Sparse coordinates, shape (N, 3) [batch, y, x]
        feats: Features at coordinates, shape (N, C)

    Output:
        coords: Downsampled unique coordinates, shape (M, 3) where M <= N
        feats: Max-pooled features, shape (M, C)
    """
    def __init__(self):
        super().__init__()

    def forward(self, coords, feats):
        coords_ds = downsample_coords_stride2(coords)
        uniq_coords, inv = torch.unique(coords_ds, dim=0, return_inverse=True)

        M = uniq_coords.shape[0]
        C = feats.shape[1]

        # expand inv so it matches feats shape
        index = inv.unsqueeze(1).expand(-1, C)

        pooled_feats = torch.full(
            (M, C),
            -torch.inf,
            dtype=feats.dtype,
            device=feats.device
        )

        pooled_feats.scatter_reduce_(
            0,
            index,
            feats,
            reduce="amax",
            include_self=True
        )

        return uniq_coords, pooled_feats


class SparseVGGStage(nn.Module):
    """
    VGG-style stage: Multiple VSC blocks followed by max pooling.

    Implements hierarchical feature extraction through stacked sparse convolutions
    (similar to VGG stages) with spatial downsampling via pooling.

    Channel progression:
    - First block: in_ch -> out_ch
    - Subsequent blocks: out_ch -> out_ch (same channel throughout stage)

    Args:
        in_ch (int): Input channels
        out_ch (int): Output channels
        n_blocks (int): Number of VSC blocks in this stage

    Input:
        coords: Sparse coordinates, shape (N, 3)
        feats: Features at coordinates, shape (N, in_ch)

    Output:
        coords: Downsampled coordinates after pooling
        feats: Hierarchical features, shape (N_pooled, out_ch)
    """
    def __init__(self, in_ch, out_ch, n_blocks):
        super().__init__()

        self.blocks = nn.ModuleList()

        for i in range(n_blocks):
            if i == 0:
                self.blocks.append(SparseVSCBlockRulebook(in_ch, out_ch))
            else:
                self.blocks.append(SparseVSCBlockRulebook(out_ch, out_ch))

        self.pool = SparseMaxPool2x2()

    def forward(self, coords, feats):
        # Build rulebook once for all VSC layers in this stage
        rules = build_vsc_rulebook(coords)

        for block in self.blocks:
            coords, feats = block(coords, feats, rules)

        # Pool changes coordinates (spatial downsampling)
        coords, feats = self.pool(coords, feats)

        return coords, feats

class Encoder1(nn.Module):
    """
    First-stage sparse encoder for initial feature hierarchy learning.

    Architecture:
    - Stage1: 8 -> 16 channels (3 VSC blocks)
    - Stage2: 16 -> 32 channels (2 VSC blocks)
    - Each stage includes spatial downsampling via max pooling

    Input shape:
        coords: (N, 3) - sparse coordinates [batch, y, x]
        feats: (N, 8) - 8-channel features (original jet data channels)

    Output shape:
        coords1: (M, 3) - downsampled coordinates
        z1: (M, 32) - latent features at 32 channels
    """
    def __init__(self):
        super().__init__()
        self.stage1 = SparseVGGStage(in_ch=8, out_ch=16, n_blocks=3)
        self.stage2 = SparseVGGStage(in_ch=16, out_ch=32, n_blocks=2)

    def forward(self, coords, feats):
        coords, feats = self.stage1(coords, feats)
        coords, feats = self.stage2(coords, feats)
        return coords, feats
    
    
class Decoder1MLP(nn.Module):
    """
    MLP-based decoder for reconstructing original dense tensor from sparse latent.

    Transforms global latent representation back to dense spatial format.
    Uses 3-layer MLP with ReLU activations.

    Args:
        latent_dim (int): Input dimension (from global pooling, default 32)
        hidden_dim (int): First hidden layer dimension (default 256)
        hidden_dim2 (int): Second hidden layer dimension (default 256)
        out_shape (tuple): Output shape as (H, W, C), default (125, 125, 8)

    Input:
        z_global: Global pooled latent, shape (B, 32)

    Output:
        x_hat: Reconstructed dense tensor, shape (B, 125, 125, 8)
    """
    def __init__(self, latent_dim=32, hidden_dim=256, hidden_dim2=256, out_shape=(125,125,8)):
        super().__init__()
        self.out_shape = out_shape
        out_dim = out_shape[0] * out_shape[1] * out_shape[2]

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, out_dim)
        )

    def forward(self, z_global):
        x_hat = self.net(z_global)
        x_hat = x_hat.view(z_global.shape[0], *self.out_shape)
        return x_hat
    
    
class AutoEncoder1(nn.Module):
    """
    Stage 1 Autoencoder: Sparse encoder + MLP decoder.

    Used for unsupervised representation learning on the unlabeled dataset.
    Trains to reconstruct original dense jet tensor from sparse encoding.

    Loss: Weighted MSE with higher weight on non-zero values (pos_weight=10.0)

    Input:
        coords: Sparse coordinates, shape (N, 3)
        feats: Sparse features, shape (N, 8)
        batch_size: Number of samples in batch

    Output:
        x_hat: Reconstructed dense tensor, shape (B, 125, 125, 8)
        coords1: Encoded sparse coordinates
        z1: Encoded sparse features, shape (N, 32)
        z_global: Global pooled latent, shape (B, 32)
    """
    def __init__(self):
        super().__init__()
        self.encoder = Encoder1()
        self.decoder = Decoder1MLP(
            latent_dim=32,
            hidden_dim=256,
            out_shape=(125, 125, 8)
        )

    def forward(self, coords, feats, batch_size):
        coords1, z1 = self.encoder(coords, feats)
        z_global = sparse_global_mean_pool(coords1, z1, batch_size)
        x_hat = self.decoder(z_global)
        return x_hat, coords1, z1, z_global


# -----------------------------------------


class Encoder2(nn.Module):
    """
    Second-stage sparse encoder for hierarchical latent refinement.

    Operates on the latent representation from Encoder1 to further refine features.
    Single stage that increases channel dimension: 32 -> 64.

    Architecture:
    - Single VGGStage: 32 -> 64 channels (3 VSC blocks)
    - Includes spatial downsampling via max pooling

    Input:
        coords: Sparse coordinates from Encoder1, shape (N, 3)
        feats: Latent features from Encoder1, shape (N, 32)

    Output:
        coords: Further downsampled coordinates
        feats: Higher-level latent features, shape (N, 64)
    """
    def __init__(self):
        super().__init__()
        self.stage = SparseVGGStage(in_ch=32, out_ch=64, n_blocks=3)

    def forward(self, coords, feats):
        coords, feats = self.stage(coords, feats)
        return coords, feats
    
    
class Decoder2MLP(nn.Module):
    """
    MLP decoder for Stage 2 Autoencoder.

    Maps global latent representation (64 dim) back to latent space of Stage 1 (32 dim).
    Uses 2-layer MLP with ReLU activation.

    Args:
        latent_dim (int): Input dimension from Encoder2 (default 64)
        hidden_dim (int): Hidden layer dimension (default 128)
        out_dim (int): Output dimension matching Encoder1 latent (default 32)

    Input:
        z_global: Global pooled latent from Encoder2, shape (B, 64)

    Output:
        pred: Reconstructed Encoder1 latent, shape (B, 32)
    """
    def __init__(self, latent_dim=64, hidden_dim=128, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, z_global):
        return self.net(z_global)
    
    

class AutoEncoder2(nn.Module):
    """
    Stage 2 Autoencoder: Sparse encoder + MLP decoder for latent space refinement.

    Takes Encoder1 latent representations and further processes them through
    Encoder2 for hierarchical refinement, then reconstructs the latent space.

    Loss: MSE between predicted and target Encoder1 latents

    Input:
        coords: Sparse coordinates from Encoder1 latent
        feats: Encoder1 latent features, shape (N, 32)
        batch_size: Number of samples

    Output:
        pred: Predicted Encoder1 latent, shape (B, 32)
        coords2: Further refined coordinates
        z2: Encoder2 output features, shape (N, 64)
        z2_global: Global pooled Encoder2 latent, shape (B, 64)
    """
    def __init__(self):
        super().__init__()
        self.encoder = Encoder2()
        self.decoder = Decoder2MLP(latent_dim=64, hidden_dim=128, out_dim=32)

    def forward(self, coords, feats, batch_size):
        coords2, z2 = self.encoder(coords, feats)
        z2_global = sparse_global_mean_pool(coords2, z2, batch_size)
        pred = self.decoder(z2_global)
        return pred, coords2, z2, z2_global
    

class SparseEventClassifier(nn.Module):
    """
    Sparse event classifier combining pretrained encoders with classification head.

    Architecture:
    1. Encoder1: 8 -> 16 -> 32 channels (hierarchical initial encoding)
    2. Encoder2: 32 -> 64 channels (refinement)
    3. Global mean pooling: Aggregate sparse representation to batch-level vector
    4. Classification head: 2-layer MLP (64 -> 64 -> num_classes)

    Training strategy:
    - Both encoders are frozen (pretrained on unlabeled data)
    - Only the classification head is trainable
    - This enables efficient transfer learning on labeled data

    Args:
        encoder1: Pretrained Encoder1 instance (frozen)
        encoder2: Pretrained Encoder2 instance (frozen)
        feat_dim (int): Latent feature dimension from Encoder2 (default 64)
        num_classes (int): Number of classification classes (default 2 for binary)
        dropout (float): Dropout rate in classifier head (default 0.2)

    Input:
        coords: Sparse coordinates, shape (N, 3)
        feats: Sparse features, shape (N, 8)

    Output:
        logits: Classification logits, shape (B, num_classes)
    """
    def __init__(self, encoder1, encoder2, feat_dim=64, num_classes=2, dropout=0.2):
        super().__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, coords, feats):
        # Determine batch size from coordinate tensor
        batch_size = int(coords[:, 0].max().item()) + 1

        # First encoding stage
        coords1, feats1 = self.encoder1(coords, feats)
        # Second encoding stage
        coords2, feats2 = self.encoder2(coords1, feats1)

        # Global aggregation: pool to per-sample vectors
        z = sparse_global_mean_pool(coords2, feats2, batch_size)   # (B, 64)
        # Classification
        logits = self.head(z)
        return logits