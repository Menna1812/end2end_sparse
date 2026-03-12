"""
Utility functions for sparse convolutional neural networks.

This module provides core utilities for operating on sparse tensors in the
sparse autoencoder pipeline. Key abstractions:

1. Sparse Format Conversion:
   - to_sparse_hwC: Dense (H,W,C) tensor -> sparse coordinates and features
   - Collation functions: Batch multiple sparse samples

2. Rulebook Construction:
   - build_vsc_rulebook: Pre-compute sparse convolution mappings from coordinates
   - Kernel-based sparse convolution indexing

3. Sparse Operations:
   - vsc_forward_rulebook: Efficient sparse convolution using rulebooks
   - sparse_global_mean_pool: Global pooling for batch-level aggregation
   - downsample_coords_stride2: Coordinate-space downsampling for pooling

4. Data Loading:
   - H5LabelledDataset: HDF5-backed dataset for labeled data
   - Collate functions for creating sparse batches

Coordinate Format:
- (N, 3) tensors with [batch_idx, y, x] for sparse samples
- Features are (N, C) with C channels per active location
"""

import torch
from torch.utils.data import Dataset
import h5py

# -----------------------------------------
# Sparse conversion
# -----------------------------------------

def to_sparse_hwC(x):
    """
    Convert dense tensor to sparse coordinate-feature format.

    Identifies non-zero spatial locations in a (H,W,C) tensor and extracts
    their coordinates and feature values.

    Args:
        x: Dense tensor, shape (H, W, C)

    Returns:
        coords: Sparse coordinates, shape (N, 2) where N is number of active sites
                Each row is [y, x] for an active spatial location
        feats: Features at those coordinates, shape (N, C)
    """
    # Create mask of non-zero locations (any channel non-zero counts as active)
    mask = (x != 0).any(dim=2)

    # Find coordinates of all active locations
    coords = mask.nonzero(as_tuple=False)     # (N, 2) [y, x]
    # Extract features at those locations
    feats = x[coords[:,0], coords[:,1]]       # (N, C)

    return coords.long(), feats.float()


# -----------------------------------------
# Collate function
# -----------------------------------------

def sparse_collate_hwC(batch):
    """
    Collate function for DataLoader to batch multiple sparse samples.

    Converts a batch of dense tensors into a single merged sparse representation
    with batch indices prefixed to coordinates.

    Process:
    1. Convert each dense sample to sparse format
    2. Prepend batch index to coordinates: (y,x) -> (batch_idx, y, x)
    3. Concatenate all coordinates and features
    4. Return sizes for later disaggregation

    Args:
        batch: List of (H, W, C) dense tensors

    Returns:
        coords: Concatenated coordinates, shape (N_total, 3) [batch_idx, y, x]
        feats: Concatenated features, shape (N_total, C)
        sizes: List of active site counts per sample in batch
    """

    coords_list = []
    feats_list = []
    sizes = []

    for b, x in enumerate(batch):
        # Convert dense sample to sparse format
        coords, feats = to_sparse_hwC(x)

        # Prepend batch index to coordinates
        batch_col = torch.full((coords.size(0),1), b, dtype=torch.long)
        coords = torch.cat([batch_col, coords], dim=1)

        coords_list.append(coords)
        feats_list.append(feats)
        sizes.append(coords.size(0))  # Count of active sites in this sample

    # Merge all samples into single sparse tensor
    coords = torch.cat(coords_list, dim=0)
    feats = torch.cat(feats_list, dim=0)

    return coords, feats, sizes


# -----------------------------------------
# VSC kernel offsets
# -----------------------------------------

KERNEL_3x3 = [
(-1,-1),(-1,0),(-1,1),
(0,-1),(0,0),(0,1),
(1,-1),(1,0),(1,1)
]
"""
3x3 convolution kernel offsets in (dy, dx) format.
Used for rulebook-based sparse convolution.
"""


# -----------------------------------------
# Rulebook construction
# -----------------------------------------

def build_vsc_rulebook(coords, offsets=KERNEL_3x3):
    """
    Build rulebooks for sparse volumetric convolution.

    A rulebook maps which input values contribute to which output values
    at each kernel offset position. This enables efficient sparse convolution
    by pre-computing which coordinates have valid neighbors.

    Process for each kernel offset (dy,dx):
    1. For each active coordinate (b,y,x), check if (b,y+dy,x+dx) is active
    2. Record input and output row indices for valid pairs
    3. These mappings enable matrix multiplication over sparse pairs only

    Args:
        coords: Sparse coordinates, shape (N, 3) [batch_idx, y, x]
        offsets: List of (dy, dx) kernel offset tuples, default is 3x3 kernel

    Returns:
        rules: List of 9 tuples (in_rows, out_rows) tensors
               One tuple per kernel offset, maps input rows to output rows
    """

    # Build dictionary: coordinates -> row indices for O(1) lookup
    coord2row = {tuple(c.tolist()): i for i,c in enumerate(coords)}

    rules = []

    # For each kernel offset
    for dy,dx in offsets:

        in_rows = []
        out_rows = []

        # For each output location
        for i,(b,y,x) in enumerate(coords.tolist()):

            # Check if input neighbor exists at this offset
            key = (b, y+dy, x+dx)

            if key in coord2row:
                # Map: input row -> output row
                in_rows.append(coord2row[key])
                out_rows.append(i)

        rules.append((torch.tensor(in_rows), torch.tensor(out_rows)))

    return rules


# -----------------------------------------
# VSC forward
# -----------------------------------------

def vsc_forward_rulebook(feats, rules, weight, bias=None):
    """
    Perform sparse 3x3 volumetric convolution using pre-computed rulebooks.

    Efficiently computes convolution by only processing active coordinate pairs.
    For each kernel offset, uses the rulebook to gather input pairs and apply
    the corresponding kernel weight.

    Computation for offset k:
    out[out_rows] += feats[in_rows] @ weight[k]

    Args:
        feats: Input features, shape (N, in_ch)
        rules: Rulebook list = [(in_rows, out_rows), ...] for each kernel offset (9 total)
        weight: Convolution weights, shape (9, in_ch, out_ch)
        bias: Optional bias, shape (out_ch,)

    Returns:
        out: Convolved features, shape (N, out_ch)
    """

    # Initialize output with zeros
    out = feats.new_zeros(feats.size(0), weight.size(-1))

    # Process each kernel offset
    for k,(in_rows,out_rows) in enumerate(rules):

        if len(in_rows) == 0:
            # Skip if no valid pairs at this offset
            continue

        # Gather input features by row indices, apply kernel weight at offset k
        out[out_rows] += feats[in_rows] @ weight[k]

    # Add bias if provided
    if bias is not None:
        out += bias

    return out


# -----------------------------------------
# Coordinate downsampling
# -----------------------------------------

def downsample_coords_stride2(coords):
    """
    Downsample sparse coordinates by stride 2 (for max pooling).

    Performs integer division of spatial coordinates (y,x) by 2.
    Multiple coordinates may map to the same downsampled location
    (handled later in pooling via aggregation).

    Args:
        coords: Sparse coordinates, shape (N, 3) [batch_idx, y, x]

    Returns:
        coords_ds: Downsampled coordinates, shape (N, 3)
    """

    coords = coords.clone()

    # Integer division of spatial coordinates by 2
    coords[:,1] //= 2
    coords[:,2] //= 2

    return coords


# -----------------------------------------
# Global pooling
# -----------------------------------------


def sparse_global_mean_pool(coords, feats, batch_size):
    """
    Global mean pooling for sparse tensors.

    Aggregates sparse features to a batch-level vector by computing mean
    feature value per batch sample.

    Process:
    1. Initialize zero tensor of shape (B, C) for batch-level outputs
    2. For each batch index, accumulate (sum) all features at that batch
    3. Count number of active sites per batch
    4. Divide accumulated sum by count to get mean

    Args:
        coords: Sparse coordinates, shape (N,3) [batch_idx, y, x]
        feats: Sparse features at coords, shape (N, C)
        batch_size: Number of samples in batch (B)

    Returns:
        pooled: Global mean pooled features, shape (B, C)
    """
    B = batch_size
    C = feats.shape[1]

    # Initialize output for batch-level aggregation
    pooled = torch.zeros(B, C, dtype=feats.dtype, device=feats.device)
    counts = torch.zeros(B, 1, dtype=feats.dtype, device=feats.device)

    # Extract batch indices from coordinates
    b = coords[:, 0]
    # Sum all features per batch index
    pooled.index_add_(0, b, feats)

    # Count active sites per batch sample
    ones = torch.ones((coords.shape[0], 1), dtype=feats.dtype, device=feats.device)
    counts.index_add_(0, b, ones)

    # Compute mean (prevent division by zero with clamp_min)
    pooled = pooled / counts.clamp_min(1.0)
    return pooled


class H5LabelledDataset(Dataset):
    """
    HDF5-backed Dataset for labeled samples.

    Lazily loads pairs (jet_tensor, label) from HDF5 file to minimize memory usage.
    File is opened per-worker to avoid multiprocessing issues.

    Args:
        h5_path (str): Path to HDF5 file containing 'jet' and 'Y' datasets

    Expected HDF5 structure:
        - 'jet': Dataset of shape (N, 125, 125, 8) containing jet tensors
        - 'Y': Dataset of shape (N, 1) containing binary labels
    """
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.file = None
        with h5py.File(h5_path, "r") as f:
            self.length = f["jet"].shape[0]

    def _ensure_open(self):
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self._ensure_open()
        x = self.file["jet"][idx]
        y = self.file["Y"][idx]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long).view(-1)[0]
        return x, y


def labelled_sparse_collate(batch):
    """
    Collate function for batching labeled sparse samples.

    Similar to sparse_collate_hwC but also collates labels.

    Args:
        batch: List of (dense_tensor, label) tuples

    Returns:
        Dictionary with keys:
            - 'coords': Concatenated coordinates, shape (N_total, 3)
            - 'feats': Concatenated features, shape (N_total, C)
            - 'labels': Batch labels, shape (B,)
    """
    coords_list = []
    feats_list = []
    labels = []

    for b, (x, y) in enumerate(batch):
        coords, feats = to_sparse_hwC(x)
        batch_col = torch.full((coords.size(0), 1), b, dtype=torch.long)
        coords_b = torch.cat([batch_col, coords], dim=1)

        coords_list.append(coords_b)
        feats_list.append(feats)
        labels.append(y)

    coords = torch.cat(coords_list, dim=0)
    feats = torch.cat(feats_list, dim=0)
    labels = torch.stack(labels)

    return {
        "coords": coords,
        "feats": feats,
        "labels": labels,
    }