import numpy as np
import pandas as pd
import torch
from typing import Optional


def calculate_gene_distance_matrix_wmda_gpu(
    expression: pd.DataFrame,
    coordinates: pd.DataFrame,
    threshold: float = 0.0,
    z: Optional[pd.Series] = None,
    device: str = "cuda"
) -> pd.DataFrame:
    """
    GPU implementation of the expression-distribution centroid (WMDA) method for computing
    a directed inter-gene distance matrix.

    Parameters:
        expression (pd.DataFrame): Spatial expression matrix; rows are locations, columns are genes.
        coordinates (pd.DataFrame): Spatial coordinate DataFrame (with x, y columns; use z for z dimension).
        threshold (float): Expression threshold; values <= threshold are treated as no expression. Default 0.0.
        z (pd.Series/np.ndarray, optional): z coordinate for each location.
        device (str): Device to use ("cuda", "cpu", etc.). Default "cuda".

    Returns:
        pd.DataFrame: Directed average centroid distance matrix (gene×gene DataFrame); NaN for no-expression cases.
    """
    genes = list(expression.columns)
    n_genes = len(genes)
    # Build coordinate and centroid tensors
    if z is not None:
        coords_arr = np.hstack([coordinates.values, np.asarray(z).reshape(-1, 1)])
    else:
        coords_arr = coordinates.values
    coords_tensor = torch.from_numpy(coords_arr).to(device=device, dtype=torch.float32)
    # Pre-compute centroid coordinates for all genes (CPU is fine due to low overhead)
    centers = {}
    for gene in genes:
        mask = expression[gene].values > threshold
        if not mask.any():
            centers[gene] = None
        else:
            sub_coords = coords_tensor[mask]  # filter coordinates directly on GPU
            sub_expr = torch.from_numpy(expression.loc[mask, gene].values).to(device=device, dtype=torch.float32)
            # Compute weighted centroid (normalize weights or convert type for numerical stability if needed)
            total_w = sub_expr.sum()
            if float(total_w) == 0.0:
                centers[gene] = None
            else:
                center = (sub_coords * sub_expr[:, None]).sum(dim=0) / total_w
                centers[gene] = center  # store centroid coordinate as Tensor
    # Initialize result matrix
    result = np.full((n_genes, n_genes), np.nan, dtype=float)
    # Compute distance matrix
    for i_idx, gene_i in enumerate(genes):
        mask_i = expression[gene_i].values > threshold
        if not mask_i.any():
            continue  # gene i has no expression
        sub_coords_i = coords_tensor[mask_i]  # gene_i expression coordinates (GPU tensor)
        sub_expr_i = torch.from_numpy(expression.loc[mask_i, gene_i].values).to(device=device, dtype=torch.float32)
        for j_idx, gene_j in enumerate(genes):
            center_j = centers.get(gene_j)
            if center_j is None:
                result[i_idx, j_idx] = np.nan
            else:
                # Compute distance tensor from each gene_i point to gene_j centroid
                # center_j is a GPU tensor; sub_coords_i is an (Ni, dims) tensor
                dists = torch.norm(sub_coords_i - center_j, dim=1)  # distance from each point to centroid (Ni,)
                # Compute expression-weighted average distance for gene_i
                avg_dist = (dists * sub_expr_i).sum() / sub_expr_i.sum()
                result[i_idx, j_idx] = avg_dist.item()
    dist_df = pd.DataFrame(result, index=genes, columns=genes)
    return dist_df


def calculate_gene_distance_matrix_ewnn_gpu(
    expression: pd.DataFrame,
    coordinates: pd.DataFrame,
    threshold: float = 0.0,
    z: Optional[pd.Series] = None,
    max_memory_gb: float = 300.0,
    device: str = "cuda"
) -> pd.DataFrame:
    """
    GPU implementation of the expression-weighted nearest-neighbor (EWNN) method for computing
    a directed inter-gene distance matrix.

    Parameters:
        expression (pd.DataFrame): Spatial expression matrix; rows are locations, columns are genes.
        coordinates (pd.DataFrame): Spatial coordinate DataFrame with x, y columns (optional z).
        threshold (float): Expression threshold; values <= threshold are treated as no expression. Default 0.0.
        z (pd.Series or np.ndarray, optional): z coordinate for each location.
        max_memory_gb (float): GPU memory limit (GB), used to determine batch size. Default 300.0.
        device (str): Device for computation, e.g. "cuda" or "cuda:0". Default "cuda".

    Returns:
        pd.DataFrame: Directed average nearest-neighbor distance matrix (gene×gene DataFrame); NaN marks invalid distances.
    """
    genes = list(expression.columns)
    n_genes = len(genes)
    # Build coordinate tensor
    if z is not None:
        coords_arr = np.hstack([coordinates.values, np.asarray(z).reshape(-1, 1)])
    else:
        coords_arr = coordinates.values
    coords_tensor = torch.from_numpy(coords_arr).to(device=device, dtype=torch.float32)

    # Pre-cache expression location indices for each gene (numpy arrays) for slicing
    gene_indices = {
        gene: np.where(expression[gene].values > threshold)[0]
        for gene in genes
    }
    # Initialize result matrix (numpy first, convert to DataFrame at the end)
    result = np.full((n_genes, n_genes), np.nan, dtype=float)

    # Compute directed distance matrix
    for i_idx, gene_i in enumerate(genes):
        idx_i = gene_indices[gene_i]
        if idx_i.size == 0:
            continue  # gene i has no expression, skip entire row
        # Transfer source gene i coordinates and expression weights to GPU
        coords_i = coords_tensor[idx_i]  # shape: (Ni, dims)
        weights_i = torch.from_numpy(expression.iloc[idx_i, i_idx].values).to(device=device, dtype=torch.float32)
        Ni = coords_i.shape[0]
        for j_idx, gene_j in enumerate(genes):
            idx_j = gene_indices[gene_j]
            if idx_j.size == 0:
                # gene j has no expression
                result[i_idx, j_idx] = np.nan
                continue
            coords_j = coords_tensor[idx_j]  # (Nj, dims) on GPU
            Nj = coords_j.shape[0]
            # Compute distance matrix size and decide on batching (float32: 4 bytes per distance)
            # If Ni*Nj is too large, split into multiple computations
            max_elements = int((max_memory_gb * (1024 ** 3)) / 4)  # maximum number of distance elements
            if Ni * Nj <= max_elements:
                # Compute distances and find nearest neighbors in one shot
                dist_matrix = torch.cdist(coords_i, coords_j, p=2.0)  # produces (Ni, Nj) distance matrix
                min_dists, _ = torch.min(dist_matrix, dim=1)  # nearest j-point distance for each i-point (Ni,)
                # Compute weighted average distance
                avg_dist = (min_dists * weights_i).sum() / weights_i.sum()
                result[i_idx, j_idx] = avg_dist.item()
            else:
                # Batched computation to avoid GPU out-of-memory
                min_dists = torch.full((Ni,), float('inf'), device=device)
                chunk_size = max(1, max_elements // Ni)  # infer allowed Nj per batch from Ni
                for start in range(0, Nj, chunk_size):
                    end = min(Nj, start + chunk_size)
                    dist_chunk = torch.cdist(coords_i, coords_j[start:end], p=2.0)
                    # Nearest distance for each i-point in current batch
                    batch_min, _ = torch.min(dist_chunk, dim=1)
                    # Update global nearest distances
                    min_dists = torch.minimum(min_dists, batch_min)
                    # Free GPU memory (explicitly delete local tensors)
                    del dist_chunk, batch_min
                    torch.cuda.empty_cache()
                avg_dist = (min_dists * weights_i).sum() / weights_i.sum()
                result[i_idx, j_idx] = avg_dist.item()
    # Transfer back to CPU and build DataFrame
    dist_df = pd.DataFrame(result, index=genes, columns=genes)
    return dist_df
