import numpy as np
import pandas as pd
from typing import Optional
from typing import Optional, Tuple
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.neighbors import NearestNeighbors


def calculate_gene_distance_matrix_ewnn(expression: pd.DataFrame,
                                        coordinates: pd.DataFrame,
                                        threshold: float = 0.0,
                                        z: Optional[pd.Series] = None,
                                        memory_limit_gb: float = 300.0,
                                        batch_size: Optional[int] = None) -> pd.DataFrame:
    """
    Memory-optimized implementation for computing a directed inter-gene distance matrix
    based on an expression dual-weight minimum-cost model.
    For each gene pair (i, j), a weighted cost C(s,t) is computed between expression point sets
    S_i and S_j, and D_{ij} is the minimum of that cost. The formula is:
        C(s,t) = d(s,t) / (E_i(s) * E_j(t))^α + ε，
    where d(s,t) is the Euclidean distance, E_i(s) and E_j(t) are the expression values of genes i and j
    at those points. alpha defaults to 0.5 (soft expression weighting), epsilon is a tiny constant to
    avoid division by zero (e.g., 1e-8). Then D_{ij} = min_{s in S_i, t in S_j} C(s,t). The matrix is asymmetric.

    Memory optimization: only expression points above threshold are considered. Genes with no expression
    points keep their row/column as NaN. The output matrix size is estimated from gene count and a
    memory-mapped temp file is used when it exceeds memory_limit_gb. batch_size can specify the chunk size.

    Parameters:
        expression: pd.DataFrame
            Row index is the spatial spot identifier; columns are gene names; values are expression levels.
        coordinates: pd.DataFrame
            Coordinate list (x, y) corresponding to expression rows; row order must match expression.
        threshold: float, default 0.0
            Expression threshold. Only spots with expression above this value are included.
        z: Optional[pd.Series], default None
            Optional z coordinate column (e.g., for 3D spatial data). Extends coordinates to (x, y, z).
        memory_limit_gb: float, default 300.0
            Maximum memory (GB) allowed for the output distance matrix. Uses disk temp file if exceeded.
        batch_size: Optional[int], default None
            Number of target gene expression points per batch. Auto-selected if None.

    Returns:
        pd.DataFrame: Distance matrix (float64) with gene names as index and columns representing
            directed inter-gene distances. Non-computable positions are NaN.
    """
    import numpy as np
    import pandas as pd
    from scipy.spatial.distance import cdist

    genes = list(expression.columns)
    n_genes = len(genes)
    # Estimate final distance matrix size from gene count to decide whether to use disk temp storage
    total_bytes = n_genes * n_genes * 8  # float64: 8 bytes per element
    memory_limit_bytes = memory_limit_gb * (1024 ** 3)
    use_memmap = total_bytes > memory_limit_bytes

    if use_memmap:
        # Create a temp memory-mapped file for the result matrix to avoid excessive memory usage
        import tempfile
        tmp = tempfile.NamedTemporaryFile(prefix="gene_dist_ewnn_", suffix=".dat", delete=False)
        tmp_filename = tmp.name
        tmp.close()
        dist_matrix_arr = np.memmap(tmp_filename, dtype='float64', mode='w+', shape=(n_genes, n_genes))
        dist_matrix_arr[:] = np.nan  # initialize with NaN
    else:
        # Initialize result matrix in memory
        dist_matrix_arr = np.full((n_genes, n_genes), np.nan, dtype=np.float64)

    # Pre-compute valid expression masks and value arrays for each gene for efficiency
    gene_masks = {gene: (expression[gene].values > threshold) for gene in genes}
    gene_expr_values = {gene: expression[gene].values for gene in genes}

    # Build coordinate matrix (include z if provided)
    coords = coordinates.values
    if z is not None:
        coords = np.hstack([coords, np.asarray(z).reshape(-1, 1)])
    n_spots = coords.shape[0]

    # Set computation parameters
    alpha = 0.5
    epsilon = 1e-8
    max_pairs = 1e7  # max point pairs per computation to control memory usage

    # Compute distance matrix: for each source gene i (row), compute distance to target gene j (column)
    for i_idx, gene_i in enumerate(genes):
        mask_i = gene_masks[gene_i]
        if not mask_i.any():
            continue  # gene i has no expression, entire row remains NaN
        # Extract expression coordinates and values for gene i
        coords_i = coords[mask_i]
        expr_i = gene_expr_values[gene_i][mask_i]
        # Compute alpha power of gene i expression values for weighting
        E_i_alpha = np.power(expr_i, alpha)
        num_i = coords_i.shape[0]

        for j_idx, gene_j in enumerate(genes):
            mask_j = gene_masks[gene_j]
            if not mask_j.any():
                continue  # gene j has no expression, entire column remains NaN
            coords_j = coords[mask_j]
            expr_j = gene_expr_values[gene_j][mask_j]
            num_j = coords_j.shape[0]

            # Determine whether to compute in batches to reduce peak memory
            if batch_size is not None:
                chunk_size = min(batch_size, num_j)
            elif num_i * num_j > max_pairs:
                chunk_size = int(max_pairs // num_i) or 1
            else:
                chunk_size = None

            if chunk_size and chunk_size < num_j:
                # Process target gene j expression points in chunks, accumulating minimum cost
                min_cost_val = np.inf
                for start in range(0, num_j, chunk_size):
                    end = min(num_j, start + chunk_size)
                    sub_coords_j = coords_j[start:end]
                    sub_expr_j = expr_j[start:end]
                    # Compute distance matrix from all gene i expression points to this batch of gene j points
                    sub_dist = cdist(coords_i, sub_coords_j)  # shape: (num_i, end-start)
                    # Apply dual expression weighting
                    sub_dist /= E_i_alpha[:, None]  # divide each row by gene i expression^alpha
                    sub_E_j_alpha = np.power(sub_expr_j, alpha)
                    sub_dist /= sub_E_j_alpha[None, :]  # divide each column by gene j expression^alpha
                    sub_dist += epsilon  # add tiny constant for numerical stability
                    # Update current minimum cost
                    sub_min = np.min(sub_dist)
                    if sub_min < min_cost_val:
                        min_cost_val = sub_min
                dist_matrix_arr[i_idx, j_idx] = min_cost_val
            else:
                # Compute all point-pair costs in one shot
                dist_matrix = cdist(coords_i, coords_j)  # shape: (num_i, num_j)
                dist_matrix /= E_i_alpha[:, None]  # weight by gene i expression
                E_j_alpha = np.power(expr_j, alpha)
                dist_matrix /= E_j_alpha[None, :]  # weight by gene j expression
                dist_matrix += epsilon  # add tiny constant to avoid zero distance
                # Take minimum cost as D_{ij}
                dist_matrix_arr[i_idx, j_idx] = np.min(dist_matrix)

    # Convert result to DataFrame with index and columns matching input gene order
    dist_matrix_df = pd.DataFrame(dist_matrix_arr, index=genes, columns=genes)
    return dist_matrix_df


def calculate_gene_distance_matrix_wmda(expression: pd.DataFrame,
                                       coordinates: pd.DataFrame,
                                       threshold: float = 0.0,
                                       z: Optional[pd.Series] = None,
                                       memory_limit_gb: float = 300.0) -> pd.DataFrame:
    """
    Memory-optimized version of the directed inter-gene distance matrix computation
    based on expression-weighted distribution centroids (WMDA method).
    Reduces memory usage through chunked computation and temporary storage when necessary.

    Additional parameter:
        memory_limit_gb: float, default 300.0
            Memory usage limit (GB). Uses a disk memory-mapped file for results when the output
            matrix is too large to fit in memory.
    All other parameters and return values are the same as the original function.
    """
    genes = list(expression.columns)
    n_genes = len(genes)
    total_bytes = n_genes * n_genes * 8
    memory_limit_bytes = memory_limit_gb * (1024 ** 3)
    use_memmap = total_bytes > memory_limit_bytes

    if use_memmap:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(prefix="gene_dist_wmda_", suffix=".dat", delete=False)
        tmp_filename = tmp.name
        tmp.close()
        dist_matrix_arr = np.memmap(tmp_filename, dtype='float64', mode='w+', shape=(n_genes, n_genes))
        dist_matrix_arr[:] = np.nan
    else:
        dist_matrix_arr = np.full((n_genes, n_genes), np.nan, dtype=np.float64)

    coords = coordinates.values
    if z is not None:
        coords = np.hstack([coords, np.asarray(z).reshape(-1, 1)])

    # Pre-compute weighted centroid coordinates for each gene
    centers = {}
    for gene in genes:
        mask = expression[gene].values > threshold
        if not mask.any():
            centers[gene] = None
        else:
            sub_coords = coords[mask]
            sub_expr = expression.loc[mask, gene].values
            # Compute weighted average over (x, y, [z]) dimensions to get the spatial centroid for this gene
            cx = np.average(sub_coords[:, 0], weights=sub_expr)
            cy = np.average(sub_coords[:, 1], weights=sub_expr)
            if z is not None:
                cz = np.average(sub_coords[:, 2], weights=sub_expr)
                centers[gene] = np.array([cx, cy, cz])
            else:
                centers[gene] = np.array([cx, cy])

    # Compute distance column per target gene centroid
    n_spots = coords.shape[0]
    for j_idx, gene_j in enumerate(genes):
        center_j = centers.get(gene_j)
        if center_j is None:
            continue  # gene j has no expression, entire column NaN
        # Compute distances from all spots to centroid j
        diff = coords - center_j  # shape: (n_spots, 2 or 3)
        dist_to_center_j = np.linalg.norm(diff, axis=1)  # Euclidean distance from each spot to centroid

        # Compute expression-weighted average distance from each source gene i to this centroid
        for i_idx, gene_i in enumerate(genes):
            mask_i = expression[gene_i].values > threshold
            if not mask_i.any():
                continue  # gene i has no expression, entire row NaN
            weights_i = expression.loc[mask_i, gene_i].values
            dist_matrix_arr[i_idx, j_idx] = np.average(dist_to_center_j[mask_i], weights=weights_i)

    dist_matrix_df = pd.DataFrame(dist_matrix_arr, index=genes, columns=genes)
    return dist_matrix_df


def compute_cophenetic_distances_from_group_mean_matrix(
    df_group_mean_clean: pd.DataFrame,
    method: str = "average",
    metric: str = "euclidean",
    normalize: bool = True,
    show_corr: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Based on a cluster×cluster "Searcher→Findee" average distance matrix, compute cophenetic distance
    matrices in both row and column directions. Returns (row_coph_df, col_coph_df), each independently
    normalized to [0,1] (can be disabled).

    Parameters
    ----------
    df_group_mean_clean : pd.DataFrame
        Average distance matrix with Searcher (source types) as rows and Findee (target types) as columns.
        Must be numeric. Recommend cleaning NaN values beforehand; remaining NaNs are filled with column means.
    method : str, default "average"
        Linkage method for hierarchical clustering ("average", "single", "complete", "ward", etc.).
    metric : str, default "euclidean"
        Distance metric for pairwise row/column vectors, passed to `scipy.spatial.distance.pdist`.
    normalize : bool, default True
        Whether to linearly scale each output matrix independently to [0,1].
    show_corr : bool, default False
        Whether to print the cophenetic correlation coefficient for rows and columns (measures tree fidelity).

    Returns
    -------
    row_coph_df : pd.DataFrame
        Cophenetic distance matrix in the row direction (index and columns are df_group_mean_clean.index).
        If normalize=True, independently scaled to [0,1] with diagonal reset to 0.
    col_coph_df : pd.DataFrame
        Cophenetic distance matrix in the column direction (index and columns are df_group_mean_clean.columns).
        If normalize=True, independently scaled to [0,1] with diagonal reset to 0.

    Notes
    -----
    - When there are 0 or 1 rows (or columns), clustering is not possible; returns an all-zero square matrix.
    - `cophenet` requires first computing `pdist(X)` (pairwise distances between observations) and the linkage.
    - Corresponds to the second half of `compute_cophenetic_distances_from_df`: that function computes
      directed nearest-neighbor means from point level to form a cluster×cluster matrix, then does
      hierarchical clustering for cophenetic distances; this function starts directly from that matrix.
    """
    if not isinstance(df_group_mean_clean, pd.DataFrame):
        raise TypeError("df_group_mean_clean must be a pandas.DataFrame.")

    # Ensure all values are numeric; make a copy to avoid in-place modification
    M = df_group_mean_clean.copy()
    # Try converting columns to numeric; use astype(float) for strict conversion
    try:
        M = M.astype(float)
    except Exception as e:
        raise ValueError("df_group_mean_clean must be a numeric matrix and could not be converted to float.") from e

    # If NaN values remain: fill with column mean; if column is all NaN, use global mean; if global is NaN, use 0
    if M.isna().any().any():
        col_means = M.mean(axis=0)
        global_mean = np.nanmean(M.values)
        col_means = col_means.fillna(global_mean if not np.isnan(global_mean) else 0.0)
        M = M.fillna(col_means)

    # ---------- A utility normalization function ----------
    def _normalize_to_01(D: pd.DataFrame) -> pd.DataFrame:
        vmin = D.values.min()
        vmax = D.values.max()
        if vmax <= vmin:
            out = pd.DataFrame(np.zeros_like(D.values), index=D.index, columns=D.columns)
        else:
            out = (D - vmin) / (vmax - vmin)
        # Set diagonal to 0 (numerical stability)
        np.fill_diagonal(out.values, 0.0)
        return out

    # ---------- Row direction (each row vector as an observation) ----------
    row_labels = M.index.to_list()
    n_row = len(row_labels)
    if n_row >= 2:
        # Row vectors: shape = (n_row, n_col)
        X_row = M.values
        Y_row = pdist(X_row, metric=metric)     # condensed distance vector
        Z_row = linkage(Y_row, method=method)   # hierarchical clustering
        c_row, coph_row = cophenet(Z_row, Y_row)
        if show_corr:
            print(f"[Row] Cophenetic correlation: {c_row:.4f}")
        row_coph = squareform(coph_row)
        row_coph_df = pd.DataFrame(row_coph, index=row_labels, columns=row_labels)
    else:
        # Degenerate case: 0 or 1 rows, return all-zero square matrix
        row_coph_df = pd.DataFrame(np.zeros((n_row, n_row)), index=row_labels, columns=row_labels)

    # ---------- Column direction (each column vector as an observation) ----------
    col_labels = M.columns.to_list()
    n_col = len(col_labels)
    if n_col >= 2:
        # Column vectors: shape = (n_col, n_row) — use rows of M.T as observations
        X_col = M.values.T
        Y_col = pdist(X_col, metric=metric)
        Z_col = linkage(Y_col, method=method)
        c_col, coph_col = cophenet(Z_col, Y_col)
        if show_corr:
            print(f"[Col] Cophenetic correlation: {c_col:.4f}")
        col_coph = squareform(coph_col)
        col_coph_df = pd.DataFrame(col_coph, index=col_labels, columns=col_labels)
    else:
        col_coph_df = pd.DataFrame(np.zeros((n_col, n_col)), index=col_labels, columns=col_labels)

    # ---------- Optional: normalize separately to [0,1] ----------
    if normalize:
        row_coph_df = _normalize_to_01(row_coph_df)
        col_coph_df = _normalize_to_01(col_coph_df)

    return row_coph_df, col_coph_df


import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform


def _get_gene_spots_and_weights(expression_df, genes, threshold=0.0):
    """
    For each gene:
      - Find row indices of spots with expression > threshold
      - Use corresponding expression values as weights
    Supports both dense and pandas sparse formats.
    """
    gene_spots = {}
    gene_weights = {}

    for g in genes:
        col = expression_df[g]

        if pd.api.types.is_sparse(col.dtype):
            coo = col.sparse.to_coo()
            mask = coo.data > threshold
            spots_idx = coo.row[mask]
            weights = coo.data[mask]
        else:
            values = col.values
            mask = values > threshold
            spots_idx = np.where(mask)[0]
            weights = values[mask]

        gene_spots[g] = spots_idx.astype(int)
        gene_weights[g] = weights.astype(float)

    return gene_spots, gene_weights


def _weighted_quantile(values, weights, q):
    """
    Compute the (optionally weighted) quantile q in (0,1) of a 1D array.
    """
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return np.nan

    if weights is None:
        return float(np.quantile(values, q))

    weights = np.asarray(weights, dtype=float)
    if np.all(weights <= 0):
        return float(np.quantile(values, q))

    order = np.argsort(values)
    v = values[order]
    w = weights[order]

    cum_w = np.cumsum(w)
    if cum_w[-1] == 0:
        return float(np.quantile(values, q))

    cum_w /= cum_w[-1]
    idx = np.searchsorted(cum_w, q)
    idx = min(idx, v.size - 1)
    return float(v[idx])


def _aggregate_distances(dists, weights, agg="quantile", q=0.9, weight_by_expression=True):
    """
    Aggregate a series of nearest-neighbor distances into a single scalar.
    """
    dists = np.asarray(dists, dtype=float)
    if dists.size == 0:
        return np.nan

    if not weight_by_expression:
        weights = None

    if agg == "mean":
        if weights is None:
            return float(dists.mean())
        else:
            return float(np.average(dists, weights=weights))
    elif agg == "median":
        return _weighted_quantile(dists, weights, 0.5)
    elif agg == "quantile":
        return _weighted_quantile(dists, weights, q)
    else:
        raise ValueError(f"Unknown agg: {agg}")


def _estimate_spot_nn_scale(coord_array):
    """
    Estimate the "typical nearest-neighbor distance" using a global KDTree,
    similar to the Visium grid spacing, used for normalization.
    """
    tree = cKDTree(coord_array)
    # k=2: first is self, second is nearest neighbor
    dists, _ = tree.query(coord_array, k=2)
    nn_dists = dists[:, 1]
    nn_dists = nn_dists[np.isfinite(nn_dists)]
    if nn_dists.size == 0:
        return 1.0
    scale = np.median(nn_dists)
    return float(scale if scale > 0 else 1.0)


def calculate_gene_distance_matrix_visium(
    expression_df: pd.DataFrame,
    coords_df: pd.DataFrame,
    genes=None,
    threshold: float = 0.0,
    min_spots: int = 20,
    min_total_expr: float = 0.0,
    agg: str = "quantile",     # "mean", "median", "quantile"
    quantile: float = 0.9,     # used when agg="quantile"
    weight_by_expression: bool = True,
    symmetric: str = "min"     # "none", "min", "mean"
) -> pd.DataFrame:
    """
    Gene-gene spatial distance matrix suitable for Visium data (COSTE-style).

    - Uses expression-weighted nearest neighbors (directed)
    - Aggregates with a robust quantile (default 0.9 quantile)
    - Normalizes distances by the spot nearest-neighbor scale
    """
    if genes is None:
        genes = list(expression_df.columns)
    else:
        genes = [g for g in genes if g in expression_df.columns]

    # Ensure coords are aligned with expression rows
    coords = coords_df.loc[expression_df.index]
    coord_array = coords.values  # (n_spots, 2)

    # Estimate the typical nearest-neighbor scale of the Visium grid for normalization
    nn_scale = _estimate_spot_nn_scale(coord_array)

    # Pre-compute expression spots and weights for each gene
    gene_spots, gene_weights = _get_gene_spots_and_weights(
        expression_df, genes, threshold=threshold
    )

    # Initialize result matrix
    dist_matrix = pd.DataFrame(
        np.nan, index=genes, columns=genes, dtype=float
    )

    # Build a KDTree for each target gene
    gene_trees = {}
    for g in genes:
        spots_j = gene_spots[g]
        if spots_j.size == 0:
            gene_trees[g] = None
        else:
            target_coords = coord_array[spots_j]
            gene_trees[g] = cKDTree(target_coords)

    # Main loop
    for gi in genes:
        spots_i = gene_spots[gi]
        weights_i = gene_weights[gi]

        if spots_i.size < min_spots or weights_i.sum() <= min_total_expr:
            # Gene is too sparse: entire row NaN
            dist_matrix.loc[gi, :] = np.nan
            continue

        source_coords = coord_array[spots_i]

        for gj in genes:
            if gi == gj:
                dist_matrix.loc[gi, gj] = 0.0
                continue

            spots_j = gene_spots[gj]
            tree_j = gene_trees[gj]

            # Target gene is too sparse or has no expression: treat as essentially unreachable
            if spots_j.size < min_spots or tree_j is None:
                dist_matrix.loc[gi, gj] = np.inf
                continue

            # Nearest-neighbor query: for each point in gi, find the nearest point in gj
            dists, _ = tree_j.query(source_coords, k=1)

            directed_raw = _aggregate_distances(
                dists=dists,
                weights=weights_i,
                agg=agg,
                q=quantile,
                weight_by_expression=weight_by_expression
            )

            # Normalize by global nearest-neighbor scale to ensure comparability across datasets/resolutions
            directed_norm = directed_raw / nn_scale
            dist_matrix.loc[gi, gj] = directed_norm

    # Optional symmetrization
    if symmetric in ("min", "mean"):
        arr = dist_matrix.values
        n = arr.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                dij = arr[i, j]
                dji = arr[j, i]
                if symmetric == "min":
                    val = np.nanmin([dij, dji])
                else:  # "mean"
                    val = np.nanmean([dij, dji])
                arr[i, j] = arr[j, i] = val
        np.fill_diagonal(arr, 0.0)
        dist_matrix.iloc[:, :] = arr

    return dist_matrix
