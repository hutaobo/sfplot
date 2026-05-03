"""Main module."""

import numpy as np
from sklearn.neighbors import NearestNeighbors

def split_B_by_distance_to_A(
    adata,
    cluster_col="cluster",
    cluster_A="A",
    cluster_B="B",
    threshold=None
):
    """
    Split cluster B into B_close and B_far based on distance to cluster A.
    Other clusters remain unchanged with their original labels.

    Parameters
    ----------
    adata : AnnData
        Must contain the following:
         - adata.obs containing column cluster_col and cell_id (unique cell identifier)
         - adata.obsm["spatial"] storing spatial coordinates (n_cells, 2) or (n_cells, 3)
    cluster_col : str, optional
        Column name for cell cluster labels. Defaults to "cluster".
    cluster_A : str, optional
        Name of the reference cluster. Defaults to "A".
    cluster_B : str, optional
        Name of the cluster to split. Defaults to "B".
    threshold : float, optional
        If specified, this value is used to separate B_close from B_far;
        if not specified, the median distance of B cells to A is used.

    Returns
    ----------
    adata : AnnData
        A new column new_cluster_col is added to adata.obs, where:
         - Cells originally in A remain labeled A
         - Cells originally in B are split into B_close and B_far based on the threshold
         - Other clusters remain unchanged
    """

    # ------------------------------
    # 1. Basic validation
    # ------------------------------
    if cluster_col not in adata.obs.columns:
        raise ValueError(f"'{cluster_col}' does not exist in adata.obs. Please check the column name.")

    if "cell_id" not in adata.obs.columns:
        raise ValueError("'cell_id' column does not exist in adata.obs. Please ensure the data contains 'cell_id'.")

    if "spatial" not in adata.obsm:
        raise ValueError("'spatial' coordinate info does not exist in adata.obsm. Please ensure the data contains spatial coordinates.")

    clusters = adata.obs[cluster_col].astype("category")
    unique_clusters = clusters.cat.categories
    coords = adata.obsm["spatial"]

    if cluster_A not in unique_clusters:
        raise ValueError(f"'{cluster_A}' is not among the existing clusters: {list(unique_clusters)}")
    if cluster_B not in unique_clusters:
        raise ValueError(f"'{cluster_B}' is not among the existing clusters: {list(unique_clusters)}")

    # ------------------------------
    # 2. Build nearest-neighbor model for cluster A only
    # ------------------------------
    mask_A = (clusters == cluster_A)
    coords_A = coords[mask_A]
    if coords_A.shape[0] == 0:
        raise ValueError(f"cluster_A='{cluster_A}' contains no cells, cannot compute distances.")

    nbrs_A = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nbrs_A.fit(coords_A)

    # ------------------------------
    # 3. Compute nearest distances from B cells to A only
    # ------------------------------
    b_mask = (clusters == cluster_B)
    coords_B = coords[b_mask]
    if coords_B.shape[0] == 0:
        raise ValueError(f"cluster_B='{cluster_B}' contains no cells, cannot split.")

    dist_to_A_for_B, _ = nbrs_A.kneighbors(coords_B)
    dist_to_A_for_B = dist_to_A_for_B[:, 0]  # shape: (num_B_cells, )

    # ------------------------------
    # 4. Determine threshold to distinguish B_close from B_far
    # ------------------------------
    if threshold is None:
        threshold = np.median(dist_to_A_for_B)

    # ------------------------------
    # 5. Generate new cluster column
    # ------------------------------
    new_cluster_col = f"split_{cluster_B}_by_{cluster_A}_in_{cluster_col}"
    adata.obs[new_cluster_col] = adata.obs[cluster_col].astype(str)

    # Use actual row index labels, not integer positions
    b_indices = adata.obs.index[b_mask]  # DataFrame row index labels for B cells

    # dist_to_A_for_B < threshold identifies which B cells are close/far
    b_close_mask_in_B = (dist_to_A_for_B < threshold)
    b_far_mask_in_B = ~b_close_mask_in_B

    # Get row index labels for close/far cells separately
    b_close_indices = b_indices[b_close_mask_in_B]
    b_far_indices = b_indices[b_far_mask_in_B]

    # Then assign using .loc[...] with index labels
    adata.obs.loc[b_close_indices, new_cluster_col] = f"{cluster_B}_close"
    adata.obs.loc[b_far_indices, new_cluster_col] = f"{cluster_B}_far"

    print("The new cluster col is:", new_cluster_col)

    return adata
