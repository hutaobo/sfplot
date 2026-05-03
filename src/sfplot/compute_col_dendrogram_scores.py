import os
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors


def compute_col_dendrogram_scores(
    data: Union['anndata.AnnData', pd.DataFrame],
    input_type: str = "adata",  # "adata" or "dataframe"
    cluster_col: str = "Cluster",
    x_col: str = "x",  # column name for x-axis coordinates in DataFrame
    y_col: str = "y",  # column name for y-axis coordinates in DataFrame
    cell_id_col: Optional[str] = None,  # cell id column name in DataFrame; auto-generated if None
    output_dir: Optional[str] = None,
    method: str = "average"
) -> dict:
    """
    Compute a mean matrix for clustering from input data (anndata object or DataFrame),
    and generate a dendrogram (tree structure) from the column-direction (cluster) linkage.
    Scores are assigned to each split from top to bottom (level 1 gets 1 point, level 2 gets 0.5 points, etc.).

    Parameters:
      data: input data, type anndata.AnnData or pandas.DataFrame
      input_type: "adata" or "dataframe", specifies the data type
      cluster_col: column name containing cluster information
      x_col, y_col: if input is DataFrame, specifies the coordinate column names
      cell_id_col: if input is DataFrame, specifies the cell id column name;
                   if None, a cell id column is auto-generated as "cell_0", "cell_1", ...
      output_dir: output directory for saving intermediate results; defaults to current working directory
      method: linkage method parameter, defaults to "average"

    Returns:
      A dict describing the dendrogram structure, distances, scores, and for leaf nodes
      the actual cluster name (under the "name" key).
    """
    # 0. Handle output directory
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    if input_type == "dataframe":
        # Use DataFrame data
        df = data.copy()
        if cluster_col not in df.columns:
            raise ValueError(f"'{cluster_col}' does not exist in the input DataFrame.")
        clusters = df[cluster_col].astype("category")
        unique_clusters = clusters.cat.categories

        # Check coordinate columns
        if not set([x_col, y_col]).issubset(df.columns):
            raise ValueError(f"The input DataFrame must contain coordinate columns: '{x_col}' and '{y_col}'.")
        coords = df[[x_col, y_col]].values

        # Get cell id: auto-generate if cell_id_col is None
        if cell_id_col is not None:
            if cell_id_col not in df.columns:
                raise ValueError(f"'{cell_id_col}' does not exist in the input DataFrame.")
            cell_ids = df[cell_id_col]
        else:
            # Auto-generate cell id column and add to DataFrame
            df["cell_id"] = ["cell_" + str(i) for i in range(len(df))]
            cell_ids = df["cell_id"]

        # Build a DataFrame with rows as cell_ids and columns as cluster
        df_nearest_cluster_dist = pd.DataFrame(
            index=cell_ids,
            columns=unique_clusters,
            dtype=float
        )

        # For each cluster, use nearest-neighbor model to compute distance from all cells
        for c in unique_clusters:
            mask_c = (clusters == c)
            coords_c = coords[mask_c.values]
            if coords_c.shape[0] == 0:
                df_nearest_cluster_dist.loc[:, c] = np.nan
                continue
            nbrs_c = NearestNeighbors(n_neighbors=1, algorithm="auto")
            nbrs_c.fit(coords_c)
            dist_c, _ = nbrs_c.kneighbors(coords)
            df_nearest_cluster_dist[c] = dist_c[:, 0]

        # Build a Series for groupby mean, with cell_ids as index and cluster as values
        clusters_by_id = pd.Series(
            data=clusters.values,
            index=cell_ids,
            name=cluster_col
        )
        df_group_mean = df_nearest_cluster_dist.groupby(clusters_by_id).mean()

    elif input_type == "adata":
        # Use anndata object
        if cluster_col not in data.obs.columns:
            raise ValueError(f"'{cluster_col}' does not exist in data.obs. Please check the column name.")
        clusters = data.obs[cluster_col].astype("category")
        unique_clusters = clusters.cat.categories

        if "spatial" not in data.obsm:
            raise ValueError("'spatial' coordinate info does not exist in data.obsm. Please ensure spatial coordinates are present.")
        coords = data.obsm["spatial"]

        if cell_id_col not in data.obs.columns:
            raise ValueError(f"'{cell_id_col}' does not exist in data.obs. Please ensure the data contains this column.")
        cell_ids = data.obs[cell_id_col]

        df_nearest_cluster_dist = pd.DataFrame(
            index=cell_ids,
            columns=unique_clusters,
            dtype=float
        )

        for c in unique_clusters:
            mask_c = (clusters == c)
            coords_c = coords[mask_c]
            if coords_c.shape[0] == 0:
                df_nearest_cluster_dist.loc[:, c] = np.nan
                continue
            nbrs_c = NearestNeighbors(n_neighbors=1, algorithm="auto")
            nbrs_c.fit(coords_c)
            dist_c, _ = nbrs_c.kneighbors(coords)
            df_nearest_cluster_dist[c] = dist_c[:, 0]

        # Save results to data.uns
        data.uns["nearest_cluster_dist"] = df_nearest_cluster_dist

        clusters_by_id = pd.Series(
            data=clusters.values,
            index=cell_ids,
            name=cluster_col
        )
        df_group_mean = df_nearest_cluster_dist.groupby(clusters_by_id).mean()
    else:
        raise ValueError("input_type must be 'adata' or 'dataframe'.")

    # Process mean matrix, drop all-NaN columns
    df_group_mean_clean = df_group_mean.dropna(axis=1, how="all")
    if df_group_mean_clean.empty:
        raise ValueError("Data is empty after processing, please check input data.")

    # Build leaf name mapping: columns of df_group_mean_clean are actual cluster names
    leaf_names = {i: name for i, name in enumerate(df_group_mean_clean.columns)}

    # Compute column-direction (cluster) linkage (cluster the transposed data)
    col_linkage = linkage(df_group_mean_clean.T, method=method)

    # Convert linkage to tree structure
    root, _ = to_tree(col_linkage, rd=True)

    # Recursive function: assign scores to each split (score halves with each level down)
    def assign_score(node, level=1):
        if node.left is not None and node.right is not None:
            score = 1 / (2 ** (level - 1))
            node.left.score = score
            node.right.score = score
            assign_score(node.left, level + 1)
            assign_score(node.right, level + 1)

    assign_score(root, level=1)

    # Recursive function: convert tree to dict; for leaf nodes, also store actual cluster name under "name"
    def tree_to_dict(node):
        if node.left is None and node.right is None:
            return {
                "id": node.id,
                "name": leaf_names.get(node.id, None),
                "dist": node.dist,
                "score": getattr(node, "score", None)
            }
        else:
            node_dict = {
                "id": node.id,
                "dist": node.dist,
                "score": getattr(node, "score", None)
            }
            if node.left is not None:
                node_dict["left"] = tree_to_dict(node.left)
            if node.right is not None:
                node_dict["right"] = tree_to_dict(node.right)
            return node_dict

    dendrogram_dict = tree_to_dict(root)
    return dendrogram_dict

# Usage example:
# For DataFrame input without providing cell_id_col (auto-generate cell id):
#
# df = pd.DataFrame({
#     "Location_Center_X": [0.1, 0.3, 0.2, 0.8, 0.85],
#     "Location_Center_Y": [1.2, 1.3, 1.1, 0.5, 0.45],
#     "Gene": ["A", "A", "B", "B", "B"]
# })
# dendro_structure = compute_col_dendrogram_scores(
#     df,
#     input_type="dataframe",
#     cluster_col="Gene",
#     x_col="Location_Center_X",
#     y_col="Location_Center_Y"
# )
# print(dendro_structure)
