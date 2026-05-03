import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import NearestNeighbors


def generate_TCR_distance_heatmap_from_df(
    df_tr_subset: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    cluster_col: str = "feature_name",
    sample: str = "mySample",
    figsize: tuple = (24, 24),
    output_dir: Optional[str] = None,
    dropna_axis: str = "columns"
):
    """
    Compute the nearest-cluster distance for each cell/row in df_tr_subset using (x, y) coordinates
    and cluster information (feature_name). Group by cluster to compute mean distances, then visualize
    with a hierarchical clustering heatmap (clustermap).

    Also demonstrates:
    1. Manually shrinking and repositioning the color legend.
    2. Aligning row/column dendrograms while keeping heatmap cells square.
    """

    # Use current working directory if output_dir is not specified
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------
    # 1. Prepare coordinates and cluster information
    # ------------------------
    coords = df_tr_subset[[x_col, y_col]].values  # shape: (n_cells, 2)
    clusters = df_tr_subset[cluster_col].astype("category")
    unique_clusters = clusters.cat.categories

    # ------------------------
    # 2. Build empty distance matrix DataFrame
    # ------------------------
    df_nearest_cluster_dist = pd.DataFrame(index=df_tr_subset.index, dtype=float)

    # ------------------------
    # 3. Compute distances for each cluster
    # ------------------------
    for c in unique_clusters:
        mask_c = (clusters == c)
        coords_c = coords[mask_c]
        if coords_c.shape[0] == 0:
            continue
        nbrs_c = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nbrs_c.fit(coords_c)
        dist_c, _ = nbrs_c.kneighbors(coords)
        df_nearest_cluster_dist[c] = dist_c[:, 0]

    # ------------------------
    # 4. Group aggregation (by cluster)
    # ------------------------
    clusters_by_id = pd.Series(data=clusters.values, index=df_tr_subset.index, name=cluster_col)
    df_group_mean = df_nearest_cluster_dist.groupby(clusters_by_id).mean()

    # ------------------------
    # 5. Replace inf/-inf with NaN
    # ------------------------
    df_group_mean = df_group_mean.replace([np.inf, -np.inf], np.nan)

    # ------------------------
    # 6. Drop all-NaN rows/columns according to dropna_axis (optional)
    # ------------------------
    if dropna_axis in ["rows", "index"]:
        df_group_mean = df_group_mean.dropna(axis=0, how="all")
    elif dropna_axis in ["columns", "cols"]:
        df_group_mean = df_group_mean.dropna(axis=1, how="all")
    elif dropna_axis == "both":
        df_group_mean = df_group_mean.dropna(axis=0, how="all").dropna(axis=1, how="all")
    # If "none", skip all-empty dropping

    # ------------------------
    # 7. Drop any row/column containing NaN to ensure all-finite input
    # ------------------------
    df_group_mean = df_group_mean.dropna(axis=0, how="any")
    df_group_mean = df_group_mean.dropna(axis=1, how="any")

    # ------------------------
    # 8. Final check for remaining NaN / inf
    # ------------------------
    if not np.all(np.isfinite(df_group_mean.values)):
        print(f"[Error] Even after cleaning, df_group_mean still has non-finite values for sample={sample}.")
        return

    # If fewer than 2 rows or columns remain after cleaning, clustering is not possible
    if df_group_mean.shape[0] < 2 or df_group_mean.shape[1] < 2:
        print(f"Warning: After cleaning, df_group_mean shape={df_group_mean.shape}, not enough for clustermap.")
        return

    # ============ Core plotting section starts here, including colorbar and alignment changes ============

    # 9. clustermap visualization
    plt.figure(figsize=figsize)
    g = sns.clustermap(
        df_group_mean,
        cmap="RdBu",
        figsize=figsize,
        row_cluster=True,
        col_cluster=True,
        linewidths=0.5,
        annot=False,
        # Set colorbar position and size (as fraction of the entire figure)
        # (x, y, width, height), values typically in [0,1]
        # x=0: 2% from the left edge of the figure
        # y=0.9: 90% up from the bottom of the figure (near the top)
        # width=0.03: colorbar width is 3% of total figure width
        # height=0.09: colorbar height is 9% of total figure height
        cbar_pos=(0, 0.9, 0.03, 0.09),
        cbar_kws={"orientation": "vertical"}  # vertical placement; change to "horizontal" if preferred
    )

    # Set heatmap cells to be square
    g.ax_heatmap.set_aspect("equal")

    # Fix row dendrogram alignment with heatmap in the y direction
    # Get (x0, y0, width, height) for each
    row_dendro_pos = g.ax_row_dendrogram.get_position()
    heatmap_pos = g.ax_heatmap.get_position()
    # Align row dendrogram top and bottom with heatmap
    g.ax_row_dendrogram.set_position([
        row_dendro_pos.x0,
        heatmap_pos.y0,          # use heatmap y0
        row_dendro_pos.width,
        heatmap_pos.height       # use heatmap height
    ])

    # Similarly, fix column dendrogram alignment with heatmap in the x direction (optional)
    col_dendro_pos = g.ax_col_dendrogram.get_position()
    g.ax_col_dendrogram.set_position([
        heatmap_pos.x0,          # use heatmap x0
        col_dendro_pos.y0,
        heatmap_pos.width,       # use heatmap width
        col_dendro_pos.height
    ])

    # Set axis labels
    g.ax_heatmap.set_xlabel("Findee", fontsize=10)
    g.ax_heatmap.set_ylabel("Searcher", fontsize=10)

    # Rotate y-axis labels 0 degrees for readability
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)

    # Set the overall figure title
    g.fig.suptitle(f"TCR_SFplot of {sample}", fontsize=12, y=1.02)

    # 10. Save PDF
    output_file = os.path.join(output_dir, f"TCR_SFplot_of_{sample}.pdf")
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Sample {sample} done. PDF saved to {output_file}")
