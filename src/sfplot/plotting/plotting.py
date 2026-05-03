# sfplot/plotting.py

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

from ..preprocessing.data_processing import load_xenium_data


def generate_cluster_distance_heatmap_from_path(
    base_path: str,
    sample: str,
    figsize: tuple = (8, 8),
    output_dir: Optional[str] = None,
    show_dendrogram: bool = True  # new parameter: whether to draw dendrogram (default: True)
):
    """
    Generate and save a distance heatmap from each cell cluster to its nearest cluster center.

    Parameters:
    ----------
    base_path : str
        Base path where data is stored.
    sample : str
        Sample name used to specify the data folder.
    output_dir : Optional[str]
        Output directory for the PDF file. Defaults to current working directory.

    Returns:
    -------
    None
    """
    # Set output directory
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # Construct path and read data
    folder = os.path.join(base_path, sample)
    adata = load_xenium_data(folder)

    # 1. Extract coordinates and cluster information
    coords = adata.obsm["spatial"]  # (n_cells, 2) or (n_cells, 3)
    clusters = adata.obs["Cluster"].astype("category")  # cluster information
    unique_clusters = clusters.cat.categories  # list of unique clusters

    # ---------------- (key change) ----------------
    # 2. Create a result DataFrame for storing nearest-cluster distances for each cell
    #    Note: using adata.obs["cell_id"] as row index
    df_nearest_cluster_dist = pd.DataFrame(
        index=adata.obs["cell_id"],
        columns=unique_clusters,
        dtype=float
    )

    # 3. For each cluster, build a nearest-neighbor model and query distances from all cells
    for c in unique_clusters:
        # 3.1 Extract coordinates for this cluster
        mask_c = (clusters == c)
        coords_c = coords[mask_c]

        # If this cluster has no cells, set the entire column to NaN
        if coords_c.shape[0] == 0:
            df_nearest_cluster_dist.loc[:, c] = np.nan
            continue

        # 3.2 Build nearest-neighbor model
        nbrs_c = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nbrs_c.fit(coords_c)

        # 3.3 Query nearest distances from all cells to this cluster
        dist_c, _ = nbrs_c.kneighbors(coords)
        df_nearest_cluster_dist[c] = dist_c[:, 0]

    # 4. Save results to adata.uns (or another suitable location)
    adata.uns["nearest_cluster_dist"] = df_nearest_cluster_dist

    # ------------------- Below: hierarchical clustering visualization of distance matrix -------------------
    # 5. Build a Series with cell_id as index and cluster as values for groupby alignment
    clusters_by_id = pd.Series(
        data=clusters.values,  # cluster values
        index=adata.obs["cell_id"],  # aligned with df_nearest_cluster_dist.index
        name="Cluster"
    )

    # Group df_nearest_cluster_dist by cluster using cell_id as index and compute mean
    df_group_mean = df_nearest_cluster_dist.groupby(clusters_by_id).mean()

    # 6. Drop columns that are entirely NaN (optional)
    df_group_mean_clean = df_group_mean.dropna(axis=1, how="all")

    if df_group_mean_clean.empty:
        print(f"Warning: df_group_mean_clean is empty for sample {sample}.")
        print("Check if there are clusters that exist in the data.")
        # If needed, process other samples here
        return

    # Visualize this matrix with clustermap
    plt.figure(figsize=figsize)
    g = sns.clustermap(
        df_group_mean_clean,
        cmap="RdBu",
        figsize=figsize,
        row_cluster=True,
        col_cluster=True,
        linewidths=0.5,
        annot=False
    )

    # 3) Set heatmap cells to be square
    g.ax_heatmap.set_aspect("equal")

    # If drawing dendrogram, adjust dendrogram and color legend positions
    if show_dendrogram:
        # 4) Fix row dendrogram alignment with heatmap in y direction
        row_dendro_pos = g.ax_row_dendrogram.get_position()
        heatmap_pos = g.ax_heatmap.get_position()
        g.ax_row_dendrogram.set_position([
            row_dendro_pos.x0,
            heatmap_pos.y0,
            row_dendro_pos.width,
            heatmap_pos.height
        ])

        # 5) Fix column dendrogram alignment with heatmap in x direction
        col_dendro_pos = g.ax_col_dendrogram.get_position()
        g.ax_col_dendrogram.set_position([
            heatmap_pos.x0,
            col_dendro_pos.y0,
            heatmap_pos.width,
            col_dendro_pos.height
        ])

        # 6) Adjust color legend (g.cax) position
        # Compute the blank area in the top-left:
        # Horizontal: from left edge of row dendrogram to left edge of heatmap;
        # Vertical: from top edge of column dendrogram to top edge of heatmap.
        empty_left = g.ax_row_dendrogram.get_position().x0
        empty_right = heatmap_pos.x0
        empty_width = empty_right - empty_left

        col_dendro_bbox = g.ax_col_dendrogram.get_position()
        empty_bottom = col_dendro_bbox.y0 + col_dendro_bbox.height
        empty_top = heatmap_pos.y0 + heatmap_pos.height
        empty_height = empty_top - empty_bottom

        # To avoid oversized legend, use 80% of the blank area and center it
        cbar_width = empty_width * 0.3
        cbar_height = empty_height * 0.7
        cbar_x = empty_left + (empty_width - cbar_width) / 2
        cbar_y = empty_bottom + (empty_height - cbar_height) / 2

        g.cax.set_position([cbar_x, cbar_y, cbar_width, cbar_height])

    # Set axis labels and title
    g.ax_heatmap.set_xlabel("Findee", fontsize=12)
    g.ax_heatmap.set_ylabel("Searcher", fontsize=12)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)

    # Set the overall figure title (not the heatmap title)
    g.fig.suptitle(f"SFplot of {sample}", fontsize=12, y=1)

    # 7. Save as PDF with sample name
    output_file = os.path.join(output_dir, f"SFplot_of_{sample}.pdf")
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"Sample {sample} done. PDF saved to {output_file}")


def generate_cluster_distance_heatmap_from_adata(
    adata: 'anndata.AnnData',
    cluster_col: str = "Cluster",
    output_dir: Optional[str] = None,
    output_filename: Optional[str] = None,
    figsize: tuple = (8, 8),
    cmap: str = "RdBu",
    max_scale: float = 10,
    show_dendrogram: bool = True  # new parameter: whether to draw dendrogram (default: True)
):
    """
    Generate and save a distance heatmap from each cell cluster to its nearest cluster center.

    Parameters:
    ----------
    adata : anndata.AnnData
        AnnData object containing preprocessed data.
    cluster_col : str, optional
        Column name in `adata.obs` containing cluster information. Defaults to "Cluster".
    output_dir : Optional[str]
        Output directory for the PDF file. Defaults to current working directory.
    output_filename : Optional[str]
        Output file name. If not specified, uses "clustermap_output_{sample}.pdf".
    figsize : tuple, optional
        Size of the heatmap. Defaults to (7, 7).
    cmap : str, optional
        Colormap for the heatmap. Defaults to "RdBu".
    max_scale : float, optional
        `max_value` parameter for `sc.pp.scale`, used to clip Z-scores. Defaults to 10.

    Returns:
    -------
    None
    """
    # Set output directory
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # Check whether cluster_col exists in adata.obs
    if cluster_col not in adata.obs.columns:
        raise ValueError(f"'{cluster_col}' does not exist in adata.obs. Please check the column name.")

    # Extract cluster information
    clusters = adata.obs[cluster_col].astype("category")  # cluster information
    unique_clusters = clusters.cat.categories  # list of unique clusters

    # 1. Extract coordinate information
    if "spatial" not in adata.obsm:
        raise ValueError("'spatial' coordinate info does not exist in adata.obsm. Please ensure spatial coordinates are present.")
    coords = adata.obsm["spatial"]  # (n_cells, 2) or (n_cells, 3)

    # 2. Create a result DataFrame for storing nearest-cluster distances for each cell
    #    Note: using adata.obs["cell_id"] as row index
    if "cell_id" not in adata.obs.columns:
        raise ValueError("'cell_id' column does not exist in adata.obs. Please ensure the data contains 'cell_id'.")
    df_nearest_cluster_dist = pd.DataFrame(
        index=adata.obs["cell_id"],
        columns=unique_clusters,
        dtype=float
    )

    # 3. For each cluster, build a nearest-neighbor model and query distances from all cells
    for c in unique_clusters:
        # 3.1 Extract coordinates for this cluster
        mask_c = (clusters == c)
        coords_c = coords[mask_c]

        # If this cluster has no cells, set the entire column to NaN
        if coords_c.shape[0] == 0:
            df_nearest_cluster_dist.loc[:, c] = np.nan
            continue

        # 3.2 Build nearest-neighbor model
        nbrs_c = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nbrs_c.fit(coords_c)

        # 3.3 Query nearest distances from all cells to this cluster
        dist_c, _ = nbrs_c.kneighbors(coords)
        df_nearest_cluster_dist[c] = dist_c[:, 0]

    # 4. Save results to adata.uns (or another suitable location)
    adata.uns["nearest_cluster_dist"] = df_nearest_cluster_dist

    # ------------------- Below: hierarchical clustering visualization of distance matrix -------------------
    # 5. Build a Series with cell_id as index and cluster as values for groupby alignment
    clusters_by_id = pd.Series(
        data=clusters.values,  # cluster values
        index=adata.obs["cell_id"],  # aligned with df_nearest_cluster_dist.index
        name=cluster_col
    )

    # Group df_nearest_cluster_dist by cluster using cell_id as index and compute mean
    df_group_mean = df_nearest_cluster_dist.groupby(clusters_by_id).mean()

    # 6. Drop columns that are entirely NaN
    df_group_mean_clean = df_group_mean.dropna(axis=1, how="all")

    if df_group_mean_clean.empty:
        print(f"Warning: df_group_mean_clean is empty. Please check that clusters exist in the data.")
        return

    # Visualize this matrix with clustermap
    plt.figure(figsize=figsize)
    g = sns.clustermap(
        df_group_mean_clean,
        cmap=cmap,
        figsize=figsize,
        row_cluster=True,
        col_cluster=True,
        linewidths=0.5,
        annot=False
    )

    # 3) Set heatmap cells to be square
    g.ax_heatmap.set_aspect("equal")

    # If drawing dendrogram, adjust dendrogram and color legend positions
    if show_dendrogram:
        # 4) Fix row dendrogram alignment with heatmap in y direction
        row_dendro_pos = g.ax_row_dendrogram.get_position()
        heatmap_pos = g.ax_heatmap.get_position()
        g.ax_row_dendrogram.set_position([
            row_dendro_pos.x0,
            heatmap_pos.y0,
            row_dendro_pos.width,
            heatmap_pos.height
        ])

        # 5) Fix column dendrogram alignment with heatmap in x direction
        col_dendro_pos = g.ax_col_dendrogram.get_position()
        g.ax_col_dendrogram.set_position([
            heatmap_pos.x0,
            col_dendro_pos.y0,
            heatmap_pos.width,
            col_dendro_pos.height
        ])

        # 6) Adjust color legend (g.cax) position
        # Compute the blank area in the top-left:
        # Horizontal: from left edge of row dendrogram to left edge of heatmap;
        # Vertical: from top edge of column dendrogram to top edge of heatmap.
        empty_left = g.ax_row_dendrogram.get_position().x0
        empty_right = heatmap_pos.x0
        empty_width = empty_right - empty_left

        col_dendro_bbox = g.ax_col_dendrogram.get_position()
        empty_bottom = col_dendro_bbox.y0 + col_dendro_bbox.height
        empty_top = heatmap_pos.y0 + heatmap_pos.height
        empty_height = empty_top - empty_bottom

        # To avoid oversized legend, use 80% of the blank area and center it
        cbar_width = empty_width * 0.3
        cbar_height = empty_height * 0.7
        cbar_x = empty_left + (empty_width - cbar_width) / 2
        cbar_y = empty_bottom + (empty_height - cbar_height) / 2

        g.cax.set_position([cbar_x, cbar_y, cbar_width, cbar_height])

    # Set axis labels and title
    g.ax_heatmap.set_xlabel("Findee", fontsize=12)
    g.ax_heatmap.set_ylabel("Searcher", fontsize=12)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)

    # Set the overall figure title (not the heatmap title)
    sample = adata.uns.get("sample", "Sample")  # assumes sample name is stored in adata.uns
    g.fig.suptitle(f"SFplot of {sample}", fontsize=12, y=1)

    # 7. Save as PDF with sample name
    if output_filename is None:
        output_filename = f"SFplot_of_{sample}.pdf"
    output_file = os.path.join(output_dir, output_filename)
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"Cluster distance heatmap saved to {output_file}")


def generate_cluster_distance_heatmap_from_df(
    df: pd.DataFrame,
    x_col: str = 'x',
    y_col: str = 'y',
    celltype_col: str = 'celltype',
    sample: str = 'Sample',
    output_dir: Optional[str] = None,
    output_filename: Optional[str] = None,
    figsize: tuple = (8, 8),
    cmap: str = "RdBu",
    show_dendrogram: bool = True  # whether to draw dendrogram (default: True)
):
    """
    Generate and save a distance heatmap from each cell cluster to its nearest cluster center.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing cell data.
    x_col : str, optional
        Column name for x coordinates. Defaults to 'x'.
    y_col : str, optional
        Column name for y coordinates. Defaults to 'y'.
    celltype_col : str, optional
        Column name for cell type. Defaults to 'celltype'.
    output_dir : Optional[str]
        Output directory for the PDF file. Defaults to current working directory.
    output_filename : Optional[str]
        Output file name. If not specified, uses "clustermap_output.pdf".
    figsize : tuple, optional
        Size of the heatmap. Defaults to (8, 8).
    cmap : str, optional
        Colormap for the heatmap. Defaults to "RdBu".

    Returns:
    -------
    None
    """
    # Set output directory
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # Check whether required columns exist in the DataFrame
    required_columns = {x_col, y_col, celltype_col}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

    # Extract cluster information
    clusters = df[celltype_col].astype('category')
    unique_clusters = clusters.cat.categories

    # Extract coordinate information
    coords = df[[x_col, y_col]].values  # (n_cells, 2)

    # Create a result DataFrame for storing nearest-cluster distances for each cell
    df_nearest_cluster_dist = pd.DataFrame(
        index=df.index,
        columns=unique_clusters,
        dtype=float
    )

    # For each cluster, build a nearest-neighbor model and query distances
    for c in unique_clusters:
        # Extract coordinates for this cluster
        mask_c = (clusters == c)
        coords_c = coords[mask_c]

        # If this cluster has no cells, set the entire column to NaN
        if coords_c.shape[0] == 0:
            df_nearest_cluster_dist.loc[:, c] = np.nan
            continue

        # Build nearest-neighbor model
        nbrs_c = NearestNeighbors(n_neighbors=1, algorithm='auto')
        nbrs_c.fit(coords_c)

        # Query nearest distances to this cluster
        dist_c, _ = nbrs_c.kneighbors(coords)
        df_nearest_cluster_dist[c] = dist_c[:, 0]

    # ------------------- Below: hierarchical clustering visualization of distance matrix -------------------
    # Group df_nearest_cluster_dist by cluster using celltype as index and compute mean
    df_group_mean = df_nearest_cluster_dist.groupby(clusters).mean()

    # Drop columns that are entirely NaN
    df_group_mean_clean = df_group_mean.dropna(axis=1, how='all')

    if df_group_mean_clean.empty:
        print("Warning: df_group_mean_clean is empty. Please check that clusters exist in the data.")
        return

    # Visualize this matrix with clustermap
    g = sns.clustermap(
        df_group_mean_clean,
        cmap=cmap,
        figsize=figsize,
        row_cluster=True,
        col_cluster=True,
        linewidths=0.5,
        annot=False
    )

    # Set heatmap cells to be square
    g.ax_heatmap.set_aspect('equal')

    # If drawing dendrogram, adjust dendrogram and color legend positions
    if show_dendrogram:
        # Fix row dendrogram alignment with heatmap in y direction
        row_dendro_pos = g.ax_row_dendrogram.get_position()
        heatmap_pos = g.ax_heatmap.get_position()
        g.ax_row_dendrogram.set_position([
            row_dendro_pos.x0,
            heatmap_pos.y0,
            row_dendro_pos.width,
            heatmap_pos.height
        ])

        # Fix column dendrogram alignment with heatmap in x direction
        col_dendro_pos = g.ax_col_dendrogram.get_position()
        g.ax_col_dendrogram.set_position([
            heatmap_pos.x0,
            col_dendro_pos.y0,
            heatmap_pos.width,
            col_dendro_pos.height
        ])

        # Adjust color legend (g.cax) position
        empty_left = g.ax_row_dendrogram.get_position().x0
        empty_right = heatmap_pos.x0
        empty_width = empty_right - empty_left

        col_dendro_bbox = g.ax_col_dendrogram.get_position()
        empty_bottom = col_dendro_bbox.y0 + col_dendro_bbox.height
        empty_top = heatmap_pos.y0 + heatmap_pos.height
        empty_height = empty_top - empty_bottom

        cbar_width = empty_width * 0.3
        cbar_height = empty_height * 0.7
        cbar_x = empty_left + (empty_width - cbar_width) / 2
        cbar_y = empty_bottom + (empty_height - cbar_height) / 2

        g.cax.set_position([cbar_x, cbar_y, cbar_width, cbar_height])

    # Set axis labels and title
    g.ax_heatmap.set_xlabel("Findee", fontsize=12)
    g.ax_heatmap.set_ylabel("Searcher", fontsize=12)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
    g.fig.suptitle(f"SFplot of {sample}", fontsize=12, y=1)

    # Save as PDF
    if output_filename is None:
        output_filename = "clustermap_output.pdf"
    output_file = os.path.join(output_dir, output_filename)
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"Cluster distance heatmap saved to {output_file}")
