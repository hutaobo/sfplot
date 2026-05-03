# sfplot/compute_cophenetic_distances_from_adata.py

import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform, pdist
from sklearn.neighbors import NearestNeighbors


def compute_cophenetic_distances_from_adata(
    adata: 'anndata.AnnData',
    cluster_col: str = "Cluster",
    output_dir: Optional[str] = None,
    method: str = "average"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute and return cophenetic distance matrices in both row and column dimensions (using cophenet),
    then apply linear normalization to [0,1] for each separately.

    Unlike the previous version, min and max values are computed independently for rows and columns.
    """

    # 0. Optional: handle output directory
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # 1. Extract cluster information
    if cluster_col not in adata.obs.columns:
        raise ValueError(f"'{cluster_col}' does not exist in adata.obs. Please check the column name.")

    clusters = adata.obs[cluster_col].astype("category")
    unique_clusters = clusters.cat.categories

    # 2. Extract spatial coordinates
    if "spatial" not in adata.obsm:
        raise ValueError("'spatial' coordinate info does not exist in adata.obsm. Please ensure the data contains spatial coordinates.")
    coords = adata.obsm["spatial"]  # typically (n_cells, 2) or (n_cells, 3)

    # 3. Build a DataFrame with rows as cell_id and columns as cluster
    if "cell_id" not in adata.obs.columns:
        raise ValueError("'cell_id' does not exist in adata.obs. Please ensure the data contains this column.")

    df_nearest_cluster_dist = pd.DataFrame(
        index=adata.obs["cell_id"],
        columns=unique_clusters,
        dtype=float
    )

    # 4. For each cluster, use a nearest-neighbor model to compute distances from all cells to that cluster
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

    # (optional) save results to adata.uns
    adata.uns["nearest_cluster_dist"] = df_nearest_cluster_dist

    # 5. Compute mean distance per cluster => cluster x cluster matrix
    clusters_by_id = pd.Series(
        data=clusters.values,
        index=adata.obs["cell_id"],
        name=cluster_col
    )
    df_group_mean = df_nearest_cluster_dist.groupby(clusters_by_id).mean()

    # 6. Drop clusters whose entire column is NaN
    df_group_mean_clean = df_group_mean.dropna(axis=1, how="all")
    if df_group_mean_clean.empty:
        print("Warning: df_group_mean_clean is empty. Please check the data.")
        return pd.DataFrame(), pd.DataFrame()

    # 7. Perform hierarchical clustering separately on rows and columns
    row_linkage = linkage(df_group_mean_clean, method=method)
    col_linkage = linkage(df_group_mean_clean.T, method=method)

    # 8. cophenet: correctly obtain cophenetic distances (condensed form)
    row_coph_corr, row_coph_condensed = cophenet(row_linkage, pdist(df_group_mean_clean.values))
    col_coph_corr, col_coph_condensed = cophenet(col_linkage, pdist(df_group_mean_clean.T.values))

    # 9. Convert condensed distances to square form (squareform)
    row_cophenetic_square = squareform(row_coph_condensed)
    col_cophenetic_square = squareform(col_coph_condensed)

    # 10. Build DataFrame
    row_labels = df_group_mean_clean.index
    col_labels = df_group_mean_clean.columns

    row_cophenetic_df = pd.DataFrame(
        row_cophenetic_square,
        index=row_labels,
        columns=row_labels
    )
    col_cophenetic_df = pd.DataFrame(
        col_cophenetic_square,
        index=col_labels,
        columns=col_labels
    )

    # -------- Compute min/max for rows and columns separately and normalize to [0,1] --------
    row_min = row_cophenetic_df.values.min()
    row_max = row_cophenetic_df.values.max()

    col_min = col_cophenetic_df.values.min()
    col_max = col_cophenetic_df.values.max()

    def normalize_df(df: pd.DataFrame, dmin: float, dmax: float) -> pd.DataFrame:
        if dmin == dmax:
            # If there is no range, return original DF unchanged (or all 0)
            return df
        return (df - dmin) / (dmax - dmin)

    row_cophenetic_df_norm = normalize_df(row_cophenetic_df, row_min, row_max)
    col_cophenetic_df_norm = normalize_df(col_cophenetic_df, col_min, col_max)

    # Print to inspect range
    print("Cophenetic distance & normalization done.")
    print(
        f"Row dist range -> original: [{row_min:.4f}, {row_max:.4f}], normalized: [{row_cophenetic_df_norm.values.min():.4f}, {row_cophenetic_df_norm.values.max():.4f}]")
    print(
        f"Col dist range -> original: [{col_min:.4f}, {col_max:.4f}], normalized: [{col_cophenetic_df_norm.values.min():.4f}, {col_cophenetic_df_norm.values.max():.4f}]")

    return row_cophenetic_df_norm, col_cophenetic_df_norm


# sfplot/compute_cophenetic_distances_from_df.py

import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors


def compute_searcher_findee_distance_matrix_from_df(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    z_col: Optional[str] = None,
    celltype_col: str = "celltype"
) -> pd.DataFrame:
    """
    Compute and return a directed inter-cluster average nearest-neighbor distance matrix.
    Row and column indices are the clusters (cell types) present in df; rows represent "Searcher" clusters, columns represent "Findee" clusters.
    Each element is the average nearest-neighbor distance from all cells in the row cluster to all cells in the column cluster.
    Clusters with no cells in the data will not appear in the result matrix.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing cell coordinates and type data.
    x_col, y_col : str, optional
        Column names for cell x/y coordinates. Defaults to "x" and "y".
    z_col : Optional[str], optional
        Column name for the z coordinate; if provided it is used, otherwise None means 2D only.
    celltype_col : str, optional
        Column name for cell type / cluster labels. Defaults to "celltype".

    Returns:
    -------
    pd.DataFrame
        Distance matrix DataFrame with cluster names as index and columns. Shape is (n_clusters, n_clusters);
        values are the average nearest-neighbor distance between the corresponding cluster pairs. NaN if unavailable.
    """
    # 1. Check required columns exist
    required_cols = {x_col, y_col, celltype_col}
    if z_col is not None:
        required_cols.add(z_col)
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_cols}")
    # 2. Extract cluster info and remove unused categories
    clusters = df[celltype_col].astype("category")
    clusters = clusters.cat.remove_unused_categories()
    unique_clusters = clusters.cat.categories  # all actually existing cluster categories
    # 3. Extract cell coordinates (numpy array)
    coord_cols = [x_col, y_col] + ([z_col] if z_col is not None else [])
    coords = df[coord_cols].values  # shape: (n_cells, dims)
    # 4. Initialize cell × cluster nearest-neighbor distance matrix
    df_nearest_cluster_dist = pd.DataFrame(index=df.index, columns=unique_clusters, dtype=float)
    # 5. Compute distances from each cell to the nearest cell in each target cluster
    for c in unique_clusters:
        mask_c = (clusters == c)
        coords_c = coords[mask_c]
        if coords_c.shape[0] == 0:
            # If this cluster has no cells, keep the entire column as NaN
            df_nearest_cluster_dist.loc[:, c] = np.nan
            continue
        # Build nearest-neighbor model for current cluster and compute distances from all cells
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(coords_c)
        dist_c, _ = nbrs.kneighbors(coords)
        df_nearest_cluster_dist[c] = dist_c[:, 0]
    # 6. Group by source cluster and compute mean to get cluster × cluster average distance matrix
    distance_matrix = df_nearest_cluster_dist.groupby(clusters).mean()
    # 7. Drop columns that are entirely NaN (clusters with no cells)
    distance_matrix = distance_matrix.dropna(axis=1, how="all")
    return distance_matrix


def compute_cophenetic_from_distance_matrix(
    distance_matrix: pd.DataFrame,
    method: str = "average",
    show_corr: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform hierarchical clustering in both row and column directions on the given inter-cluster distance matrix,
    and compute cophenetic distance matrices. Results are independently normalized to [0,1] for rows and columns.

    Parameters:
    ----------
    distance_matrix : pd.DataFrame
        Input distance matrix with source clusters as rows and target clusters as columns
        (e.g. output of compute_searcher_findee_distance_matrix_from_df).
    method : str, optional
        Linkage method for hierarchical clustering. Defaults to "average".
    show_corr : bool, optional
        Whether to print the cophenetic correlation coefficient (printed separately for rows and columns). Defaults to False.

    Returns:
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (row_coph, col_coph). Cophenetic distance matrices (DataFrames) for row and column clusters,
        each independently normalized to [0,1].
    """
    # 1. Perform hierarchical clustering on rows
    row_linkage = linkage(distance_matrix, method=method)
    # 2. Perform hierarchical clustering on columns
    col_linkage = linkage(distance_matrix.T, method=method)
    # 3. Compute cophenetic distances and correlation coefficients
    row_coph_corr, row_coph_condensed = cophenet(row_linkage, pdist(distance_matrix.values))
    col_coph_corr, col_coph_condensed = cophenet(col_linkage, pdist(distance_matrix.T.values))
    # 4. Convert condensed distances to square form
    row_cophenetic_square = squareform(row_coph_condensed)
    col_cophenetic_square = squareform(col_coph_condensed)
    # 5. Build DataFrames (preserve cluster labels)
    row_labels = distance_matrix.index
    col_labels = distance_matrix.columns
    row_cophenetic_df = pd.DataFrame(row_cophenetic_square, index=row_labels, columns=row_labels)
    col_cophenetic_df = pd.DataFrame(col_cophenetic_square, index=col_labels, columns=col_labels)
    # 6. Normalize row and column distance matrices separately to [0,1]
    def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        dmin, dmax = df.values.min(), df.values.max()
        return df if dmin == dmax else (df - dmin) / (dmax - dmin)
    row_cophenetic_df_norm = normalize_df(row_cophenetic_df)
    col_cophenetic_df_norm = normalize_df(col_cophenetic_df)
    # 7. Optionally print cophenetic correlation coefficients
    if show_corr:
        print(f"Row cophenetic correlation coefficient: {row_coph_corr:.4f}")
        print(f"Column cophenetic correlation coefficient: {col_coph_corr:.4f}")
    # 8. Return result matrices
    return row_cophenetic_df_norm, col_cophenetic_df_norm


def compute_cophenetic_distances_from_df(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    z_col: Optional[str] = None,
    celltype_col: str = "celltype",
    output_dir: Optional[str] = None,
    method: str = "average",
    show_corr: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute and return cophenetic distance matrices in both row and column dimensions,
    then apply linear normalization to [0, 1] for each separately.

    If z_col is provided, uses (x, y, z) for distance computation; otherwise uses only (x, y).

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing cell data.
    x_col, y_col, z_col : str, optional
        Column names for spatial coordinates. z_col defaults to None.
    celltype_col : str, optional
        Column name for cell type.
    output_dir : Optional[str]
        Output file directory; if None, uses the current working directory.
    method : str, optional
        Linkage method for hierarchical clustering. Defaults to "average".
    show_corr : bool, optional
        Whether to print the cophenetic correlation coefficient for rows and columns. Defaults to False.

    Returns:
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Row and column cophenetic distance matrices, both normalized to [0, 1].
    """
    # 0. Ensure output directory exists
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    # 1. Compute inter-cluster average nearest-neighbor distance matrix
    distance_matrix = compute_searcher_findee_distance_matrix_from_df(df, x_col, y_col, z_col, celltype_col)
    # 2. Check if the matrix is empty
    if distance_matrix.empty:
        raise ValueError("df_group_mean_clean is empty, please check the data.")
    # 3. Compute cophenetic distance matrix and normalize
    row_coph, col_coph = compute_cophenetic_from_distance_matrix(distance_matrix, method=method, show_corr=show_corr)
    # 4. Return results
    return row_coph, col_coph


# ---------------- plot_cophenetic_heatmap.py ----------------
import os
import contextlib
import logging
from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.font_manager as fm
mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ---------- 1. Make text in PDF editable ----------
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# ---------- 2. Utility: silence any logger ----------
@contextlib.contextmanager
def silence(logger_name: str, level: int = logging.ERROR):
    """Temporarily raise the logging level of *logger_name*."""
    logger = logging.getLogger(logger_name)
    old_level = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(old_level)


# ---------- 3. Ensure a valid sans-serif font is available ----------
def _ensure_font():
    """Use Arial if present; otherwise switch to Liberation Sans / DejaVu Sans."""
    want = "Arial"
    if any(want in f.name for f in fm.fontManager.ttflist):
        mpl.rcParams["font.family"] = want
        return
    fallback = "Liberation Sans"
    if not any(fallback in f.name for f in fm.fontManager.ttflist):
        fallback = "DejaVu Sans"
    # Override font.family and font.sans-serif list, removing Arial
    mpl.rcParams["font.family"] = [fallback]
    mpl.rcParams["font.sans-serif"] = [fallback]


# ---------- 4. Core function ----------
def plot_cophenetic_heatmap(
    matrix: pd.DataFrame,
    matrix_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    output_filename: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = "RdBu",
    linewidths: float = 0.5,
    annot: bool = False,
    sample: str = "Sample",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    show_dendrogram: bool = True,
    quiet: bool = True,
    return_figure: bool = False,
    return_image: bool = False,
    dpi: int = 300,  # image DPI, affects image quality
):
    """
    Draw a cophenetic heatmap (seaborn.clustermap), guaranteeing:
      • Text in PDF is editable
      • Legend position is auto-adjusted
      • figsize is dynamically adjusted
      • fontTools.subset & findfont logs are silenced

    Parameters:
      ...existing parameters...
      return_figure: whether to return the figure object instead of saving to file
      return_image: whether to return a high-resolution PIL image instead of the figure object
      dpi: image DPI resolution, only effective when return_image=True

    Returns:
      If return_figure=True, returns a seaborn.ClusterGrid object
      If return_image=True, returns a PIL.Image object
      Otherwise returns None
    """
    # When both return modes are specified, return image takes priority
    if return_image:
        return_figure = False

    # ---- Ensure a usable font is available, suppress findfont warnings ----
    _ensure_font()

    # ---- Dynamic figsize ----
    if figsize is None:
        rows, cols = matrix.shape
        figsize = (max(8.0, 0.25 * cols + 0.5), max(8.0, 0.25 * rows + 0.5))

    # ---- Output path & title ----
    if not (return_figure or return_image):  # only handle path when saving a file
        if output_dir is None:
            output_dir = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)

    title_map = {
        "row_coph": (
            f"StructureMap of {sample}",
            f"StructureMap_of_{sample}.pdf",
            "Searcher",
            "Searcher",
        ),
        "col_coph": (
            f"Findee's D score of {sample}",
            f"Findee_D_score_of_{sample}.pdf",
            "Findee",
            "Findee",
        ),
    }
    title, default_pdf, xlab, ylab = title_map.get(
        matrix_name,
        (
            f"D score of {sample}",
            f"D_score_of_{sample}.pdf",
            xlabel or "Findee",
            ylabel or "Searcher",
        ),
    )
    xlabel, ylabel = xlabel or xlab, ylabel or ylab

    # Only set path when saving a file
    if not (return_figure or return_image):
        pdf_path = os.path.join(output_dir, output_filename or default_pdf)

    # ---- Internal drawing function ----
    def _draw():
        g = sns.clustermap(
            data=matrix,
            figsize=figsize,
            cmap=cmap,
            row_cluster=show_dendrogram,
            col_cluster=show_dendrogram,
            linewidths=linewidths,
            annot=annot,
        )

        # 1) Ensure heatmap cells are square
        g.ax_heatmap.set_aspect("equal")

        # 2) Adjust dendrogram & colorbar positions
        if show_dendrogram:
            heat = g.ax_heatmap.get_position()
            row_d = g.ax_row_dendrogram.get_position()
            col_d = g.ax_col_dendrogram.get_position()

            # 2-1 Align row dendrogram vertically
            g.ax_row_dendrogram.set_position(
                [row_d.x0, heat.y0, row_d.width, heat.height]
            )
            # 2-2 Align column dendrogram horizontally
            g.ax_col_dendrogram.set_position(
                [heat.x0, col_d.y0, heat.width, col_d.height]
            )
            # 2-3 Place colorbar in the top-left empty area
            empty_w = heat.x0 - row_d.x0
            empty_h = (heat.y0 + heat.height) - (col_d.y0 + col_d.height)
            g.cax.set_position(
                [
                    row_d.x0 + empty_w * 0.35,
                    col_d.y0 + col_d.height + empty_h * 0.15,
                    empty_w * 0.30,
                    empty_h * 0.70,
                ]
            )

        # 3) Axis labels & title
        g.ax_heatmap.set_xlabel(xlabel, fontsize=12)
        g.ax_heatmap.set_ylabel(ylabel, fontsize=12)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
        g.fig.suptitle(title, fontsize=12, y=1.02)

        # Handle figure according to return type
        if return_image:
            # Convert figure to high-resolution image
            from io import BytesIO
            from PIL import Image

            # Create an in-memory buffer for the image
            buf = BytesIO()
            g.fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
            buf.seek(0)

            # Load as PIL image
            image = Image.open(buf)
            image_copy = image.copy()  # create a copy so the original can be closed
            buf.close()
            plt.close(g.fig)  # close figure to avoid memory leaks

            return image_copy
        elif not return_figure:
            plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
            plt.close(g.fig)
            return None

        # Return ClusterGrid object
        return g

    # ---- Execute drawing (with optional log silencing) ----
    if quiet:
        with silence("fontTools.subset", logging.ERROR), silence(
            "matplotlib.font_manager", logging.ERROR
        ):
            result = _draw()
    else:
        result = _draw()

    # If saving a file, print message
    if not (return_figure or return_image):
        print(f"Heat‑map saved to: {pdf_path}")

    return result
