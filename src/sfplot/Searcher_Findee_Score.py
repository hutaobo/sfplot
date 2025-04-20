# sfplot/compute_cophenetic_distances_from_adata.py

import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform, pdist


def compute_cophenetic_distances_from_adata(
    adata: 'anndata.AnnData',
    cluster_col: str = "Cluster",
    output_dir: Optional[str] = None,
    method: str = "average"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    计算并返回行、列两个维度上的 cophenetic distance 矩阵 (使用 cophenet)，
    并在最后将距离分别做线性归一化到 [0,1]。

    与之前不同的是，这里将行、列分开计算最小值和最大值，各自独立进行归一化。
    """

    # 0. 可选: 处理输出目录
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # 1. 提取 cluster 信息
    if cluster_col not in adata.obs.columns:
        raise ValueError(f"'{cluster_col}' 不存在于 adata.obs 中。请检查列名。")

    clusters = adata.obs[cluster_col].astype("category")
    unique_clusters = clusters.cat.categories

    # 2. 提取空间坐标
    if "spatial" not in adata.obsm:
        raise ValueError("'spatial' 坐标信息不存在于 adata.obsm 中。请确保数据包含空间坐标。")
    coords = adata.obsm["spatial"]  # 通常 (n_cells, 2) 或 (n_cells, 3)

    # 3. 构建一个 DataFrame, 行是 cell_id, 列是 cluster
    if "cell_id" not in adata.obs.columns:
        raise ValueError("'cell_id' 不存在于 adata.obs 中。请确保数据包含该列。")

    df_nearest_cluster_dist = pd.DataFrame(
        index=adata.obs["cell_id"],
        columns=unique_clusters,
        dtype=float
    )

    # 4. 对每个 cluster, 用最近邻模型计算所有细胞到该 cluster 最近中心的距离
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

    # (可选) 保存结果到 adata.uns
    adata.uns["nearest_cluster_dist"] = df_nearest_cluster_dist

    # 5. 对每个 cluster 求距离均值 => cluster x cluster 矩阵
    clusters_by_id = pd.Series(
        data=clusters.values,
        index=adata.obs["cell_id"],
        name=cluster_col
    )
    df_group_mean = df_nearest_cluster_dist.groupby(clusters_by_id).mean()

    # 6. 删除整列全 NaN 的 cluster
    df_group_mean_clean = df_group_mean.dropna(axis=1, how="all")
    if df_group_mean_clean.empty:
        print("Warning: df_group_mean_clean is empty. 请检查数据。")
        return pd.DataFrame(), pd.DataFrame()

    # 7. 分别对行、列做层次聚类
    row_linkage = linkage(df_group_mean_clean, method=method)
    col_linkage = linkage(df_group_mean_clean.T, method=method)

    # 8. cophenet：正确获取 cophenetic 距离 (condensed form)
    row_coph_corr, row_coph_condensed = cophenet(row_linkage, pdist(df_group_mean_clean.values))
    col_coph_corr, col_coph_condensed = cophenet(col_linkage, pdist(df_group_mean_clean.T.values))

    # 9. 将 condensed 距离转为方阵 (squareform)
    row_cophenetic_square = squareform(row_coph_condensed)
    col_cophenetic_square = squareform(col_coph_condensed)

    # 10. 构建 DataFrame
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

    # -------- 分开计算行、列各自的 min / max 并归一化到 [0,1] --------
    row_min = row_cophenetic_df.values.min()
    row_max = row_cophenetic_df.values.max()

    col_min = col_cophenetic_df.values.min()
    col_max = col_cophenetic_df.values.max()

    def normalize_df(df: pd.DataFrame, dmin: float, dmax: float) -> pd.DataFrame:
        if dmin == dmax:
            # 如果没有跨度，直接返回原始 DF（或全 0）
            return df
        return (df - dmin) / (dmax - dmin)

    row_cophenetic_df_norm = normalize_df(row_cophenetic_df, row_min, row_max)
    col_cophenetic_df_norm = normalize_df(col_cophenetic_df, col_min, col_max)

    # 打印一下查看范围
    print("Cophenetic distance & normalization done.")
    print(
        f"Row dist range -> original: [{row_min:.4f}, {row_max:.4f}], normalized: [{row_cophenetic_df_norm.values.min():.4f}, {row_cophenetic_df_norm.values.max():.4f}]")
    print(
        f"Col dist range -> original: [{col_min:.4f}, {col_max:.4f}], normalized: [{col_cophenetic_df_norm.values.min():.4f}, {col_cophenetic_df_norm.values.max():.4f}]")

    return row_cophenetic_df_norm, col_cophenetic_df_norm


# sfplot/compute_cophenetic_distances_from_adata.py

import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors


def compute_cophenetic_distances_from_df(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    celltype_col: str = "celltype",
    output_dir: Optional[str] = None,
    method: str = "average",
    show_corr: bool = False,           # 新增参数，控制是否打印 cophenetic 相关系数
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    计算并返回行、列两个维度上的 cophenetic distance 矩阵，
    并在最后将距离分别线性归一化到 [0, 1]。

    参数
    ----
    df : pd.DataFrame
        包含细胞数据的 DataFrame。
    x_col, y_col : str, optional
        表示空间坐标的列名。
    celltype_col : str, optional
        表示细胞类型的列名。
    output_dir : Optional[str]
        输出文件目录；若为 None 则使用当前工作目录。
    method : str, optional
        层次聚类使用的链接方法，默认为 "average"。
    show_corr : bool, optional
        是否打印行、列的 cophenetic correlation coefficient。
        默认 False（不打印）。

    返回值
    ------
    Tuple[pd.DataFrame, pd.DataFrame]
        行和列的 cophenetic 距离矩阵，均已归一化到 [0, 1]。
    """
    # 0. 处理输出目录
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # 1. 检查必需列
    required_columns = {x_col, y_col, celltype_col}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame 必须包含以下列：{required_columns}")

    # 2. 提取 cluster 信息
    clusters = df[celltype_col].astype("category")
    unique_clusters = clusters.cat.categories

    # 3. 提取空间坐标
    coords = df[[x_col, y_col]].values  # (n_cells, 2)

    # 4. 初始化最近距离矩阵 (cell x cluster)
    df_nearest_cluster_dist = pd.DataFrame(
        index=df.index, columns=unique_clusters, dtype=float
    )

    # 5. 计算每个细胞到各 cluster 最近中心的距离
    for c in unique_clusters:
        mask_c = clusters == c
        coords_c = coords[mask_c]

        if coords_c.shape[0] == 0:
            df_nearest_cluster_dist.loc[:, c] = np.nan
            continue

        nbrs_c = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nbrs_c.fit(coords_c)
        dist_c, _ = nbrs_c.kneighbors(coords)
        df_nearest_cluster_dist[c] = dist_c[:, 0]

    # 6. 计算 cluster × cluster 平均距离
    df_group_mean = df_nearest_cluster_dist.groupby(clusters).mean()

    # 7. 删除全 NaN 的列
    df_group_mean_clean = df_group_mean.dropna(axis=1, how="all")
    if df_group_mean_clean.empty:
        raise ValueError("df_group_mean_clean 为空，请检查数据。")

    # 8. 层次聚类
    row_linkage = linkage(df_group_mean_clean, method=method)
    col_linkage = linkage(df_group_mean_clean.T, method=method)

    # 9. 计算 cophenetic 距离
    row_coph_corr, row_coph_condensed = cophenet(
        row_linkage, pdist(df_group_mean_clean.values)
    )
    col_coph_corr, col_coph_condensed = cophenet(
        col_linkage, pdist(df_group_mean_clean.T.values)
    )

    # 10. condensed → 方阵
    row_cophenetic_square = squareform(row_coph_condensed)
    col_cophenetic_square = squareform(col_coph_condensed)

    # 11. 构建 DataFrame
    row_labels = df_group_mean_clean.index
    col_labels = df_group_mean_clean.columns

    row_cophenetic_df = pd.DataFrame(
        row_cophenetic_square, index=row_labels, columns=row_labels
    )
    col_cophenetic_df = pd.DataFrame(
        col_cophenetic_square, index=col_labels, columns=col_labels
    )

    # 12. 归一化
    def normalize_df(mat: pd.DataFrame) -> pd.DataFrame:
        dmin, dmax = mat.values.min(), mat.values.max()
        return mat if dmin == dmax else (mat - dmin) / (dmax - dmin)

    row_cophenetic_df_norm = normalize_df(row_cophenetic_df)
    col_cophenetic_df_norm = normalize_df(col_cophenetic_df)

    # 13. 可选打印
    if show_corr:
        print(f"Row cophenetic correlation coefficient: {row_coph_corr:.4f}")
        print(f"Column cophenetic correlation coefficient: {col_coph_corr:.4f}")

    return row_cophenetic_df_norm, col_cophenetic_df_norm


# sfplot/plot_cophenetic_heatmap.py

"""
Self‑contained cophenetic‑heat‑map utility.

*  Editable text in Illustrator (Type‑42 embedding)
*  Tries local Arial TrueType first; otherwise falls back to Liberation Sans
*  Dynamic figsize so every label is legible
"""

import os
import pathlib
from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
#                       FONT / PDF SETTINGS
# ---------------------------------------------------------------------
mpl.rcParams['pdf.fonttype'] = 42        # embed TrueType, keep text editable
mpl.rcParams['ps.fonttype']  = 42

# -- 1) Try to register a local Arial TTF shipped with the project -----
_CANDIDATES = ['arial.ttf', 'Arial.ttf', 'ARIAL.TTF']
_SEARCH_DIRS = [
    pathlib.Path.cwd(),
    pathlib.Path.cwd() / "fonts"
]
_font_found = None
for d in _SEARCH_DIRS:
    for name in _CANDIDATES:
        p = d / name
        if p.is_file():
            fm.fontManager.addfont(str(p))
            _font_found = "Arial"
            break
    if _font_found:
        break

# -- 2) Decide which default family to use ----------------------------
if _font_found == "Arial":
    mpl.rcParams['font.family'] = 'Arial'
else:
    # Liberation Sans is metrically compatible with Arial, falls back cleanly
    if any("Liberation Sans" in f.name for f in fm.fontManager.ttflist):
        mpl.rcParams['font.family'] = 'Liberation Sans'
        _font_found = "Liberation Sans"
    else:
        mpl.rcParams['font.family'] = 'DejaVu Sans'   # guaranteed to exist
        _font_found = "DejaVu Sans"

print(f"[heatmap_fontsafe] Using font family: {_font_found}")

# ---------------------------------------------------------------------
#            HEURISTIC FOR FIGURE SIZE WHEN USER LEAVES IT BLANK
# ---------------------------------------------------------------------
_CELL_W, _CELL_H   = 0.30, 0.30   # inches per matrix cell
_MARGINS           = dict(left=2.0, right=0.5, top=0.5, bottom=2.0)

def _auto_figsize(mat: pd.DataFrame) -> Tuple[float, float]:
    rows, cols = mat.shape
    w = _MARGINS['left'] + cols * _CELL_W + _MARGINS['right']
    h = _MARGINS['bottom'] + rows * _CELL_H + _MARGINS['top']
    return max(w, 4.0), max(h, 4.0)          # never smaller than 4×4 in

# ---------------------------------------------------------------------
#                  MAIN PLOTTING FUNCTION
# ---------------------------------------------------------------------
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
    show_dendrogram: bool = True
):
    """Draw a Seaborn clustermap with Illustrator‑friendly fonts."""
    # -- paths ---------------------------------------------------------
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # -- labels / titles ----------------------------------------------
    if matrix_name == "row_coph":
        title       = f"StructureMap of {sample}"
        default_pdf = f"StructureMap_of_{sample}.pdf"
        xlabel, ylabel = "Searcher", "Searcher"
    elif matrix_name == "col_coph":
        title       = f"Findee's D score of {sample}"
        default_pdf = f"Findee_D_score_of_{sample}.pdf"
        xlabel, ylabel = "Findee", "Findee"
    else:
        title       = f"D score of {sample}"
        default_pdf = f"D_score_of_{sample}.pdf"
        xlabel = xlabel or "Findee"
        ylabel = ylabel or "Searcher"

    # -- dynamic figsize ----------------------------------------------
    if figsize is None:
        figsize = _auto_figsize(matrix)

    # -- draw heat‑map -------------------------------------------------
    g = sns.clustermap(
        data        = matrix,
        figsize     = figsize,
        cmap        = cmap,
        row_cluster = show_dendrogram,
        col_cluster = show_dendrogram,
        linewidths  = linewidths,
        annot       = annot
    )
    g.ax_heatmap.set_aspect("equal")

    # -- align dendrograms & colour bar --------------------------------
    if show_dendrogram:
        heat = g.ax_heatmap.get_position()
        row  = g.ax_row_dendrogram.get_position()
        col  = g.ax_col_dendrogram.get_position()

        g.ax_row_dendrogram.set_position(
            [row.x0, heat.y0, row.width, heat.height])
        g.ax_col_dendrogram.set_position(
            [heat.x0, col.y0, heat.width, col.height])

        # place colour‑bar in empty top‑left corner
        empty_w = heat.x0 - row.x0
        empty_h = (heat.y0 + heat.height) - (col.y0 + col.height)
        g.cax.set_position([
            row.x0 + empty_w * 0.35,
            col.y0 + col.height + empty_h * 0.15,
            empty_w * 0.30,
            empty_h * 0.70
        ])

    # -- labels --------------------------------------------------------
    g.ax_heatmap.set_xlabel(xlabel, fontsize=12)
    g.ax_heatmap.set_ylabel(ylabel, fontsize=12)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
    g.fig.suptitle(title, fontsize=12, y=1.02)

    # -- save ----------------------------------------------------------
    pdf_name = output_filename or default_pdf
    path = os.path.join(output_dir, pdf_name)
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Heat‑map saved to: {path}")
