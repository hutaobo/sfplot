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
    计算并返回一个有向的簇间平均最近邻距离矩阵。
    矩阵的行和列索引都是 df 中出现的簇（细胞类型），行表示“搜索者”簇，列表示“被搜索”簇。
    每个元素值表示：行簇中的所有细胞到列簇中所有细胞的最近邻距离的平均值。
    若某簇在数据中没有任何细胞，则不会出现在结果矩阵中。

    参数:
    ----
    df : pd.DataFrame
        包含细胞坐标和类别数据的 DataFrame。
    x_col, y_col : str, optional
        表示细胞横纵坐标的列名。默认为 "x" 和 "y"。
    z_col : Optional[str], optional
        表示细胞 z 坐标的列名，如有则使用，否则为 None 表示只使用二维坐标。
    celltype_col : str, optional
        表示细胞类别/簇名的列名。默认为 "celltype"。

    返回值:
    ------
    pd.DataFrame
        返回的距离矩阵 DataFrame，索引和列均为簇名。矩阵形状为 (n_clusters, n_clusters)，
        值为对应簇对之间的平均最近邻距离。若无可用距离则该元素为 NaN。
    """
    # 1. 检查必需列是否存在
    required_cols = {x_col, y_col, celltype_col}
    if z_col is not None:
        required_cols.add(z_col)
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame 必须包含以下列：{required_cols}")
    # 2. 提取簇信息并移除未出现的簇
    clusters = df[celltype_col].astype("category")
    clusters = clusters.cat.remove_unused_categories()
    unique_clusters = clusters.cat.categories  # 所有实际存在的簇类别
    # 3. 提取细胞坐标 (numpy 数组)
    coord_cols = [x_col, y_col] + ([z_col] if z_col is not None else [])
    coords = df[coord_cols].values  # shape: (n_cells, dims)
    # 4. 初始化细胞 × 簇的最近邻距离矩阵
    df_nearest_cluster_dist = pd.DataFrame(index=df.index, columns=unique_clusters, dtype=float)
    # 5. 计算每个簇作为目标时，各细胞到该簇最近邻细胞的距离
    for c in unique_clusters:
        mask_c = (clusters == c)
        coords_c = coords[mask_c]
        if coords_c.shape[0] == 0:
            # 若该簇没有细胞，则整列保持 NaN
            df_nearest_cluster_dist.loc[:, c] = np.nan
            continue
        # 建立当前簇的最近邻模型，并计算所有细胞到该簇的距离
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(coords_c)
        dist_c, _ = nbrs.kneighbors(coords)
        df_nearest_cluster_dist[c] = dist_c[:, 0]
    # 6. 按源簇分组求均值，得到簇 × 簇的平均距离矩阵
    distance_matrix = df_nearest_cluster_dist.groupby(clusters).mean()
    # 7. 删除全为 NaN 的列（无任何细胞的簇）
    distance_matrix = distance_matrix.dropna(axis=1, how="all")
    return distance_matrix


def compute_cophenetic_from_distance_matrix(
    distance_matrix: pd.DataFrame,
    method: str = "average",
    show_corr: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    对给定的簇间距离矩阵进行行、列两个方向的层次聚类，并计算 cophenetic 距离矩阵。
    结果将分别对行聚类和列聚类的 cophenetic 距离做独立的线性归一化到 [0,1]。

    参数:
    ----
    distance_matrix : pd.DataFrame
        输入的距离矩阵，行为源簇，列为目标簇（例如 compute_searcher_findee_distance_matrix_from_df 的输出）。
    method : str, optional
        层次聚类使用的 linkage 方法，默认为 "average"。
    show_corr : bool, optional
        是否打印 cophenetic correlation coefficient（行与列分别打印）。默认为 False。

    返回值:
    ------
    Tuple[pd.DataFrame, pd.DataFrame]
        (row_coph, col_coph)。分别为行簇和列簇的 cophenetic 距离矩阵（DataFrame），数值已各自归一化到 [0,1]。
    """
    # 1. 对行进行层次聚类
    row_linkage = linkage(distance_matrix, method=method)
    # 2. 对列进行层次聚类
    col_linkage = linkage(distance_matrix.T, method=method)
    # 3. 计算 cophenetic 距离及相关系数
    row_coph_corr, row_coph_condensed = cophenet(row_linkage, pdist(distance_matrix.values))
    col_coph_corr, col_coph_condensed = cophenet(col_linkage, pdist(distance_matrix.T.values))
    # 4. 将 condensed 距离转为方阵形式
    row_cophenetic_square = squareform(row_coph_condensed)
    col_cophenetic_square = squareform(col_coph_condensed)
    # 5. 构建 DataFrame（保留簇标签）
    row_labels = distance_matrix.index
    col_labels = distance_matrix.columns
    row_cophenetic_df = pd.DataFrame(row_cophenetic_square, index=row_labels, columns=row_labels)
    col_cophenetic_df = pd.DataFrame(col_cophenetic_square, index=col_labels, columns=col_labels)
    # 6. 分别对行、列距离矩阵归一化到 [0,1]
    def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        dmin, dmax = df.values.min(), df.values.max()
        return df if dmin == dmax else (df - dmin) / (dmax - dmin)
    row_cophenetic_df_norm = normalize_df(row_cophenetic_df)
    col_cophenetic_df_norm = normalize_df(col_cophenetic_df)
    # 7. 如需要，打印 cophenetic 相关系数
    if show_corr:
        print(f"Row cophenetic correlation coefficient: {row_coph_corr:.4f}")
        print(f"Column cophenetic correlation coefficient: {col_coph_corr:.4f}")
    # 8. 返回结果矩阵
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
    计算并返回行、列两个维度上的 cophenetic distance 矩阵，
    并在最后将距离分别线性归一化到 [0, 1]。

    如果提供 z_col，则在计算距离时使用 (x, y, z)，否则仅使用 (x, y)。

    参数:
    ----
    df : pd.DataFrame
        包含细胞数据的 DataFrame。
    x_col, y_col, z_col : str, optional
        表示空间坐标的列名。其中 z_col 默认为 None。
    celltype_col : str, optional
        表示细胞类型的列名。
    output_dir : Optional[str]
        输出文件目录；若为 None 则使用当前工作目录。
    method : str, optional
        层次聚类使用的链接方法，默认为 "average"。
    show_corr : bool, optional
        是否打印行、列的 cophenetic correlation coefficient。默认 False。

    返回值:
    ------
    Tuple[pd.DataFrame, pd.DataFrame]
        行和列的 cophenetic 距离矩阵，均已归一化到 [0, 1]。
    """
    # 0. 确保输出目录存在
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    # 1. 计算簇间平均最近邻距离矩阵
    distance_matrix = compute_searcher_findee_distance_matrix_from_df(df, x_col, y_col, z_col, celltype_col)
    # 2. 检查矩阵是否为空
    if distance_matrix.empty:
        raise ValueError("df_group_mean_clean 为空，请检查数据。")
    # 3. 计算 cophenetic 距离矩阵并归一化
    row_coph, col_coph = compute_cophenetic_from_distance_matrix(distance_matrix, method=method, show_corr=show_corr)
    # 4. 返回结果
    return row_coph, col_coph


# ---------------- plot_cophenetic_heatmap.py ----------------
import os
import contextlib
import logging
from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ---------- 1. 让 PDF 中的文字可编辑 ----------
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# ---------- 2. 小工具：静音任意 logger ----------
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


# ---------- 3. 保证存在合法 sans‑serif 字体 ----------
def _ensure_font():
    """Use Arial if present; otherwise switch to Liberation Sans / DejaVu Sans."""
    want = "Arial"
    if any(want in f.name for f in fm.fontManager.ttflist):
        mpl.rcParams["font.family"] = want
        return
    fallback = "Liberation Sans"
    if not any(fallback in f.name for f in fm.fontManager.ttflist):
        fallback = "DejaVu Sans"
    # 覆盖 font.family 与 font.sans-serif 列表，移除 Arial
    mpl.rcParams["font.family"] = [fallback]
    mpl.rcParams["font.sans-serif"] = [fallback]


# ---------- 4. 核心函数 ----------
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
    dpi: int = 300,  # 图像 DPI，影响图像质量
):
    """
    绘制 cophenetic heatmap（seaborn.clustermap），并保证：
      • PDF 文字可编辑
      • 自动调整 legend 位置
      • 动态调整 figsize
      • 静默 fontTools.subset & findfont 日志

    参数:
      ...现有参数...
      return_figure: 是否返回图形对象而不是保存到文件
      return_image: 是否返回高清 PIL 图像而不是图形对象
      dpi: 图像的 DPI 分辨率，仅当 return_image=True 时有效

    返回值:
      如果 return_figure=True，返回 seaborn.ClusterGrid 对象
      如果 return_image=True，返回 PIL.Image 图像对象
      否则返回 None
    """
    # 当同时指定两种返回方式时，优先返回图像
    if return_image:
        return_figure = False

    # ---- 保证有可用字体，避免 findfont 警告 ----
    _ensure_font()

    # ---- 动态 figsize ----
    if figsize is None:
        rows, cols = matrix.shape
        figsize = (max(8.0, 0.25 * cols + 0.5), max(8.0, 0.25 * rows + 0.5))

    # ---- 输出路径 & 标题 ----
    if not (return_figure or return_image):  # 只有在需要保存文件时才处理路径
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

    # 只在需要保存文件时设置路径
    if not (return_figure or return_image):
        pdf_path = os.path.join(output_dir, output_filename or default_pdf)

    # ---- 内部绘图函数 ----
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

        # 1) 保证热图方格是正方形
        g.ax_heatmap.set_aspect("equal")

        # 2) 调整 dendrogram & color‑bar 位置
        if show_dendrogram:
            heat = g.ax_heatmap.get_position()
            row_d = g.ax_row_dendrogram.get_position()
            col_d = g.ax_col_dendrogram.get_position()

            # 2‑1 行 dendrogram 垂直对齐
            g.ax_row_dendrogram.set_position(
                [row_d.x0, heat.y0, row_d.width, heat.height]
            )
            # 2‑2 列 dendrogram 水平对齐
            g.ax_col_dendrogram.set_position(
                [heat.x0, col_d.y0, heat.width, col_d.height]
            )
            # 2‑3 color‑bar 放进左上角空白区域
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

        # 3) 轴标签 & 标题
        g.ax_heatmap.set_xlabel(xlabel, fontsize=12)
        g.ax_heatmap.set_ylabel(ylabel, fontsize=12)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
        g.fig.suptitle(title, fontsize=12, y=1.02)

        # 根据返回类型处理图形
        if return_image:
            # 将图形转为高清图像
            from io import BytesIO
            from PIL import Image

            # 创建内存缓冲区用于保存图像
            buf = BytesIO()
            g.fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
            buf.seek(0)

            # 加载为 PIL 图像
            image = Image.open(buf)
            image_copy = image.copy()  # 创建副本以便关闭原图
            buf.close()
            plt.close(g.fig)  # 关闭图形避免内存泄漏

            return image_copy
        elif not return_figure:
            plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
            plt.close(g.fig)
            return None

        # 返回 ClusterGrid 对象
        return g

    # ---- 执行绘图（可静音日志）----
    if quiet:
        with silence("fontTools.subset", logging.ERROR), silence(
            "matplotlib.font_manager", logging.ERROR
        ):
            result = _draw()
    else:
        result = _draw()

    # 如果保存文件，打印消息
    if not (return_figure or return_image):
        print(f"Heat‑map saved to: {pdf_path}")

    return result
