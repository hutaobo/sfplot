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


import os
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
from typing import Optional, Tuple

def compute_cophenetic_distances_from_df(
    df: pd.DataFrame,
    x_col: str = 'x',
    y_col: str = 'y',
    celltype_col: str = 'celltype',
    output_dir: Optional[str] = None,
    method: str = 'average'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    计算并返回行、列两个维度上的 cophenetic distance 矩阵，
    并在最后将距离分别做线性归一化到 [0,1]。

    参数:
    --------
    df : pd.DataFrame
        包含细胞数据的 DataFrame。
    x_col : str, optional
        表示 x 坐标的列名。默认为 'x'。
    y_col : str, optional
        表示 y 坐标的列名。默认为 'y'。
    celltype_col : str, optional
        表示细胞类型的列名。默认为 'celltype'。
    output_dir : Optional[str]
        输出文件的目录。默认为当前工作目录。
    method : str, optional
        层次聚类中使用的链接方法。默认为 'average'。

    返回值:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        返回行和列的 cophenetic distance 矩阵，均已归一化到 [0,1]。
    """

    # 0. 可选: 处理输出目录
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # 1. 检查必要的列是否存在于 DataFrame 中
    required_columns = {x_col, y_col, celltype_col}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame 必须包含以下列：{required_columns}")

    # 2. 提取 cluster 信息
    clusters = df[celltype_col].astype('category')
    unique_clusters = clusters.cat.categories

    # 3. 提取空间坐标
    coords = df[[x_col, y_col]].values  # (n_cells, 2)

    # 4. 构建一个 DataFrame, 行是索引, 列是 cluster
    df_nearest_cluster_dist = pd.DataFrame(
        index=df.index,
        columns=unique_clusters,
        dtype=float
    )

    # 5. 对每个 cluster, 用最近邻模型计算所有细胞到该 cluster 最近中心的距离
    for c in unique_clusters:
        mask_c = (clusters == c)
        coords_c = coords[mask_c]

        if coords_c.shape[0] == 0:
            df_nearest_cluster_dist.loc[:, c] = np.nan
            continue

        nbrs_c = NearestNeighbors(n_neighbors=1, algorithm='auto')
        nbrs_c.fit(coords_c)
        dist_c, _ = nbrs_c.kneighbors(coords)
        df_nearest_cluster_dist[c] = dist_c[:, 0]

    # 6. 对每个 cluster 求距离均值 => cluster x cluster 矩阵
    df_group_mean = df_nearest_cluster_dist.groupby(clusters).mean()

    # 7. 删除整列全 NaN 的 cluster
    df_group_mean_clean = df_group_mean.dropna(axis=1, how='all')
    if df_group_mean_clean.empty:
        print("Warning: df_group_mean_clean is empty. 请检查数据。")
        return pd.DataFrame(), pd.DataFrame()

    # 8. 分别对行、列做层次聚类
    row_linkage = linkage(df_group_mean_clean, method=method)
    col_linkage = linkage(df_group_mean_clean.T, method=method)

    # 9. 计算 cophenetic 距离
    row_coph_corr, row_coph_condensed = cophenet(row_linkage, pdist(df_group_mean_clean.values))
    col_coph_corr, col_coph_condensed = cophenet(col_linkage, pdist(df_group_mean_clean.T.values))

    # 10. 将 condensed 距离转为方阵
    row_cophenetic_square = squareform(row_coph_condensed)
    col_cophenetic_square = squareform(col_coph_condensed)

    # 11. 构建 DataFrame
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

    # 12. 分别对行、列的距离矩阵进行归一化
    def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        dmin = df.values.min()
        dmax = df.values.max()
        if dmin == dmax:
            return df
        return (df - dmin) / (dmax - dmin)

    row_cophenetic_df_norm = normalize_df(row_cophenetic_df)
    col_cophenetic_df_norm = normalize_df(col_cophenetic_df)

    # 打印 cophenetic correlation coefficient
    print(f"Row cophenetic correlation coefficient: {row_coph_corr:.4f}")
    print(f"Column cophenetic correlation coefficient: {col_coph_corr:.4f}")

    return row_cophenetic_df_norm, col_cophenetic_df_norm


# sfplot/plot_cophenetic_heatmap.py

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional

def plot_cophenetic_heatmap(
    matrix: "pd.DataFrame",
    matrix_name: Optional[str] = None,  # 用于区分 row_coph 和 col_coph
    output_dir: Optional[str] = None,
    output_filename: Optional[str] = None,
    figsize: tuple = (8, 8),
    cmap: str = "RdBu",
    linewidths: float = 0.5,
    annot: bool = False,
    sample: str = "Sample",
    xlabel: str = None,
    ylabel: str = None,
    show_dendrogram: bool = True  # 新增参数：是否绘制 dendrogram，默认绘制
):
    """
    对给定矩阵进行 clustermap 可视化，并调整行、列 dendrogram 及 color legend 的位置，
    使得 color legend 总是位于左上角的空白区域中，且不会过大以致与 dendrogram 重叠。

    参数
    ----
    matrix : pd.DataFrame
        待可视化的距离或相似度矩阵 (可为对称矩阵)。
    matrix_name : str, optional
        矩阵的名称，用于区分是 "row_coph" 还是 "col_coph"。
    output_dir : str, optional
        保存输出文件的目录。若为 None，使用当前工作目录。
    output_filename : str, optional
        保存输出文件名。若为 None，则根据 matrix_name 和 sample 决定。
    figsize : tuple, optional
        图像大小，默认 (8,8)。
    cmap : str, optional
        颜色映射，默认 "RdBu"。
    linewidths : float, optional
        热图单元格之间的间隔线宽，默认 0.5。
    annot : bool, optional
        是否在格子中显示数值，默认 False。
    sample : str, optional
        用于设置图形标题中的样本名称，默认 "Sample"。
    xlabel : str, optional
        热图 X 轴标签。
    ylabel : str, optional
        热图 Y 轴标签。
    show_dendrogram : bool, optional
        是否绘制 dendrogram，默认 True（绘制）。如果为 False，则不绘制 dendrogram，且禁用行、列聚类。
    """

    # 1) 设置输出目录
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # 根据 matrix_name 设置标题、默认文件名以及轴标签
    if matrix_name == "row_coph":
        title_str = f"Searcher's D score of {sample}"
        default_filename = f"Searcher's D score_of_{sample}.pdf"
        xlabel = "Searcher"
        ylabel = "Searcher"
    elif matrix_name == "col_coph":
        title_str = f"Findee's D score of {sample}"
        default_filename = f"Findee's D score_of_{sample}.pdf"
        xlabel = "Findee"
        ylabel = "Findee"
    else:
        title_str = f"D score of {sample}"
        default_filename = f"D_score_of_{sample}.pdf"
        if xlabel is None:
            xlabel = "Findee"
        if ylabel is None:
            ylabel = "Searcher"

    # 根据 show_dendrogram 参数确定是否进行聚类
    row_cluster = show_dendrogram
    col_cluster = show_dendrogram

    # 2) 生成 clustermap
    plt.figure(figsize=figsize)
    g = sns.clustermap(
        data=matrix,
        cmap=cmap,
        figsize=figsize,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        linewidths=linewidths,
        annot=annot
    )

    # 3) 设置热图单元格为正方形
    g.ax_heatmap.set_aspect("equal")

    # 如果绘制 dendrogram，则调整 dendrogram 和 color legend 的位置
    if show_dendrogram:
        # 4) 修正行 dendrogram 与热图在 y 方向上的对齐
        row_dendro_pos = g.ax_row_dendrogram.get_position()
        heatmap_pos = g.ax_heatmap.get_position()
        g.ax_row_dendrogram.set_position([
            row_dendro_pos.x0,
            heatmap_pos.y0,
            row_dendro_pos.width,
            heatmap_pos.height
        ])

        # 5) 修正列 dendrogram 与热图在 x 方向上的对齐
        col_dendro_pos = g.ax_col_dendrogram.get_position()
        g.ax_col_dendrogram.set_position([
            heatmap_pos.x0,
            col_dendro_pos.y0,
            heatmap_pos.width,
            col_dendro_pos.height
        ])

        # 6) 调整 color legend（g.cax）位置
        # 计算左上角的空白区域：
        # 该区域的水平范围为：从 row dendrogram 的左边界到热图左边界；
        # 垂直范围为：从 col dendrogram 的上边界到热图的上边界。
        empty_left = g.ax_row_dendrogram.get_position().x0
        empty_right = heatmap_pos.x0
        empty_width = empty_right - empty_left

        col_dendro_bbox = g.ax_col_dendrogram.get_position()
        empty_bottom = col_dendro_bbox.y0 + col_dendro_bbox.height
        empty_top = heatmap_pos.y0 + heatmap_pos.height
        empty_height = empty_top - empty_bottom

        # 为避免 legend 太大，取空白区域的 80% 大小，并居中放置
        cbar_width = empty_width * 0.3
        cbar_height = empty_height * 0.7
        cbar_x = empty_left + (empty_width - cbar_width) / 2
        cbar_y = empty_bottom + (empty_height - cbar_height) / 2

        g.cax.set_position([cbar_x, cbar_y, cbar_width, cbar_height])

    # 7) 设置轴标签和标题
    g.ax_heatmap.set_xlabel(xlabel, fontsize=12)
    g.ax_heatmap.set_ylabel(ylabel, fontsize=12)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
    g.fig.suptitle(title_str, fontsize=12, y=1)

    # 8) 保存为 PDF
    if output_filename is None:
        output_filename = default_filename
    output_file = os.path.join(output_dir, output_filename)
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"Heatmap saved to: {output_file}")
