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


# sfplot/plot_cophenetic_heatmap.py

import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


def plot_cophenetic_heatmap(
    matrix: "pd.DataFrame",
    matrix_name: Optional[str] = None,  # <-- 新增的参数，用于区分 row_coph 和 col_coph
    output_dir: Optional[str] = None,
    output_filename: Optional[str] = None,
    figsize: tuple = (8, 8),
    cmap: str = "RdBu",
    linewidths: float = 0.5,
    annot: bool = False,
    sample: str = "Sample",
    xlabel: str = None,
    ylabel: str = None
):
    """
    对给定矩阵进行 clustermap 可视化，并修正行列 dendrogram 的位置。

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
    """

    # 1) 设置输出目录
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # 根据 matrix_name 设置标题、默认文件名、以及轴标签
    if matrix_name == "row_coph":
        title_str = f"Searcher's D score of {sample}"
        default_filename = f"Searcher's D score_of_{sample}.pdf"
        # 覆盖 x、y label
        xlabel = "Searcher"
        ylabel = "Searcher"
    elif matrix_name == "col_coph":
        title_str = f"Findee's D score of {sample}"
        default_filename = f"Findee's D score_of_{sample}.pdf"
        # 覆盖 x、y label
        xlabel = "Findee"
        ylabel = "Findee"
    else:
        # 若没提供或提供了其他名称，则沿用原先的写法（可根据需要自行调整）
        title_str = f"D score of {sample}"
        default_filename = f"D_score_of_{sample}.pdf"
        # 若不需要覆盖，可让用户自定义或提供默认
        if xlabel is None:
            xlabel = "Findee"
        if ylabel is None:
            ylabel = "Searcher"

    # 2) 生成 clustermap
    plt.figure(figsize=figsize)
    g = sns.clustermap(
        data=matrix,
        cmap=cmap,
        figsize=figsize,
        row_cluster=True,
        col_cluster=True,
        linewidths=linewidths,
        annot=annot
    )

    # 3) 设置热图单元格为正方形
    g.ax_heatmap.set_aspect("equal")

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

    # 6) 设置轴标签和标题
    g.ax_heatmap.set_xlabel(xlabel, fontsize=12)
    g.ax_heatmap.set_ylabel(ylabel, fontsize=12)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)

    # 设置整个图形的标题
    g.fig.suptitle(title_str, fontsize=12, y=1)

    # 获取当前 color legend 的位置
    cax_pos = g.cax.get_position()
    # 将 color legend 的宽度和高度各缩小 20%
    g.cax.set_position([cax_pos.x0, cax_pos.y0, cax_pos.width * 0.8, cax_pos.height * 0.6])

    # 7) 保存为 PDF
    if output_filename is None:
        output_filename = default_filename
    output_file = os.path.join(output_dir, output_filename)
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"Heatmap saved to: {output_file}")
