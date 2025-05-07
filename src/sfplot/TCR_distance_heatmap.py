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
    根据 df_tr_subset 中的坐标 (x, y) 以及 cluster 信息 (feature_name)，计算每个细胞/行到各 cluster 最近中心的距离，
    并对距离矩阵按 cluster 分组求均值，再用 clustermap 进行层次聚类热图可视化。

    同时演示：
    1. 手动缩小并移动颜色 legend。
    2. 在热图方格保持正方形的情况下，对齐行/列 dendrogram。
    """

    # 若未指定输出目录，则使用当前工作目录
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------
    # 1. 准备坐标和 cluster 信息
    # ------------------------
    coords = df_tr_subset[[x_col, y_col]].values  # shape: (n_cells, 2)
    clusters = df_tr_subset[cluster_col].astype("category")
    unique_clusters = clusters.cat.categories

    # ------------------------
    # 2. 建立空的距离矩阵 DF
    # ------------------------
    df_nearest_cluster_dist = pd.DataFrame(index=df_tr_subset.index, dtype=float)

    # ------------------------
    # 3. 对每个 cluster，计算距离
    # ------------------------
    from sklearn.neighbors import NearestNeighbors
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
    # 4. 分组聚合 (按 cluster)
    # ------------------------
    clusters_by_id = pd.Series(data=clusters.values, index=df_tr_subset.index, name=cluster_col)
    df_group_mean = df_nearest_cluster_dist.groupby(clusters_by_id).mean()

    # ------------------------
    # 5. 替换 inf/-inf 为 NaN
    # ------------------------
    df_group_mean = df_group_mean.replace([np.inf, -np.inf], np.nan)

    # ------------------------
    # 6. 根据 dropna_axis 删除全 NaN 行/列 (可选)
    # ------------------------
    if dropna_axis in ["rows", "index"]:
        df_group_mean = df_group_mean.dropna(axis=0, how="all")
    elif dropna_axis in ["columns", "cols"]:
        df_group_mean = df_group_mean.dropna(axis=1, how="all")
    elif dropna_axis == "both":
        df_group_mean = df_group_mean.dropna(axis=0, how="all").dropna(axis=1, how="all")
    # 如果是 "none" 则不做任何全空丢弃

    # ------------------------
    # 7. 删除任何包含 NaN 的行/列，确保输入全为 finite
    # ------------------------
    df_group_mean = df_group_mean.dropna(axis=0, how="any")
    df_group_mean = df_group_mean.dropna(axis=1, how="any")

    # ------------------------
    # 8. 最终检查是否还有 NaN / inf
    # ------------------------
    if not np.all(np.isfinite(df_group_mean.values)):
        print(f"[Error] Even after cleaning, df_group_mean still has non-finite values for sample={sample}.")
        return

    # 若清理后已经没有足够行或列（至少需要 >=2），则无法做层次聚类
    if df_group_mean.shape[0] < 2 or df_group_mean.shape[1] < 2:
        print(f"Warning: After cleaning, df_group_mean shape={df_group_mean.shape}, not enough for clustermap.")
        return

    # ============ 这里开始是核心画图部分，包含颜色条和对齐的改动 ============

    # 9. clustermap 可视化
    plt.figure(figsize=figsize)
    g = sns.clustermap(
        df_group_mean,
        cmap="RdBu",
        figsize=figsize,
        row_cluster=True,
        col_cluster=True,
        linewidths=0.5,
        annot=False,
        # 设置 colorbar 的位置和大小（相对于整幅图的百分比）
        # (x, y, width, height), 取值一般在 [0,1] 之间
        # x=0：距离图左边界 2% 的位置
        # y=0.9：距离图底部 90% 的位置（也就是比较靠上）
        # width=0.03：颜色条宽度占整图宽度的 3%
        # height=0.09：颜色条高度占整图高度的 9%
        cbar_pos=(0, 0.9, 0.03, 0.09),
        cbar_kws={"orientation": "vertical"}  # 竖直放置，可自行改成 "horizontal"
    )

    # 设置热图单元格为正方形
    g.ax_heatmap.set_aspect("equal")

    # 修正行 dendrogram 与热图在 y 方向上的对齐
    # 拿到各自的 (x0, y0, width, height)
    row_dendro_pos = g.ax_row_dendrogram.get_position()
    heatmap_pos = g.ax_heatmap.get_position()
    # 让行 dendrogram 的顶部和底部与热图对齐
    g.ax_row_dendrogram.set_position([
        row_dendro_pos.x0,
        heatmap_pos.y0,          # 用热图的 y0
        row_dendro_pos.width,
        heatmap_pos.height       # 用热图的 height
    ])

    # 同理，修正列 dendrogram 与热图在 x 方向上的对齐（可选）
    col_dendro_pos = g.ax_col_dendrogram.get_position()
    g.ax_col_dendrogram.set_position([
        heatmap_pos.x0,          # 用热图的 x0
        col_dendro_pos.y0,
        heatmap_pos.width,       # 用热图的 width
        col_dendro_pos.height
    ])

    # 设置轴标签
    g.ax_heatmap.set_xlabel("Findee", fontsize=10)
    g.ax_heatmap.set_ylabel("Searcher", fontsize=10)

    # y 轴标签旋转 0 度，方便阅读
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)

    # 设置整个图形的标题
    g.fig.suptitle(f"TCR_SFplot of {sample}", fontsize=12, y=1.02)

    # 10. 保存 PDF
    output_file = os.path.join(output_dir, f"TCR_SFplot_of_{sample}.pdf")
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Sample {sample} done. PDF saved to {output_file}")
