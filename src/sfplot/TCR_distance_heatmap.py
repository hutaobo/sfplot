import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from typing import Optional


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

    主要增加了严格的数据清理，确保在做 clustermap 之前，
    DataFrame 没有任何 NaN 或 inf，以避免 "The condensed distance matrix must contain only finite values." 的错误。

    参数:
    --------
    df_tr_subset : pd.DataFrame
        必须包含 x, y, feature_name 这三个列（列名可由 x_col, y_col, cluster_col 参数指定）。
    x_col : str
        x 坐标列的名称，默认为 "x"。
    y_col : str
        y 坐标列的名称，默认为 "y"。
    cluster_col : str
        表示 cluster/feature 分组的列名，默认为 "feature_name"。
    sample : str
        样本名称，会用于输出文件的命名以及图形标题。
    figsize : tuple
        clustermap 的图像大小，默认为 (8, 8)。
    output_dir : Optional[str]
        输出 PDF 文件的目录。若不指定，默认为当前工作目录。
    dropna_axis : str
        原本只删除全 NaN 行/列的策略。此处保留此参数，但我们会在后面做更严格的 NaN 清理。
        可选 "rows"/"columns"/"both"/"none"。
        默认为 "columns"。

    返回:
    --------
    None
    """

    # 若未指定输出目录，则使用当前工作目录
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------
    # 1. 准备坐标和 cluster 信息
    # ------------------------
    coords = df_tr_subset[[x_col, y_col]].values  # shape: (n_cells, 2)
    clusters = df_tr_subset[cluster_col].astype("category")  # 将 feature_name 转为 category
    unique_clusters = clusters.cat.categories  # 不同的 feature/cluster

    # ------------------------
    # 2. 建立空的距离矩阵 DF
    # ------------------------
    #   注意先不预先创建所有列，以避免出现空 cluster 的全 NaN 列
    df_nearest_cluster_dist = pd.DataFrame(index=df_tr_subset.index, dtype=float)

    # ------------------------
    # 3. 对每个 cluster，计算距离
    # ------------------------
    for c in unique_clusters:
        mask_c = (clusters == c)
        coords_c = coords[mask_c]
        if coords_c.shape[0] == 0:
            # 跳过空 cluster，不创建任何列
            continue

        # 建立最近邻模型
        nbrs_c = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nbrs_c.fit(coords_c)
        dist_c, _ = nbrs_c.kneighbors(coords)  # shape: (n_cells, 1)

        # 新增一列到 df_nearest_cluster_dist
        df_nearest_cluster_dist[c] = dist_c[:, 0]

    # ------------------------
    # 4. 分组聚合 (按 cluster)
    # ------------------------
    clusters_by_id = pd.Series(data=clusters.values, index=df_tr_subset.index, name=cluster_col)
    # 求出各搜索者（行所在 cluster）对目标列（各 cluster）的平均距离
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
    # 7. 进一步删除任何包含 NaN 的行/列
    #    这样可以确保 clustermap 输入全为 finite
    # ------------------------
    #   如果你不想这么“激进”，可以注释掉这两行，但就要自己确保没有 NaN。
    df_group_mean = df_group_mean.dropna(axis=0, how="any")
    df_group_mean = df_group_mean.dropna(axis=1, how="any")

    # ------------------------
    # 8. 最终检查是否还有 NaN / inf
    # ------------------------
    #   如果还有，就直接退出，避免 clustermap 报错
    if not np.all(np.isfinite(df_group_mean.values)):
        print(f"[Error] Even after cleaning, df_group_mean still has non-finite values for sample={sample}.")
        print("Check your data or consider additional cleaning steps.")
        return

    # 若清理后已经没有足够行或列（至少需要 >=2 才能做层次聚类，否则也会报错）
    if df_group_mean.shape[0] < 2 or df_group_mean.shape[1] < 2:
        print(f"Warning: After cleaning, df_group_mean shape={df_group_mean.shape}, not enough for clustermap.")
        return

    # ------------------------
    # 9. clustermap 可视化
    # ------------------------
    plt.figure(figsize=figsize)
    g = sns.clustermap(
        df_group_mean,
        cmap="RdBu",
        figsize=figsize,
        row_cluster=True,
        col_cluster=True,
        linewidths=0.5,
        annot=False
    )

    # 调整热图纵横比
    g.ax_heatmap.set_aspect("equal")

    # 设置轴标签
    g.ax_heatmap.set_xlabel("Findee", fontsize=10)
    g.ax_heatmap.set_ylabel("Searcher", fontsize=10)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)

    # 设置整个图形的标题
    g.fig.suptitle(f"TCR_SFplot of {sample}", fontsize=10, y=1)

    # ------------------------
    # 10. 保存为 PDF
    # ------------------------
    output_file = os.path.join(output_dir, f"TCR_SFplot_of_{sample}.pdf")
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Sample {sample} done. PDF saved to {output_file}")
