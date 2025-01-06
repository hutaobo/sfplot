# sfplot/plotting.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from typing import Optional

from .data_processing import load_xenium_data  # 确保这个导入路径正确


def generate_cluster_distance_heatmap_from_path(
    base_path: str,
    sample: str,
    figsize: tuple = (8, 8),
    output_dir: Optional[str] = None
):
    """
    生成并保存每个细胞群到最近群中心的距离热图。

    参数:
    --------
    base_path : str
        数据所在的基础路径。
    sample : str
        样本名称，用于指定具体的数据文件夹。
    output_dir : Optional[str]
        输出 PDF 文件的目录。默认为当前工作目录。

    返回值:
    --------
    None
    """
    # 设置输出目录
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # 拼接路径并读取数据
    folder = os.path.join(base_path, sample)
    adata = load_xenium_data(folder)

    # 1. 提取坐标和 cluster 信息
    coords = adata.obsm["spatial"]  # (n_cells, 2)或(n_cells, 3)
    clusters = adata.obs["Cluster"].astype("category")  # cluster 信息
    unique_clusters = clusters.cat.categories  # 不同的 cluster 列表

    # ---------------- (关键修改) ----------------
    # 2. 新建一个结果 DataFrame，用于存放各细胞到每个 cluster 最近中心的距离
    #    注意这里使用 adata.obs["cell_id"] 作为行索引
    df_nearest_cluster_dist = pd.DataFrame(
        index=adata.obs["cell_id"],
        columns=unique_clusters,
        dtype=float
    )

    # 3. 对每个 cluster，构建邻居模型并查询所有细胞到该 cluster 的最近距离
    for c in unique_clusters:
        # 3.1 取出该 cluster 下的细胞坐标
        mask_c = (clusters == c)
        coords_c = coords[mask_c]

        # 如果这个 cluster 没有细胞，则整个列都置为 NaN
        if coords_c.shape[0] == 0:
            df_nearest_cluster_dist.loc[:, c] = np.nan
            continue

        # 3.2 建立最近邻模型
        nbrs_c = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nbrs_c.fit(coords_c)

        # 3.3 查询所有细胞到该 cluster 最近的距离
        dist_c, _ = nbrs_c.kneighbors(coords)
        df_nearest_cluster_dist[c] = dist_c[:, 0]

    # 4. 将结果保存到 adata.uns 中（或其他合适的位置）
    adata.uns["nearest_cluster_dist"] = df_nearest_cluster_dist

    # ------------------- 以下为距离矩阵的层次聚类可视化 -------------------
    # 5. 建立一个 Series，让它的 index 也是 cell_id，值是 cluster，这样后续 groupby 才能对得上
    clusters_by_id = pd.Series(
        data=clusters.values,  # cluster 的值
        index=adata.obs["cell_id"],  # 与 df_nearest_cluster_dist.index 对齐
        name="Cluster"
    )

    # 然后以 cell_id 为索引，对 df_nearest_cluster_dist 按 cluster 分组并计算均值
    df_group_mean = df_nearest_cluster_dist.groupby(clusters_by_id).mean()

    # 6. 因为有可能某些列全部是 NaN，也可以选择只删除整列全 NaN 的情况
    #    如果你不需要删除也可以不做这一步
    df_group_mean_clean = df_group_mean.dropna(axis=1, how="all")

    if df_group_mean_clean.empty:
        print(f"Warning: df_group_mean_clean is empty for sample {sample}.")
        print("Check if there are clusters that exist in the data.")
        # 如果需要，可以继续处理其他样本
        return

    # 用 clustermap 对该矩阵进行聚类可视化
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

    # 调整热图的纵横比
    g.ax_heatmap.set_aspect("equal")

    # 设置轴标签和标题
    g.ax_heatmap.set_xlabel("Findee", fontsize=12)
    g.ax_heatmap.set_ylabel("Searcher", fontsize=12)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)

    # 设置整个图形的标题，而不是 heatmap 的标题
    g.fig.suptitle(f"{sample} Cluster Distance Heatmap", fontsize=12, y=1)

    # 7. 保存为带有样本名的 PDF
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
    max_scale: float = 10
):
    """
    生成并保存每个细胞群到最近群中心的距离热图。

    参数:
    --------
    adata : anndata.AnnData
        包含预处理数据的 AnnData 对象。
    cluster_col : str, optional
        `adata.obs` 中包含 cluster 信息的列名。默认为 "Cluster"。
    output_dir : Optional[str]
        输出 PDF 文件的目录。默认为当前工作目录。
    output_filename : Optional[str]
        输出文件的名称。如果未指定，将使用 "clustermap_output_{sample}.pdf" 格式。
    figsize : tuple, optional
        热图的大小。默认为 (7, 7)。
    cmap : str, optional
        热图的颜色映射。默认为 "RdBu"。
    max_scale : float, optional
        `sc.pp.scale` 的 `max_value` 参数，用于裁剪 Z-score。默认为 10。

    返回值:
    --------
    None
    """
    # 设置输出目录
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # 检查 cluster_col 是否存在于 adata.obs
    if cluster_col not in adata.obs.columns:
        raise ValueError(f"'{cluster_col}' 不存在于 adata.obs 中。请检查列名。")

    # 提取 cluster 信息
    clusters = adata.obs[cluster_col].astype("category")  # cluster 信息
    unique_clusters = clusters.cat.categories  # 不同的 cluster 列表

    # 1. 提取坐标信息
    if "spatial" not in adata.obsm:
        raise ValueError("'spatial' 坐标信息不存在于 adata.obsm 中。请确保数据包含空间坐标。")
    coords = adata.obsm["spatial"]  # (n_cells, 2)或(n_cells, 3)

    # 2. 新建一个结果 DataFrame，用于存放各细胞到每个 cluster 最近中心的距离
    #    注意这里使用 adata.obs["cell_id"] 作为行索引
    if "cell_id" not in adata.obs.columns:
        raise ValueError("'cell_id' 列不存在于 adata.obs 中。请确保数据包含 'cell_id'。")
    df_nearest_cluster_dist = pd.DataFrame(
        index=adata.obs["cell_id"],
        columns=unique_clusters,
        dtype=float
    )

    # 3. 对每个 cluster，构建邻居模型并查询所有细胞到该 cluster 的最近距离
    for c in unique_clusters:
        # 3.1 取出该 cluster 下的细胞坐标
        mask_c = (clusters == c)
        coords_c = coords[mask_c]

        # 如果这个 cluster 没有细胞，则整个列都置为 NaN
        if coords_c.shape[0] == 0:
            df_nearest_cluster_dist.loc[:, c] = np.nan
            continue

        # 3.2 建立最近邻模型
        nbrs_c = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nbrs_c.fit(coords_c)

        # 3.3 查询所有细胞到该 cluster 最近的距离
        dist_c, _ = nbrs_c.kneighbors(coords)
        df_nearest_cluster_dist[c] = dist_c[:, 0]

    # 4. 将结果保存到 adata.uns 中（或其他合适的位置）
    adata.uns["nearest_cluster_dist"] = df_nearest_cluster_dist

    # ------------------- 以下为距离矩阵的层次聚类可视化 -------------------
    # 5. 建立一个 Series，让它的 index 也是 cell_id，值是 cluster，这样后续 groupby 才能对得上
    clusters_by_id = pd.Series(
        data=clusters.values,  # cluster 的值
        index=adata.obs["cell_id"],  # 与 df_nearest_cluster_dist.index 对齐
        name=cluster_col
    )

    # 以 cell_id 为索引，对 df_nearest_cluster_dist 按 cluster 分组并计算均值
    df_group_mean = df_nearest_cluster_dist.groupby(clusters_by_id).mean()

    # 6. 删除整列全 NaN 的情况
    df_group_mean_clean = df_group_mean.dropna(axis=1, how="all")

    if df_group_mean_clean.empty:
        print(f"Warning: df_group_mean_clean is empty. 请检查数据中是否存在 cluster。")
        return

    # 用 clustermap 对该矩阵进行聚类可视化
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

    # 调整热图的纵横比
    g.ax_heatmap.set_aspect("equal")

    # 设置轴标签和标题
    g.ax_heatmap.set_xlabel("Findee", fontsize=12)
    g.ax_heatmap.set_ylabel("Searcher", fontsize=12)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)

    # 设置整个图形的标题，而不是 heatmap 的标题
    sample = adata.uns.get("sample", "Sample")  # 假设你在 adata.uns 中保存了样本名
    g.fig.suptitle(f"{sample} Cluster Distance Heatmap", fontsize=12, y=1)

    # 7. 保存为带有样本名的 PDF
    if output_filename is None:
        output_filename = f"SFplot_of_{sample}.pdf"
    output_file = os.path.join(output_dir, output_filename)
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"Cluster distance heatmap saved to {output_file}")
