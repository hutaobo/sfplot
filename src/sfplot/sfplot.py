"""Main module."""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def split_B_by_distance_to_A(
    adata,
    cluster_col="cluster",
    cluster_A="A",
    cluster_B="B",
    threshold=None
):
    """
    根据与 cluster A 的距离远近，将 cluster B 拆分为 B_close 和 B_far。
    其他 cluster 不做改变，依然保持原始标注。

    参数
    ----------
    adata : AnnData
        需要包含以下信息：
         - adata.obs 中包含列 cluster_col, 以及 cell_id（存放每个细胞的唯一标识）
         - adata.obsm["spatial"] 存放空间坐标 (n_cells, 2) 或 (n_cells, 3)
    cluster_col : str, optional
        表示细胞分群的列名，默认为 "cluster"。
    cluster_A : str, optional
        需要作为参照的 cluster 名，默认为 "A"。
    cluster_B : str, optional
        需要被拆分的 cluster 名，默认为 "B"。
    threshold : float, optional
        如果指定了该阈值，则以该值区分 B_close 与 B_far；
        如果不指定，则默认使用 B 细胞到 A 的距离中位数。

    返回
    ----------
    adata : AnnData
        在 adata.obs 中新增一列 cluster_col + "_split_B"，其中：
         - 原先为 A 的细胞标记为 A
         - 原先为 B 的细胞根据距离阈值拆分为 B_close 和 B_far
         - 其他 cluster 不变
        同时在 adata.uns["nearest_cluster_dist"] 中存放各细胞到所有 cluster 最近中心的距离矩阵
    """

    # ------------------------------
    # 1. 基础检查
    # ------------------------------
    if cluster_col not in adata.obs.columns:
        raise ValueError(f"'{cluster_col}' 不存在于 adata.obs 中。请检查列名。")

    if "cell_id" not in adata.obs.columns:
        raise ValueError("'cell_id' 列不存在于 adata.obs 中。请确保数据包含 'cell_id'。")

    if "spatial" not in adata.obsm:
        raise ValueError("'spatial' 坐标信息不存在于 adata.obsm 中。请确保数据包含空间坐标。")

    # ------------------------------
    # 2. 提取 cluster 与坐标信息
    # ------------------------------
    clusters = adata.obs[cluster_col].astype("category")  # cluster 信息
    unique_clusters = clusters.cat.categories  # 不同的 cluster 列表
    coords = adata.obsm["spatial"]  # (n_cells, 2) 或 (n_cells, 3)

    # ------------------------------
    # 3. 计算每个细胞到各个 cluster 最近中心的距离
    # ------------------------------
    df_nearest_cluster_dist = pd.DataFrame(
        index=adata.obs["cell_id"],
        columns=unique_clusters,
        dtype=float
    )

    for c in unique_clusters:
        mask_c = (clusters == c)
        coords_c = coords[mask_c]

        # 如果这个 cluster 没有细胞，则整列置为 NaN
        if coords_c.shape[0] == 0:
            df_nearest_cluster_dist.loc[:, c] = np.nan
            continue

        # 建立最近邻模型并查询所有细胞到该 cluster 最近距离
        nbrs_c = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nbrs_c.fit(coords_c)
        dist_c, _ = nbrs_c.kneighbors(coords)
        df_nearest_cluster_dist[c] = dist_c[:, 0]

    # 将结果存到 adata.uns 中
    adata.uns["nearest_cluster_dist"] = df_nearest_cluster_dist

    # ------------------------------
    # 4. 根据与 A 的距离，将 B 拆分为 B_close 和 B_far
    # ------------------------------
    # 如果用户指定了 cluster_A 或 cluster_B，但实际上 data 里没有，就抛错
    if cluster_A not in unique_clusters:
        raise ValueError(f"'{cluster_A}' 不在已有的 clusters 中: {list(unique_clusters)}")
    if cluster_B not in unique_clusters:
        raise ValueError(f"'{cluster_B}' 不在已有的 clusters 中: {list(unique_clusters)}")

    # 取出每个细胞到 A 的距离
    dist_to_A = df_nearest_cluster_dist[cluster_A]

    # 找到所有属于 B 的细胞的 cell_id
    b_mask = (clusters == cluster_B)
    b_cell_ids = adata.obs.loc[b_mask, "cell_id"]

    # 如果 threshold 未指定，就使用 B 细胞到 A 的距离的中位数
    if threshold is None:
        threshold = dist_to_A.loc[b_cell_ids].median()

    # ------------------------------
    # 5. 生成新的分群列 cluster_col+"_split_B"
    # ------------------------------
    new_cluster_col = cluster_col + "_split_" + cluster_B + "_by_" + cluster_A

    # 先复制原始 cluster 列
    adata.obs[new_cluster_col] = adata.obs[cluster_col].astype(str)

    # 对于属于 B 的细胞，根据阈值拆分
    for idx in b_cell_ids:
        if dist_to_A.loc[idx] < threshold:
            adata.obs.loc[adata.obs["cell_id"] == idx, new_cluster_col] = f"{cluster_B}_close"
        else:
            adata.obs.loc[adata.obs["cell_id"] == idx, new_cluster_col] = f"{cluster_B}_far"

    # ------------------------------
    # 返回更新后的 adata
    # ------------------------------
    return adata
