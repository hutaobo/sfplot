"""Main module."""

import numpy as np
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
        在 adata.obs 中新增一列 new_cluster_col，其中：
         - 原先为 A 的细胞标记为 A
         - 原先为 B 的细胞根据距离阈值拆分为 B_close 和 B_far
         - 其他 cluster 不变
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

    clusters = adata.obs[cluster_col].astype("category")
    unique_clusters = clusters.cat.categories
    coords = adata.obsm["spatial"]

    if cluster_A not in unique_clusters:
        raise ValueError(f"'{cluster_A}' 不在已有的 clusters 中: {list(unique_clusters)}")
    if cluster_B not in unique_clusters:
        raise ValueError(f"'{cluster_B}' 不在已有的 clusters 中: {list(unique_clusters)}")

    # ------------------------------
    # 2. 构建只用于 A 的最近邻模型
    # ------------------------------
    mask_A = (clusters == cluster_A)
    coords_A = coords[mask_A]
    if coords_A.shape[0] == 0:
        raise ValueError(f"cluster_A='{cluster_A}' 未包含任何细胞，无法计算距离。")

    nbrs_A = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nbrs_A.fit(coords_A)

    # ------------------------------
    # 3. 仅计算 B 细胞到 A 的最近距离
    # ------------------------------
    b_mask = (clusters == cluster_B)
    coords_B = coords[b_mask]
    if coords_B.shape[0] == 0:
        raise ValueError(f"cluster_B='{cluster_B}' 未包含任何细胞，无法拆分。")

    dist_to_A_for_B, _ = nbrs_A.kneighbors(coords_B)
    dist_to_A_for_B = dist_to_A_for_B[:, 0]  # shape: (num_B_cells, )

    # ------------------------------
    # 4. 确定 threshold，用于区分 B_close 与 B_far
    # ------------------------------
    if threshold is None:
        threshold = np.median(dist_to_A_for_B)

    # ------------------------------
    # 5. 生成新的分群列
    # ------------------------------
    new_cluster_col = f"split_{cluster_B}_by_{cluster_A}_in_{cluster_col}"
    adata.obs[new_cluster_col] = adata.obs[cluster_col].astype(str)

    # 使用真正的行索引标签，而非整数索引
    b_indices = adata.obs.index[b_mask]  # B 细胞对应的 DataFrame 行索引标签

    # dist_to_A_for_B < threshold 给出 B 细胞中谁是 close/far
    b_close_mask_in_B = (dist_to_A_for_B < threshold)
    b_far_mask_in_B = ~b_close_mask_in_B

    # 分别拿到 close/far 的行索引标签
    b_close_indices = b_indices[b_close_mask_in_B]
    b_far_indices = b_indices[b_far_mask_in_B]

    # 然后用 .loc[...] 用索引标签进行赋值
    adata.obs.loc[b_close_indices, new_cluster_col] = f"{cluster_B}_close"
    adata.obs.loc[b_far_indices, new_cluster_col] = f"{cluster_B}_far"

    print("The new cluster col is:", new_cluster_col)

    return adata
