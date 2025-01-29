import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

def compute_cophenetic_distances_from_adata (
    adata: 'anndata.AnnData',
    cluster_col: str = "Cluster",
    output_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    计算并返回行、列两个维度上的 cophenetic distance 矩阵。

    修改后功能:
    1. 提取各 cluster 间的平均距离矩阵 (df_group_mean_clean)。
    2. 对该矩阵做层次聚类(行、列分别聚类)。
    3. 计算并返回行与列各自的 cophenetic distance。

    参数:
    --------
    adata : anndata.AnnData
        包含预处理数据的 AnnData 对象。
    cluster_col : str, optional
        `adata.obs` 中包含 cluster 信息的列名。默认为 "Cluster"。

    返回:
    --------
    (row_cophenetic_df, col_cophenetic_df):
        分别是行和列 dendrogram 的 cophenetic 距离矩阵 (DataFrame 格式)。
    """
    # 设置输出目录(本例中不再保存到文件, 所以仅在必要时创建)
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

    # 4. 将结果保存到 adata.uns 中（可选）
    adata.uns["nearest_cluster_dist"] = df_nearest_cluster_dist

    # 5. 建立一个 Series，让它的 index 也是 cell_id，值是 cluster
    clusters_by_id = pd.Series(
        data=clusters.values,  # cluster 的值
        index=adata.obs["cell_id"],  # 与 df_nearest_cluster_dist.index 对齐
        name=cluster_col
    )

    # 按 cluster 分组并计算均值，得到 df_group_mean
    df_group_mean = df_nearest_cluster_dist.groupby(clusters_by_id).mean()

    # 6. 删除整列全 NaN 的情况
    df_group_mean_clean = df_group_mean.dropna(axis=1, how="all")

    if df_group_mean_clean.empty:
        print("Warning: df_group_mean_clean is empty. 请检查数据中是否存在 cluster。")
        # 在此种情况下返回空的 DataFrame
        return pd.DataFrame(), pd.DataFrame()

    # ------------------- 以下为计算 cophenetic distance -------------------
    # 对行进行层次聚类
    row_linkage = linkage(df_group_mean_clean, method='average')
    # 对列进行层次聚类（转置后聚类）
    col_linkage = linkage(df_group_mean_clean.T, method='average')

    # 从 linkage 中计算 pairwise cophenetic distance (以 condensed 距离矩阵形式)
    row_coph_condensed = cophenet(row_linkage)
    col_coph_condensed = cophenet(col_linkage)

    # 将 condensed 距离矩阵转换为 square form
    row_cophenetic_square = squareform(row_coph_condensed)
    col_cophenetic_square = squareform(col_coph_condensed)

    # 构建带行、列标签的 DataFrame
    row_labels = df_group_mean_clean.index
    col_labels = df_group_mean_clean.columns

    # row_cophenetic_df 的 index 和 columns 都是 row_labels
    row_cophenetic_df = pd.DataFrame(
        row_cophenetic_square,
        index=row_labels,
        columns=row_labels
    )

    # col_cophenetic_df 的 index 和 columns 都是 col_labels
    col_cophenetic_df = pd.DataFrame(
        col_cophenetic_square,
        index=col_labels,
        columns=col_labels
    )

    print("Cophenetic distance calculation done.")
    return row_cophenetic_df, col_cophenetic_df
