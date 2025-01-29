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
