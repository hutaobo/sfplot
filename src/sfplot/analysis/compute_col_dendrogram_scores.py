import os
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors


def compute_col_dendrogram_scores(
    data: Union['anndata.AnnData', pd.DataFrame],
    input_type: str = "adata",  # "adata" 或 "dataframe"
    cluster_col: str = "Cluster",
    x_col: str = "x",  # DataFrame中x轴坐标的列名
    y_col: str = "y",  # DataFrame中y轴坐标的列名
    cell_id_col: Optional[str] = None,  # DataFrame中细胞id列名，若为 None，则自动生成
    output_dir: Optional[str] = None,
    method: str = "average"
) -> dict:
    """
    根据输入数据（anndata 对象或 DataFrame），计算用于聚类的均值矩阵，
    并对列方向（cluster）的 linkage 生成 dendrogram（树状结构），
    从上到下为每个分叉赋分（第一层得1分，第二层得0.5分，依次类推）。

    参数说明：
      data: 输入数据，类型为 anndata.AnnData 或 pandas.DataFrame
      input_type: "adata" 或 "dataframe"，用于指定数据类型
      cluster_col: 指定包含 cluster 信息的列名
      x_col, y_col: 如果输入为 DataFrame，则用于指定存储坐标的列名
      cell_id_col: 如果输入为 DataFrame，则指定细胞 id 的列名；
                   若为 None，则自动生成一个 cell id 列，格式为 "cell_0", "cell_1", …
      output_dir: 结果输出目录（用于保存中间结果），默认为当前工作目录
      method: linkage 的方法参数，默认为 "average"

    返回：
      一个字典，描述了 dendrogram 的结构、每个节点的距离、赋分，以及对于叶子节点包含实际的 cluster 名称（"name" 键）。
    """
    # 0. 处理输出目录
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    if input_type == "dataframe":
        # 使用 DataFrame 数据
        df = data.copy()
        if cluster_col not in df.columns:
            raise ValueError(f"'{cluster_col}' 不存在于输入的 DataFrame 中。")
        clusters = df[cluster_col].astype("category")
        unique_clusters = clusters.cat.categories

        # 检查坐标列
        if not set([x_col, y_col]).issubset(df.columns):
            raise ValueError(f"输入的 DataFrame 必须包含坐标列：'{x_col}' 和 '{y_col}'。")
        coords = df[[x_col, y_col]].values

        # 获取细胞 id：如果 cell_id_col 为 None，则自动生成
        if cell_id_col is not None:
            if cell_id_col not in df.columns:
                raise ValueError(f"'{cell_id_col}' 不存在于输入的 DataFrame 中。")
            cell_ids = df[cell_id_col]
        else:
            # 自动生成 cell id 列并添加到 DataFrame
            df["cell_id"] = ["cell_" + str(i) for i in range(len(df))]
            cell_ids = df["cell_id"]

        # 构建一个 DataFrame, 行为 cell_ids, 列为 cluster
        df_nearest_cluster_dist = pd.DataFrame(
            index=cell_ids,
            columns=unique_clusters,
            dtype=float
        )

        # 对每个 cluster，用最近邻模型计算所有细胞到该 cluster 最近中心的距离
        for c in unique_clusters:
            mask_c = (clusters == c)
            coords_c = coords[mask_c.values]
            if coords_c.shape[0] == 0:
                df_nearest_cluster_dist.loc[:, c] = np.nan
                continue
            nbrs_c = NearestNeighbors(n_neighbors=1, algorithm="auto")
            nbrs_c.fit(coords_c)
            dist_c, _ = nbrs_c.kneighbors(coords)
            df_nearest_cluster_dist[c] = dist_c[:, 0]

        # 构建用于分组计算均值的 Series，索引为 cell_ids，值为 cluster 信息
        clusters_by_id = pd.Series(
            data=clusters.values,
            index=cell_ids,
            name=cluster_col
        )
        df_group_mean = df_nearest_cluster_dist.groupby(clusters_by_id).mean()

    elif input_type == "adata":
        # 使用 anndata 对象
        if cluster_col not in data.obs.columns:
            raise ValueError(f"'{cluster_col}' 不存在于 data.obs 中。请检查列名。")
        clusters = data.obs[cluster_col].astype("category")
        unique_clusters = clusters.cat.categories

        if "spatial" not in data.obsm:
            raise ValueError("'spatial' 坐标信息不存在于 data.obsm 中。请确保数据包含空间坐标。")
        coords = data.obsm["spatial"]

        if cell_id_col not in data.obs.columns:
            raise ValueError(f"'{cell_id_col}' 不存在于 data.obs 中。请确保数据包含该列。")
        cell_ids = data.obs[cell_id_col]

        df_nearest_cluster_dist = pd.DataFrame(
            index=cell_ids,
            columns=unique_clusters,
            dtype=float
        )

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

        # 保存结果到 data.uns
        data.uns["nearest_cluster_dist"] = df_nearest_cluster_dist

        clusters_by_id = pd.Series(
            data=clusters.values,
            index=cell_ids,
            name=cluster_col
        )
        df_group_mean = df_nearest_cluster_dist.groupby(clusters_by_id).mean()
    else:
        raise ValueError("input_type 参数必须为 'adata' 或 'dataframe'.")

    # 处理均值矩阵，删除全为 NaN 的列
    df_group_mean_clean = df_group_mean.dropna(axis=1, how="all")
    if df_group_mean_clean.empty:
        raise ValueError("数据处理后为空，请检查输入数据。")

    # 构造叶子名称映射：df_group_mean_clean 的列为实际的 cluster 名称
    leaf_names = {i: name for i, name in enumerate(df_group_mean_clean.columns)}

    # 计算列方向（cluster）的 linkage（对转置后的数据进行聚类）
    col_linkage = linkage(df_group_mean_clean.T, method=method)

    # 将 linkage 转换为树状结构
    root, _ = to_tree(col_linkage, rd=True)

    # 递归函数：为树中每个分叉赋分（从上到下，每下降一层分数减半）
    def assign_score(node, level=1):
        if node.left is not None and node.right is not None:
            score = 1 / (2 ** (level - 1))
            node.left.score = score
            node.right.score = score
            assign_score(node.left, level + 1)
            assign_score(node.right, level + 1)

    assign_score(root, level=1)

    # 递归函数：将树状结构转换为字典表示，对于叶子节点，额外保存实际的 cluster 名称（"name" 键）
    def tree_to_dict(node):
        if node.left is None and node.right is None:
            return {
                "id": node.id,
                "name": leaf_names.get(node.id, None),
                "dist": node.dist,
                "score": getattr(node, "score", None)
            }
        else:
            node_dict = {
                "id": node.id,
                "dist": node.dist,
                "score": getattr(node, "score", None)
            }
            if node.left is not None:
                node_dict["left"] = tree_to_dict(node.left)
            if node.right is not None:
                node_dict["right"] = tree_to_dict(node.right)
            return node_dict

    dendrogram_dict = tree_to_dict(root)
    return dendrogram_dict

# 使用示例：
# 对于 DataFrame 输入且未提供 cell_id_col（自动生成 cell id）：
#
# df = pd.DataFrame({
#     "Location_Center_X": [0.1, 0.3, 0.2, 0.8, 0.85],
#     "Location_Center_Y": [1.2, 1.3, 1.1, 0.5, 0.45],
#     "Gene": ["A", "A", "B", "B", "B"]
# })
# dendro_structure = compute_col_dendrogram_scores(
#     df,
#     input_type="dataframe",
#     cluster_col="Gene",
#     x_col="Location_Center_X",
#     y_col="Location_Center_Y"
# )
# print(dendro_structure)
