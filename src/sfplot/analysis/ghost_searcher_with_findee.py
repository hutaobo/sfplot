import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors


def compute_groupwise_average_distance_between_two_dfs(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df1_x_col: str = 'x',
    df1_y_col: str = 'y',
    df1_celltype_col: str = 'celltype',
    df2_x_col: str = 'x',
    df2_y_col: str = 'y',
    df2_celltype_col: str = 'celltype',
    n_jobs: int = -1  # 并行计算使用的作业数，-1 表示使用所有 CPU
) -> pd.DataFrame:
    """
    计算并返回一个矩阵，该矩阵的行表示 df1 中每个 unique 的源细胞类型，
    列表示 df2 中每个 unique 的目标细胞类型，
    每个元素为：df1 中对应细胞组内所有细胞到 df2 中对应细胞组内最近邻细胞距离的平均值。

    参数:
    --------
    df1 : pd.DataFrame
        包含源细胞数据的 DataFrame。
    df2 : pd.DataFrame
        包含目标细胞数据的 DataFrame。
    df1_x_col : str, optional
        df1 中表示 x 坐标的列名。默认为 'x'。
    df1_y_col : str, optional
        df1 中表示 y 坐标的列名。默认为 'y'。
    df1_celltype_col : str, optional
        df1 中表示细胞类型的列名。默认为 'celltype'。
    df2_x_col : str, optional
        df2 中表示 x 坐标的列名。默认为 'x'。
    df2_y_col : str, optional
        df2 中表示 y 坐标的列名。默认为 'y'。
    df2_celltype_col : str, optional
        df2 中表示细胞类型的列名。默认为 'celltype'。
    n_jobs : int, optional
        并行计算的作业数，默认为 -1（使用所有 CPU）。

    返回值:
    --------
    pd.DataFrame
        一个矩阵，行索引为 df1 的 unique 源细胞类型，
        列索引为 df2 的 unique 目标细胞类型，每个元素为对应组的平均最近邻距离。
    """
    # 检查 df1 和 df2 是否包含必需的列
    required_df1_columns = {df1_x_col, df1_y_col, df1_celltype_col}
    required_df2_columns = {df2_x_col, df2_y_col, df2_celltype_col}
    if not required_df1_columns.issubset(df1.columns):
        raise ValueError(f"df1 必须包含以下列：{required_df1_columns}")
    if not required_df2_columns.issubset(df2.columns):
        raise ValueError(f"df2 必须包含以下列：{required_df2_columns}")

    # 提取源和目标细胞类型
    source_cell_types = df1[df1_celltype_col].unique()
    target_cell_types = df2[df2_celltype_col].unique()

    # 初始化结果矩阵
    average_distance_matrix = pd.DataFrame(index=source_cell_types, columns=target_cell_types, dtype=float)

    # 定义一个内部函数，用于并行计算单个源细胞类型对应的行
    def compute_for_source(src_type):
        df1_subset = df1[df1[df1_celltype_col] == src_type]
        source_coords = df1_subset[[df1_x_col, df1_y_col]].values
        row_result = {}

        if source_coords.shape[0] == 0:
            for tgt_type in target_cell_types:
                row_result[tgt_type] = np.nan
            return src_type, row_result

        for tgt_type in target_cell_types:
            df2_subset = df2[df2[df2_celltype_col] == tgt_type]
            target_coords = df2_subset[[df2_x_col, df2_y_col]].values
            if target_coords.shape[0] == 0:
                row_result[tgt_type] = np.nan
                continue
            # 构建最近邻模型，计算 source_coords 中各点到 target_coords 中最近点的距离
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto')
            nbrs.fit(target_coords)
            distances, _ = nbrs.kneighbors(source_coords)
            average_distance = distances[:, 0].mean()
            row_result[tgt_type] = average_distance

        return src_type, row_result

    # 并行计算所有源细胞类型对应的行
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(compute_for_source)(src_type) for src_type in source_cell_types)

    # 将计算结果写入结果矩阵
    for src_type, row_result in results:
        for tgt_type, avg_dist in row_result.items():
            average_distance_matrix.loc[src_type, tgt_type] = avg_dist

    return average_distance_matrix
