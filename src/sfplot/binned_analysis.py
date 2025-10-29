import numpy as np
from scipy.spatial.distance import cdist


def calculate_gene_distance_matrix_ewnn(expression: pd.DataFrame, coordinates: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """
    基于表达加权最近邻计算基因间有向距离矩阵 (EWNN 方法)。
    参数:
        expression: 行索引为spot、列索引为gene的 DataFrame，值为表达量。
        coordinates: 行索引为spot、列为 ['x','y'] 的 DataFrame，对应每个spot的空间坐标。
        threshold: 表达阈值，只有表达量大于此值的spot才视为基因有表达 (默认0，即只要>0皆算)。
    返回:
        DataFrame，索引和列均为基因名，元素 D(i->j) 为基因 i 到 基因 j 的加权最近邻距离。
    """
    genes = expression.columns
    dist_matrix = pd.DataFrame(np.nan, index=genes, columns=genes)
    # 预先计算每个基因的有效spot布尔索引，避免重复计算
    gene_spots = {gene: expression[gene].values > threshold for gene in genes}
    coords_arr = coordinates.values  # 转为 numpy 数组加速计算
    for i, gene_i in enumerate(genes):
        mask_i = gene_spots[gene_i]
        if not mask_i.any():
            continue  # 基因i无表达
        coords_i = coords_arr[mask_i]                         # 基因i表达的坐标
        weights_i = expression.loc[mask_i, gene_i].values     # 基因i在这些坐标的表达量
        for gene_j in genes:
            mask_j = gene_spots[gene_j]
            if not mask_j.any():
                dist_matrix.loc[gene_i, gene_j] = np.nan      # 基因j无表达
            else:
                coords_j = coords_arr[mask_j]
                # 计算i表达坐标到j表达坐标的距离矩阵，并取每行最小值 (最近邻)
                dists = cdist(coords_i, coords_j, metric='euclidean')  # SciPy距离计算
                min_dists = dists.min(axis=1)                         # 对每个i点，最近j距离
                # 按基因i表达量加权平均
                avg_dist = np.average(min_dists, weights=weights_i)
                dist_matrix.loc[gene_i, gene_j] = avg_dist
    return dist_matrix


def calculate_gene_distance_matrix_wmda(expression: pd.DataFrame, coordinates: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """
    基于表达分布质心计算基因间有向距离矩阵 (WMDA 方法)。
    参数:
        expression: [n_spots × n_genes] 的表达矩阵 DataFrame。
        coordinates: [n_spots × 2] 的坐标 DataFrame，含 'x','y' 列。
        threshold: 表达阈值，只有表达量大于此值的spot才计入计算。
    返回:
        DataFrame，索引和列为基因名，元素 D(i->j) 为基因 i 到 基因 j 的质心距离 (加权平均)。
    """
    genes = expression.columns
    dist_matrix = pd.DataFrame(np.nan, index=genes, columns=genes)
    coords_arr = coordinates.values
    # 预先计算每个基因的质心坐标
    centers = {}
    for gene in genes:
        mask = expression[gene].values > threshold
        if not mask.any():
            centers[gene] = None
        else:
            sub_coords = coords_arr[mask]
            sub_expr = expression.loc[mask, gene].values
            # 计算加权质心
            cx = np.average(sub_coords[:,0], weights=sub_expr)
            cy = np.average(sub_coords[:,1], weights=sub_expr)
            centers[gene] = (cx, cy)
    # 计算有向距离
    for gene_i in genes:
        mask_i = expression[gene_i].values > threshold
        if not mask_i.any():
            continue
        sub_coords_i = coords_arr[mask_i]
        sub_expr_i = expression.loc[mask_i, gene_i].values
        for gene_j in genes:
            center_j = centers.get(gene_j)
            if center_j is None:
                dist_matrix.loc[gene_i, gene_j] = np.nan
            else:
                # 计算i的每个点到j质心的距离，再按i表达加权
                dists = np.linalg.norm(sub_coords_i - np.array(center_j), axis=1)
                avg_dist = np.average(dists, weights=sub_expr_i)
                dist_matrix.loc[gene_i, gene_j] = avg_dist
    return dist_matrix


import numpy as np
import pandas as pd
from typing import Optional, Tuple
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist, squareform

def compute_cophenetic_distances_from_group_mean_matrix(
    df_group_mean_clean: pd.DataFrame,
    method: str = "average",
    metric: str = "euclidean",
    normalize: bool = True,
    show_corr: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    基于簇×簇的“Searcher→Findee”平均距离矩阵，计算行/列两个方向的 cophenetic 距离矩阵。
    返回值与 `compute_cophenetic_distances_from_df` 一致：(row_coph_df, col_coph_df)，且各自（独立）归一化到 [0,1]（可关闭）。

    Parameters
    ----------
    df_group_mean_clean : pd.DataFrame
        行为 Searcher（源类别）、列为 Findee（目标类别）的平均距离矩阵，要求为数值型。
        建议对缺失值事先清洗；若仍有少量 NaN，本函数会使用列均值填充（列仍全 NaN 则用全局均值）。
    method : str, default "average"
        层次聚类的 linkage 方法（"average"、"single"、"complete"、"ward" 等）。
    metric : str, default "euclidean"
        行/列向量两两之间的距离度量，传给 `scipy.spatial.distance.pdist`。
    normalize : bool, default True
        是否将两个输出矩阵分别线性缩放到 [0,1]。
    show_corr : bool, default False
        是否打印行/列的 cophenetic 相关系数（衡量树形结构保真度）。

    Returns
    -------
    row_coph_df : pd.DataFrame
        行方向的 cophenetic 距离矩阵（索引与列均为 df_group_mean_clean.index）。
        若 normalize=True，则各自（独立）缩放到 [0,1]，对角线重置为 0。
    col_coph_df : pd.DataFrame
        列方向的 cophenetic 距离矩阵（索引与列均为 df_group_mean_clean.columns）。
        若 normalize=True，则各自（独立）缩放到 [0,1]，对角线重置为 0。

    Notes
    -----
    - 当只有 0 或 1 个行（或列）时，无法进行聚类；本函数将返回全 0 的方阵（对角线为 0）。
    - `cophenet` 需要先计算 `pdist(X)`（原始观测之间的两两距离）以及对应 linkage。
    - 与 `compute_cophenetic_distances_from_df` 的逻辑对应：后者先从点级别求“到每个簇的定向最近邻均值”，
      形成类似的簇×簇矩阵，再做层次聚类并取 cophenetic 距离；本函数直接从已给的矩阵出发做后半段。
    """
    if not isinstance(df_group_mean_clean, pd.DataFrame):
        raise TypeError("df_group_mean_clean 必须是 pandas.DataFrame。")

    # 确保全为数值型，拷贝一份避免原地修改
    M = df_group_mean_clean.copy()
    # 尝试将列转为数值；若失败会保留原值，这里统一使用 astype(float) 更严格
    try:
        M = M.astype(float)
    except Exception as e:
        raise ValueError("df_group_mean_clean 必须为数值型矩阵，无法转换为 float。") from e

    # 若仍有 NaN：用列均值填充；若该列全 NaN，则用全局均值；若全局也 NaN，则置 0
    if M.isna().any().any():
        col_means = M.mean(axis=0)
        global_mean = np.nanmean(M.values)
        col_means = col_means.fillna(global_mean if not np.isnan(global_mean) else 0.0)
        M = M.fillna(col_means)

    # ---------- 一个实用的归一化函数 ----------
    def _normalize_to_01(D: pd.DataFrame) -> pd.DataFrame:
        vmin = D.values.min()
        vmax = D.values.max()
        if vmax <= vmin:
            out = pd.DataFrame(np.zeros_like(D.values), index=D.index, columns=D.columns)
        else:
            out = (D - vmin) / (vmax - vmin)
        # 对角线设为 0（数值稳定）
        np.fill_diagonal(out.values, 0.0)
        return out

    # ---------- 行方向（以每一行向量为观测） ----------
    row_labels = M.index.to_list()
    n_row = len(row_labels)
    if n_row >= 2:
        # 行向量：shape = (n_row, n_col)
        X_row = M.values
        Y_row = pdist(X_row, metric=metric)     # condensed 距离向量
        Z_row = linkage(Y_row, method=method)   # 层次聚类
        c_row, coph_row = cophenet(Z_row, Y_row)
        if show_corr:
            print(f"[Row] Cophenetic correlation: {c_row:.4f}")
        row_coph = squareform(coph_row)
        row_coph_df = pd.DataFrame(row_coph, index=row_labels, columns=row_labels)
    else:
        # 退化情形：只有 0 或 1 行，返回全 0 方阵
        row_coph_df = pd.DataFrame(np.zeros((n_row, n_row)), index=row_labels, columns=row_labels)

    # ---------- 列方向（以每一列向量为观测） ----------
    col_labels = M.columns.to_list()
    n_col = len(col_labels)
    if n_col >= 2:
        # 列向量：shape = (n_col, n_row)  —— 使用 M.T 的行作为观测
        X_col = M.values.T
        Y_col = pdist(X_col, metric=metric)
        Z_col = linkage(Y_col, method=method)
        c_col, coph_col = cophenet(Z_col, Y_col)
        if show_corr:
            print(f"[Col] Cophenetic correlation: {c_col:.4f}")
        col_coph = squareform(coph_col)
        col_coph_df = pd.DataFrame(col_coph, index=col_labels, columns=col_labels)
    else:
        col_coph_df = pd.DataFrame(np.zeros((n_col, n_col)), index=col_labels, columns=col_labels)

    # ---------- 可选：分别归一化到 [0,1] ----------
    if normalize:
        row_coph_df = _normalize_to_01(row_coph_df)
        col_coph_df = _normalize_to_01(col_coph_df)

    return row_coph_df, col_coph_df
