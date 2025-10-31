import numpy as np
import pandas as pd
from typing import Optional
from typing import Optional, Tuple
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.neighbors import NearestNeighbors


def calculate_gene_distance_matrix_ewnn(expression: pd.DataFrame,
                                       coordinates: pd.DataFrame,
                                       threshold: float = 0.0,
                                       z: Optional[pd.Series] = None,
                                       memory_limit_gb: float = 300.0,
                                       batch_size: Optional[int] = None) -> pd.DataFrame:
    """
    基于表达加权最近邻计算基因间有向距离矩阵 (EWNN 方法)的内存优化版本。
    通过分块计算降低内存占用，避免在大规模矩阵计算时出现 MemoryError。

    新增参数:
        memory_limit_gb: float, 默认 300.0
            内存使用上限 (GB)。当预计最终距离矩阵所需内存超过该值时，
            将采用磁盘内存映射文件暂存结果，避免占用过多内存。
        batch_size: Optional[int], 默认 None
            最近邻查询的批大小。如为 None，则根据可用内存自动选择批大小；
            如指定具体数值，则按该批大小分批查询邻近距离。
    其余参数与返回值同原函数。
    """
    genes = list(expression.columns)
    n_genes = len(genes)
    # 根据基因数估算最终距离矩阵大小，决定是否使用磁盘临时存储
    total_bytes = n_genes * n_genes * 8  # float64 每个元素8字节
    memory_limit_bytes = memory_limit_gb * (1024 ** 3)
    use_memmap = total_bytes > memory_limit_bytes

    if use_memmap:
        # 创建临时内存映射文件用于存储结果矩阵
        import tempfile
        tmp = tempfile.NamedTemporaryFile(prefix="gene_dist_ewnn_", suffix=".dat", delete=False)
        tmp_filename = tmp.name
        tmp.close()
        dist_matrix_arr = np.memmap(tmp_filename, dtype='float64', mode='w+', shape=(n_genes, n_genes))
        dist_matrix_arr[:] = np.nan  # 初始填充 NaN
    else:
        # 在内存中初始化结果矩阵
        dist_matrix_arr = np.full((n_genes, n_genes), np.nan, dtype=np.float64)

    # 预先计算每个基因的有效表达掩码，提升重复使用效率
    gene_masks = {gene: (expression[gene].values > threshold) for gene in genes}
    # 可选：预存每个基因的表达值数组以加速索引
    gene_expr_values = {gene: expression[gene].values for gene in genes}

    # 构造坐标矩阵 (如果提供了 z 坐标则一并包含)
    coords = coordinates.values
    if z is not None:
        coords = np.hstack([coords, np.asarray(z).reshape(-1, 1)])
    n_spots = coords.shape[0]

    # 确定批大小：如未指定则自动估算
    if batch_size is None:
        try:
            # 尝试利用已有工具按内存情况选取批大小
            from sfplot.compute_cophenetic_distances_from_df_memory_opt import pick_batch_size
            batch = pick_batch_size(n_cells=n_spots, dims=coords.shape[1])
        except ImportError:
            batch = n_spots  # 无法导入则不分批
    else:
        batch = min(batch_size, n_spots)

    # 逐个目标基因计算距离列（“Findee”方向），按需分批查询最近邻
    for j_idx, gene_j in enumerate(genes):
        mask_j = gene_masks[gene_j]
        if not mask_j.any():
            continue  # 基因 j 无表达，整列保持 NaN

        # 构建基因 j 表达坐标的 1-近邻搜索模型
        coords_j = coords[mask_j]
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(coords_j)

        # 计算所有样本点到当前基因 j 表达点集的最近邻距离，可分批执行
        if batch >= n_spots:
            # 单批次直接查询所有点
            dist_all, _ = nbrs.kneighbors(coords)
            dist_all = dist_all[:, 0]  # 提取最近邻距离
        else:
            # 分批查询以降低内存峰值
            dist_all = np.empty(n_spots, dtype=np.float64)
            for start in range(0, n_spots, batch):
                end = min(n_spots, start + batch)
                dist_batch, _ = nbrs.kneighbors(coords[start:end])
                dist_all[start:end] = dist_batch[:, 0]

        # 利用预计算的距离数组和表达权重，计算各源基因到基因 j 的加权平均最近邻距离
        for i_idx, gene_i in enumerate(genes):
            mask_i = gene_masks[gene_i]
            if not mask_i.any():
                continue  # 基因 i 无表达，整行保持 NaN
            # 提取基因 i 在表达位置的距离和权重
            dists_i = dist_all[mask_i]
            weights_i = gene_expr_values[gene_i][mask_i]
            # 计算加权平均距离
            dist_matrix_arr[i_idx, j_idx] = np.average(dists_i, weights=weights_i)

    # 将结果转换为 DataFrame 返回，索引/列与输入基因顺序一致
    dist_matrix_df = pd.DataFrame(dist_matrix_arr, index=genes, columns=genes)
    return dist_matrix_df


def calculate_gene_distance_matrix_wmda(expression: pd.DataFrame,
                                       coordinates: pd.DataFrame,
                                       threshold: float = 0.0,
                                       z: Optional[pd.Series] = None,
                                       memory_limit_gb: float = 300.0) -> pd.DataFrame:
    """
    基于表达分布质心计算基因间有向距离矩阵 (WMDA 方法)的内存优化版本。
    通过分块计算和必要的临时存储降低内存占用。

    新增参数:
        memory_limit_gb: float, 默认 300.0
            内存使用上限 (GB)。当预计输出矩阵过大时，将使用磁盘内存映射文件暂存结果，避免内存不足。
    其余参数与返回值同原函数。
    """
    genes = list(expression.columns)
    n_genes = len(genes)
    total_bytes = n_genes * n_genes * 8
    memory_limit_bytes = memory_limit_gb * (1024 ** 3)
    use_memmap = total_bytes > memory_limit_bytes

    if use_memmap:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(prefix="gene_dist_wmda_", suffix=".dat", delete=False)
        tmp_filename = tmp.name
        tmp.close()
        dist_matrix_arr = np.memmap(tmp_filename, dtype='float64', mode='w+', shape=(n_genes, n_genes))
        dist_matrix_arr[:] = np.nan
    else:
        dist_matrix_arr = np.full((n_genes, n_genes), np.nan, dtype=np.float64)

    coords = coordinates.values
    if z is not None:
        coords = np.hstack([coords, np.asarray(z).reshape(-1, 1)])

    # 预计算每个基因的加权质心坐标
    centers = {}
    for gene in genes:
        mask = expression[gene].values > threshold
        if not mask.any():
            centers[gene] = None
        else:
            sub_coords = coords[mask]
            sub_expr = expression.loc[mask, gene].values
            # 计算 (x, y, [z]) 三个方向的加权平均，得到该基因的空间质心
            cx = np.average(sub_coords[:, 0], weights=sub_expr)
            cy = np.average(sub_coords[:, 1], weights=sub_expr)
            if z is not None:
                cz = np.average(sub_coords[:, 2], weights=sub_expr)
                centers[gene] = np.array([cx, cy, cz])
            else:
                centers[gene] = np.array([cx, cy])

    # 逐个目标基因质心计算距离列
    n_spots = coords.shape[0]
    for j_idx, gene_j in enumerate(genes):
        center_j = centers.get(gene_j)
        if center_j is None:
            continue  # 基因 j 无表达，整列 NaN
        # 计算所有 spot 到质心 j 的距离
        diff = coords - center_j  # shape: (n_spots, 2或3)
        dist_to_center_j = np.linalg.norm(diff, axis=1)  # 每个spot到质心的欧氏距离

        # 计算每个源基因 i 到该质心的加权平均距离
        for i_idx, gene_i in enumerate(genes):
            mask_i = expression[gene_i].values > threshold
            if not mask_i.any():
                continue  # 基因 i 无表达，整行 NaN
            weights_i = expression.loc[mask_i, gene_i].values
            dist_matrix_arr[i_idx, j_idx] = np.average(dist_to_center_j[mask_i], weights=weights_i)

    dist_matrix_df = pd.DataFrame(dist_matrix_arr, index=genes, columns=genes)
    return dist_matrix_df


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
