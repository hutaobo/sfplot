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
    基于表达双重权重最小成本模型计算基因间有向距离矩阵的内存优化实现。
    在该模型中，对每对基因 (i, j)，在基因 i 的表达点集 S_i 与基因 j 的表达点集 S_j
    之间计算加权成本 C(s,t) 并取最小值作为 D_{ij}。计算公式为：
        C(s,t) = d(s,t) / (E_i(s) * E_j(t))^α + ε，
    其中 d(s,t) 为点 s 和点 t 间的欧氏距离，E_i(s)、E_j(t) 分别为基因 i 和 j
    在这些点的表达值。α 默认为 0.5（根据表达值进行软权重），ε 为避免除零的极小常数 (如 1e-8)。
    然后 D_{ij} = min_{s∈S_i, t∈S_j} C(s,t)。该距离矩阵非对称。

    内存优化：仅考虑高于阈值 threshold 的表达点。对于没有任意表达点的基因，其对应行或列保持 NaN。
    函数会根据基因总数估计输出矩阵大小，超过 memory_limit_gb 时使用内存映射临时文件存储结果。
    参数 batch_size 可用于指定按批处理目标点的数量，以降低单次计算的内存峰值；如未指定则自动判断。

    参数:
        expression: pd.DataFrame
            行索引为空间点（spot）的标识，列为基因名称，值为表达量。
        coordinates: pd.DataFrame
            与 expression 对应的坐标 (x, y) 列表（如为空间位置坐标），行顺序需与 expression 对齐。
        threshold: float, 默认 0.0
            表达阈值。仅当表达量大于该阈值时，空间点才计入计算。
        z: Optional[pd.Series], 默认 None
            可选的 z 坐标列（如 3D 空间坐标）。提供时将坐标维度扩展为 (x, y, z)。
        memory_limit_gb: float, 默认 300.0
            输出距离矩阵允许占用的最大内存（GB）。若预计矩阵过大将使用磁盘临时文件存储结果。
        batch_size: Optional[int], 默认 None
            每批处理的目标基因表达点数量。如未指定则根据点对数量自动选择分批策略；若指定则按该大小对目标点集分块计算。

    返回:
        pd.DataFrame: 行索引和列均为基因名称的距离矩阵（float64），表示基因间的定向距离。非可计算位置为 NaN。
    """
    import numpy as np
    import pandas as pd
    from scipy.spatial.distance import cdist

    genes = list(expression.columns)
    n_genes = len(genes)
    # 根据基因数估算最终距离矩阵大小，决定是否使用磁盘临时存储
    total_bytes = n_genes * n_genes * 8  # float64 每个元素8字节
    memory_limit_bytes = memory_limit_gb * (1024 ** 3)
    use_memmap = total_bytes > memory_limit_bytes

    if use_memmap:
        # 创建临时内存映射文件用于存储结果矩阵，以避免占用过多内存
        import tempfile
        tmp = tempfile.NamedTemporaryFile(prefix="gene_dist_ewnn_", suffix=".dat", delete=False)
        tmp_filename = tmp.name
        tmp.close()
        dist_matrix_arr = np.memmap(tmp_filename, dtype='float64', mode='w+', shape=(n_genes, n_genes))
        dist_matrix_arr[:] = np.nan  # 初始填充 NaN
    else:
        # 在内存中初始化结果矩阵
        dist_matrix_arr = np.full((n_genes, n_genes), np.nan, dtype=np.float64)

    # 预先计算每个基因的有效表达掩码和表达值数组，提升重复使用效率
    gene_masks = {gene: (expression[gene].values > threshold) for gene in genes}
    gene_expr_values = {gene: expression[gene].values for gene in genes}

    # 构造坐标矩阵 (如果提供了 z 坐标则一并包含)
    coords = coordinates.values
    if z is not None:
        coords = np.hstack([coords, np.asarray(z).reshape(-1, 1)])
    n_spots = coords.shape[0]

    # 设置计算参数
    alpha = 0.5
    epsilon = 1e-8
    max_pairs = 1e7  # 单次计算的最大点对数，以控制内存使用

    # 计算距离矩阵：对每个源基因 i（行）计算到每个目标基因 j（列）的距离
    for i_idx, gene_i in enumerate(genes):
        mask_i = gene_masks[gene_i]
        if not mask_i.any():
            continue  # 基因 i 无表达，整行保持 NaN
        # 提取基因 i 的表达坐标和表达值
        coords_i = coords[mask_i]
        expr_i = gene_expr_values[gene_i][mask_i]
        # 计算基因 i 表达值的α次幂，用于权重
        E_i_alpha = np.power(expr_i, alpha)
        num_i = coords_i.shape[0]

        for j_idx, gene_j in enumerate(genes):
            mask_j = gene_masks[gene_j]
            if not mask_j.any():
                continue  # 基因 j 无表达，整列保持 NaN
            coords_j = coords[mask_j]
            expr_j = gene_expr_values[gene_j][mask_j]
            num_j = coords_j.shape[0]

            # 确定是否需要分批计算以降低内存峰值
            if batch_size is not None:
                chunk_size = min(batch_size, num_j)
            elif num_i * num_j > max_pairs:
                chunk_size = int(max_pairs // num_i) or 1
            else:
                chunk_size = None

            if chunk_size and chunk_size < num_j:
                # 分块计算目标基因 j 的表达点，累积最小成本
                min_cost_val = np.inf
                for start in range(0, num_j, chunk_size):
                    end = min(num_j, start + chunk_size)
                    sub_coords_j = coords_j[start:end]
                    sub_expr_j = expr_j[start:end]
                    # 计算基因 i 所有表达点到本批次基因 j 表达点的距离矩阵
                    sub_dist = cdist(coords_i, sub_coords_j)  # 形状: (num_i, end-start)
                    # 按表达值进行双重权重
                    sub_dist /= E_i_alpha[:, None]  # 每行除以对应基因 i 表达值^α
                    sub_E_j_alpha = np.power(sub_expr_j, alpha)
                    sub_dist /= sub_E_j_alpha[None, :]  # 每列除以对应基因 j 表达值^α
                    sub_dist += epsilon  # 加上微小常数稳定计算
                    # 更新当前最小成本
                    sub_min = np.min(sub_dist)
                    if sub_min < min_cost_val:
                        min_cost_val = sub_min
                dist_matrix_arr[i_idx, j_idx] = min_cost_val
            else:
                # 一次性计算所有点对成本
                dist_matrix = cdist(coords_i, coords_j)  # 形状: (num_i, num_j)
                dist_matrix /= E_i_alpha[:, None]  # 按基因 i 表达值加权
                E_j_alpha = np.power(expr_j, alpha)
                dist_matrix /= E_j_alpha[None, :]  # 按基因 j 表达值加权
                dist_matrix += epsilon  # 加上微小常数以避免零距离
                # 取最小成本作为 D_{ij}
                dist_matrix_arr[i_idx, j_idx] = np.min(dist_matrix)

    # 将结果转换为 DataFrame，索引和列与输入基因顺序一致
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


import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform


def _get_gene_spots_and_weights(expression_df, genes, threshold=0.0):
    """
    对每个基因：
      - 找出表达 > threshold 的 spot 行号
      - 对应的表达值作为权重
    支持 dense 和 pandas sparse。
    """
    gene_spots = {}
    gene_weights = {}

    for g in genes:
        col = expression_df[g]

        if pd.api.types.is_sparse(col.dtype):
            coo = col.sparse.to_coo()
            mask = coo.data > threshold
            spots_idx = coo.row[mask]
            weights = coo.data[mask]
        else:
            values = col.values
            mask = values > threshold
            spots_idx = np.where(mask)[0]
            weights = values[mask]

        gene_spots[g] = spots_idx.astype(int)
        gene_weights[g] = weights.astype(float)

    return gene_spots, gene_weights


def _weighted_quantile(values, weights, q):
    """
    计算一维数组的（可选权重）分位数 q∈(0,1)。
    """
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return np.nan

    if weights is None:
        return float(np.quantile(values, q))

    weights = np.asarray(weights, dtype=float)
    if np.all(weights <= 0):
        return float(np.quantile(values, q))

    order = np.argsort(values)
    v = values[order]
    w = weights[order]

    cum_w = np.cumsum(w)
    if cum_w[-1] == 0:
        return float(np.quantile(values, q))

    cum_w /= cum_w[-1]
    idx = np.searchsorted(cum_w, q)
    idx = min(idx, v.size - 1)
    return float(v[idx])


def _aggregate_distances(dists, weights, agg="quantile", q=0.9, weight_by_expression=True):
    """
    把一串最近邻距离聚合成一个标量。
    """
    dists = np.asarray(dists, dtype=float)
    if dists.size == 0:
        return np.nan

    if not weight_by_expression:
        weights = None

    if agg == "mean":
        if weights is None:
            return float(dists.mean())
        else:
            return float(np.average(dists, weights=weights))
    elif agg == "median":
        return _weighted_quantile(dists, weights, 0.5)
    elif agg == "quantile":
        return _weighted_quantile(dists, weights, q)
    else:
        raise ValueError(f"Unknown agg: {agg}")


def _estimate_spot_nn_scale(coord_array):
    """
    用全局 KDTree 估计“典型最近邻距离”,
    类似 Visium 的格子边长，用来归一化。
    """
    tree = cKDTree(coord_array)
    # k=2: 第一个是自己，第二个是最近邻
    dists, _ = tree.query(coord_array, k=2)
    nn_dists = dists[:, 1]
    nn_dists = nn_dists[np.isfinite(nn_dists)]
    if nn_dists.size == 0:
        return 1.0
    scale = np.median(nn_dists)
    return float(scale if scale > 0 else 1.0)


def calculate_gene_distance_matrix_visium(
    expression_df: pd.DataFrame,
    coords_df: pd.DataFrame,
    genes=None,
    threshold: float = 0.0,
    min_spots: int = 20,
    min_total_expr: float = 0.0,
    agg: str = "quantile",     # "mean", "median", "quantile"
    quantile: float = 0.9,     # agg="quantile" 时使用
    weight_by_expression: bool = True,
    symmetric: str = "min"     # "none", "min", "mean"
) -> pd.DataFrame:
    """
    适合 Visium 的基因–基因空间距离矩阵（COSTE 风格）.

    - 使用表达加权最近邻（directed）
    - 用稳健分位数聚合（默认 0.9 quantile）
    - 对距离按“spot 最近邻尺度”归一化
    """
    if genes is None:
        genes = list(expression_df.columns)
    else:
        genes = [g for g in genes if g in expression_df.columns]

    # 确保 coords 顺序和 expression 行一致
    coords = coords_df.loc[expression_df.index]
    coord_array = coords.values  # (n_spots, 2)

    # 估计 Visium 网格的典型最近邻尺度，用来归一化
    nn_scale = _estimate_spot_nn_scale(coord_array)

    # 预计算基因的表达 spot 和权重
    gene_spots, gene_weights = _get_gene_spots_and_weights(
        expression_df, genes, threshold=threshold
    )

    # 初始化结果矩阵
    dist_matrix = pd.DataFrame(
        np.nan, index=genes, columns=genes, dtype=float
    )

    # 为每个 target gene 建 KDTree
    gene_trees = {}
    for g in genes:
        spots_j = gene_spots[g]
        if spots_j.size == 0:
            gene_trees[g] = None
        else:
            target_coords = coord_array[spots_j]
            gene_trees[g] = cKDTree(target_coords)

    # 主循环
    for gi in genes:
        spots_i = gene_spots[gi]
        weights_i = gene_weights[gi]

        if spots_i.size < min_spots or weights_i.sum() <= min_total_expr:
            # 太稀疏的基因：整行 NaN
            dist_matrix.loc[gi, :] = np.nan
            continue

        source_coords = coord_array[spots_i]

        for gj in genes:
            if gi == gj:
                dist_matrix.loc[gi, gj] = 0.0
                continue

            spots_j = gene_spots[gj]
            tree_j = gene_trees[gj]

            # 目标基因太稀疏或完全没表达：认为基本不可达
            if spots_j.size < min_spots or tree_j is None:
                dist_matrix.loc[gi, gj] = np.inf
                continue

            # 最近邻查询：gi 的每个点到 gj 的最近点
            dists, _ = tree_j.query(source_coords, k=1)

            directed_raw = _aggregate_distances(
                dists=dists,
                weights=weights_i,
                agg=agg,
                q=quantile,
                weight_by_expression=weight_by_expression
            )

            # 用全局最近邻尺度归一化，避免不同数据集/分辨率不可比
            directed_norm = directed_raw / nn_scale
            dist_matrix.loc[gi, gj] = directed_norm

    # 可选对称化
    if symmetric in ("min", "mean"):
        arr = dist_matrix.values
        n = arr.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                dij = arr[i, j]
                dji = arr[j, i]
                if symmetric == "min":
                    val = np.nanmin([dij, dji])
                else:  # "mean"
                    val = np.nanmean([dij, dji])
                arr[i, j] = arr[j, i] = val
        np.fill_diagonal(arr, 0.0)
        dist_matrix.iloc[:, :] = arr

    return dist_matrix
