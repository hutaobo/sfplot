from typing import Optional, Tuple
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import math
import os
import psutil


def pick_batch_size(
    n_cells: int,
    dims: int = 2,
    frac: float = 0.30,
    hard_min: int = 50_000,
    hard_max: int | None = None,
    bytes_per_row: int | None = None,
    safety_gb: float = 8.0,
    env_override_var: str = "BATCH_SIZE_OVERRIDE",
) -> int:
    """
    Pick a batch size that better utilizes RAM on big machines.

    Key ideas:
    - Allow an env override (for quick experiments).
    - Subtract a fixed safety buffer (safety_gb) from available RAM.
    - Make bytes_per_row configurable; provide a conservative default.
    - Optional hard_max; if None, we don't clamp by a hard cap.

    Parameters
    ----------
    n_cells : int
        Total number of items to process.
    dims : int
        Dimensionality; may influence copies inside algorithms.
    frac : float
        Fraction of *available* RAM to budget.
    hard_min : int
        Lower bound for stability on small RAM.
    hard_max : Optional[int]
        Upper bound; set None to disable hard clamping.
    bytes_per_row : Optional[int]
        Estimated peak bytes per row for the step. If None, pick a conservative default.
    safety_gb : float
        Keep this amount of RAM free regardless (OS/page cache/etc.).
    env_override_var : str
        If set, this env var forces the batch size (int), bypassing heuristics.

    Returns
    -------
    int
        A batch size in [hard_min, n_cells] (and <= hard_max if provided).
    """
    # 0) environment override for quick control
    ov = os.environ.get(env_override_var)
    if ov:
        try:
            forced = int(ov)
            return max(hard_min, min(n_cells, forced))
        except ValueError:
            pass  # ignore bad value

    # 1) available RAM minus a safety buffer
    avail = psutil.virtual_memory().available
    safety = int(max(0, safety_gb) * (1024**3))
    usable = max(0, avail - safety)

    # 2) fraction of usable
    budget = int(usable * max(0.05, min(frac, 0.95)))  # clamp frac to [5%,95%]

    # 3) estimate bytes/row
    if bytes_per_row is None:
        # Slightly more generous than 64 to cover index arrays/temporary buffers
        bytes_per_row = 64 if dims <= 2 else 80

    # 4) target rows from memory budget
    target = 1 if bytes_per_row <= 0 else budget // bytes_per_row

    # 5) apply clamps
    bsz = max(hard_min, int(target))
    if hard_max is not None:
        bsz = min(bsz, hard_max)

    return min(bsz, n_cells)


def compute_cophenetic_distances_from_df_memory_opt(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    z_col: Optional[str] = None,
    celltype_col: str = "celltype",
    method: str = "average",
    show_corr: bool = False,
    batch_size: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    功能与原始 compute_cophenetic_distances_from_df 相同，但通过分批计算降低内存占用。
    """
    # 1. 必要列检查
    required_cols = {x_col, y_col, celltype_col}
    if z_col:
        required_cols.add(z_col)
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame 缺少必要列：{required_cols}")

    # 2. 提取簇信息和坐标
    clusters = df[celltype_col].astype("category")
    unique_clusters = clusters.cat.categories
    coord_cols = [x_col, y_col] + ([z_col] if z_col else [])
    coords = df[coord_cols].values  # shape: (n_cells, dims)
    n_cells = coords.shape[0]
    n_clusters = len(unique_clusters)

    # 3. 初始化簇间平均距离矩阵 (cluster × cluster)
    #    用 DataFrame 方便按照簇标签对齐填值，初始全 NaN
    df_group_mean = pd.DataFrame(
        np.nan, index=unique_clusters, columns=unique_clusters, dtype=float
    )

    # 4. 对每个簇计算所有细胞到该簇的最近邻距离，并计算均值
    #    逐列计算，避免构建完整矩阵
    # 可选批量大小：如果 batch_size 提供，则分批查询以节省内存
    # 预先计算每个细胞的簇代码，以便快速分组
    cluster_codes = clusters.cat.codes.values  # each cell's cluster code (int)
    for c_label in unique_clusters:
        mask_c = (clusters == c_label)
        coords_c = coords[mask_c]  # 该簇的所有坐标
        if coords_c.shape[0] == 0:
            # 若该簇没有细胞，则整列保持 NaN
            continue
        # 建立该簇的1-NN模型
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(coords_c)

        # 查询所有细胞到当前簇的距离，支持按批次查询
        if batch_size is None or batch_size >= n_cells:
            # 一次性查询所有细胞
            dist_all, _ = nbrs.kneighbors(coords)
            dist_all = dist_all[:, 0]  # 提取距离列
            # 依据预存的簇代码直接计算每个源簇的平均距离
            sums = np.bincount(cluster_codes, weights=dist_all, minlength=n_clusters)
            counts = np.bincount(cluster_codes, minlength=n_clusters)
        else:
            # 分批查询
            sums = np.zeros(n_clusters, dtype=float)
            counts = np.zeros(n_clusters, dtype=int)
            for start in range(0, n_cells, batch_size):
                end = min(n_cells, start + batch_size)
                dist_batch, _ = nbrs.kneighbors(coords[start:end])
                dist_batch = dist_batch[:, 0]
                code_batch = cluster_codes[start:end]
                # 累加该批次的距离和计数
                sums += np.bincount(code_batch, weights=dist_batch, minlength=n_clusters)
                counts += np.bincount(code_batch, minlength=n_clusters)
        # 计算平均值（对有数据的簇）
        means = np.divide(sums, counts, out=np.full_like(sums, np.nan, dtype=float), where=(counts>0))
        # 将结果填入输出矩阵的对应列
        df_group_mean.loc[:, c_label] = means

    # 5. 清理全 NaN 列（如果有簇在任何其他簇中均无邻居，会出现该列全 NaN）
    df_group_mean_clean = df_group_mean.dropna(axis=1, how="all")
    if df_group_mean_clean.empty:
        raise ValueError("df_group_mean_clean 为空，请检查数据。")

    # 6. 对行和列分别进行层次聚类
    row_linkage = linkage(df_group_mean_clean.values, method=method)
    col_linkage = linkage(df_group_mean_clean.T.values, method=method)

    # 7. 计算 cophenetic 距离矩阵（condensed form），以及相关系数（可选打印）
    row_coph_corr, row_coph_condensed = cophenet(row_linkage, pdist(df_group_mean_clean.values))
    col_coph_corr, col_coph_condensed = cophenet(col_linkage, pdist(df_group_mean_clean.T.values))
    if show_corr:
        print(f"Row cophenetic correlation: {row_coph_corr:.4f}")
        print(f"Col cophenetic correlation: {col_coph_corr:.4f}")

    # 8. 将 condensed 距离转为方阵并构建 DataFrame
    row_cophenetic_df = pd.DataFrame(squareform(row_coph_condensed),
                                     index=df_group_mean_clean.index,
                                     columns=df_group_mean_clean.index)
    col_cophenetic_df = pd.DataFrame(squareform(col_coph_condensed),
                                     index=df_group_mean_clean.columns,
                                     columns=df_group_mean_clean.columns)

    # 9. 分别归一化行、列cophenetic距离矩阵到 [0, 1]
    def normalize_df(mat: pd.DataFrame) -> pd.DataFrame:
        dmin, dmax = mat.values.min(), mat.values.max()
        return mat if dmin == dmax else (mat - dmin) / (dmax - dmin)

    row_cophenetic_df_norm = normalize_df(row_cophenetic_df)
    col_cophenetic_df_norm = normalize_df(col_cophenetic_df)
    return row_cophenetic_df_norm, col_cophenetic_df_norm
