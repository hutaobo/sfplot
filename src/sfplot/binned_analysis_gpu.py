import torch


def calculate_gene_distance_matrix_wmda_gpu(
    expression: pd.DataFrame,
    coordinates: pd.DataFrame,
    threshold: float = 0.0,
    z: Optional[pd.Series] = None,
    device: str = "cuda"
) -> pd.DataFrame:
    """
    基于表达分布质心 (WMDA) 的 GPU 实现，计算基因间有向距离矩阵。

    参数：
        expression (pd.DataFrame): 空间表达矩阵，行为位置，列为基因。
        coordinates (pd.DataFrame): 空间坐标 DataFrame (含 x, y 列，若有 z 维则使用 z 参数)。
        threshold (float): 表达阈值，低于等于该值视为无表达。默认0.0。
        z (pd.Series/np.ndarray, 可选): 每个位置的 z 坐标。
        device (str): 使用的设备 ("cuda", "cpu" 等)，默认 "cuda"。

    返回：
        pd.DataFrame: 基因×基因的有向平均质心距离矩阵(DataFrame)，无表达情况以 NaN 表示。
    """
    genes = list(expression.columns)
    n_genes = len(genes)
    # 构建坐标和质心张量
    if z is not None:
        coords_arr = np.hstack([coordinates.values, np.asarray(z).reshape(-1, 1)])
    else:
        coords_arr = coordinates.values
    coords_tensor = torch.from_numpy(coords_arr).to(device=device, dtype=torch.float32)
    # 先计算所有基因的质心坐标（在CPU上完成也可，因为开销小）
    centers = {}
    for gene in genes:
        mask = expression[gene].values > threshold
        if not mask.any():
            centers[gene] = None
        else:
            sub_coords = coords_tensor[mask]  # 直接在GPU上筛选坐标
            sub_expr = torch.from_numpy(expression.loc[mask, gene].values).to(device=device, dtype=torch.float32)
            # 计算加权质心 (注意：为了数值稳定，可能需要将权重归一或转换类型)
            total_w = sub_expr.sum()
            if float(total_w) == 0.0:
                centers[gene] = None
            else:
                center = (sub_coords * sub_expr[:, None]).sum(dim=0) / total_w
                centers[gene] = center  # Tensor 存储质心坐标
    # 初始化结果矩阵
    result = np.full((n_genes, n_genes), np.nan, dtype=float)
    # 计算距离矩阵
    for i_idx, gene_i in enumerate(genes):
        mask_i = expression[gene_i].values > threshold
        if not mask_i.any():
            continue  # 基因 i 无表达
        sub_coords_i = coords_tensor[mask_i]  # gene_i 表达坐标 (GPU tensor)
        sub_expr_i = torch.from_numpy(expression.loc[mask_i, gene_i].values).to(device=device, dtype=torch.float32)
        for j_idx, gene_j in enumerate(genes):
            center_j = centers.get(gene_j)
            if center_j is None:
                result[i_idx, j_idx] = np.nan
            else:
                # 计算 gene_i 每个点到 gene_j 质心的距离张量
                # center_j 是 GPU tensor，sub_coords_i 是 (Ni, dims) tensor
                dists = torch.norm(sub_coords_i - center_j, dim=1)  # 每个点到质心的距离 (Ni,)
                # 按 gene_i 表达权重加权平均距离
                avg_dist = (dists * sub_expr_i).sum() / sub_expr_i.sum()
                result[i_idx, j_idx] = avg_dist.item()
    dist_df = pd.DataFrame(result, index=genes, columns=genes)
    return dist_df


def calculate_gene_distance_matrix_ewnn_gpu(
    expression: pd.DataFrame,
    coordinates: pd.DataFrame,
    threshold: float = 0.0,
    z: Optional[pd.Series] = None,
    max_memory_gb: float = 300.0,
    device: str = "cuda"
) -> pd.DataFrame:
    """
    基于表达加权最近邻 (EWNN) 的 GPU 实现，计算基因间有向距离矩阵。

    参数：
        expression (pd.DataFrame): 空间表达矩阵，行为位置，列为基因。
        coordinates (pd.DataFrame): 空间坐标 DataFrame，列为 x, y （可选 z）。
        threshold (float): 表达阈值，小于等于该值视为无表达。默认0.0。
        z (pd.Series 或 np.ndarray, 可选): 每个位置的 z 坐标。
        max_memory_gb (float): GPU内存上限 (GB)。用于确定批处理大小，默认300.0。
        device (str): 在哪个设备上执行计算，例如 "cuda" 或 "cuda:0"。默认 "cuda"。

    返回：
        pd.DataFrame: 基因×基因的有向平均最近邻距离矩阵(DataFrame格式)，含 NaN 标记无效距离。
    """
    genes = list(expression.columns)
    n_genes = len(genes)
    # 构建坐标张量
    if z is not None:
        coords_arr = np.hstack([coordinates.values, np.asarray(z).reshape(-1, 1)])
    else:
        coords_arr = coordinates.values
    coords_tensor = torch.from_numpy(coords_arr).to(device=device, dtype=torch.float32)

    # 预先缓存每个基因的表达位置索引（numpy数组），用于后续切片
    gene_indices = {
        gene: np.where(expression[gene].values > threshold)[0]
        for gene in genes
    }
    # 初始化结果矩阵 (先用numpy，最后转换为DataFrame)
    result = np.full((n_genes, n_genes), np.nan, dtype=float)

    # 计算有向距离矩阵
    for i_idx, gene_i in enumerate(genes):
        idx_i = gene_indices[gene_i]
        if idx_i.size == 0:
            continue  # 基因 i 无表达，跳过整行
        # 将源基因 i 的坐标和表达权重转移到 GPU
        coords_i = coords_tensor[idx_i]  # shape: (Ni, dims)
        weights_i = torch.from_numpy(expression.iloc[idx_i, i_idx].values).to(device=device, dtype=torch.float32)
        Ni = coords_i.shape[0]
        for j_idx, gene_j in enumerate(genes):
            idx_j = gene_indices[gene_j]
            if idx_j.size == 0:
                # 基因 j 无表达
                result[i_idx, j_idx] = np.nan
                continue
            coords_j = coords_tensor[idx_j]  # (Nj, dims) on GPU
            Nj = coords_j.shape[0]
            # 计算距离矩阵大小并判断是否需要分批 (以float32计，每个距离占4字节)
            # 如果 Ni*Nj 过大，拆分为多次计算
            max_elements = int((max_memory_gb * (1024 ** 3)) / 4)  # 可容纳的距离元素个数
            if Ni * Nj <= max_elements:
                # 一次性计算距离并取最近邻
                dist_matrix = torch.cdist(coords_i, coords_j, p=2.0)  # 得到 (Ni, Nj) 距离矩阵
                min_dists, _ = torch.min(dist_matrix, dim=1)  # 每个 i点最近的 j点距离 (Ni,)
                # 按权重计算平均距离
                avg_dist = (min_dists * weights_i).sum() / weights_i.sum()
                result[i_idx, j_idx] = avg_dist.item()
            else:
                # 分批计算以避免 GPU 内存不足
                min_dists = torch.full((Ni,), float('inf'), device=device)
                chunk_size = max(1, max_elements // Ni)  # 基于Ni大小反推每批允许的Nj数目
                for start in range(0, Nj, chunk_size):
                    end = min(Nj, start + chunk_size)
                    dist_chunk = torch.cdist(coords_i, coords_j[start:end], p=2.0)
                    # 当前批次每个 i点的最近距离
                    batch_min, _ = torch.min(dist_chunk, dim=1)
                    # 更新全局最近距离
                    min_dists = torch.minimum(min_dists, batch_min)
                    # 释放显存（显式删除局部张量）
                    del dist_chunk, batch_min
                    torch.cuda.empty_cache()
                avg_dist = (min_dists * weights_i).sum() / weights_i.sum()
                result[i_idx, j_idx] = avg_dist.item()
    # 转回CPU构建 DataFrame
    dist_df = pd.DataFrame(result, index=genes, columns=genes)
    return dist_df
