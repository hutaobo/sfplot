import os
import tarfile

import h5py
import pandas as pd
import scanpy as sc
from spatialdata_io import xenium


def load_xenium_data(folder: str, normalize: bool = True):
    """
    加载并预处理来自指定文件夹的 10X Xenium 数据，返回处理后的 AnnData 对象。

    当文件夹下有 analysis 文件夹和 csv 文件时，直接读取；如果只有 analysis.tar.gz，
    则先解压再读取；若都没有则尝试从 analysis.h5 中读取信息。

    参数:
    --------
    folder : str
        Xenium 数据所在的文件夹路径。
    normalize : bool, optional
        是否进行归一化处理。默认值为 True。

    返回值:
    --------
    anndata.AnnData
        包含预处理后表达矩阵及注释信息的 AnnData 对象。
    """
    # 1. 从指定文件夹中加载 Xenium 数据，这里只获取细胞表格(cells_table=True)
    sdata = xenium(
        folder,
        cells_boundaries=False,
        nucleus_boundaries=False,
        cells_as_circles=False,
        cells_labels=False,
        nucleus_labels=False,
        transcripts=False,
        morphology_mip=False,
        morphology_focus=False,
        aligned_images=False,
        cells_table=True,
    )

    # 2. 将该样本的 AnnData 对象拷贝出来，避免修改原始的 sdata
    adata = sdata.tables["table"].copy()

    # 让 obs 中的 cell_id 都变为字符串，便于后续合并
    adata.obs["cell_id"] = adata.obs["cell_id"].astype(str)

    # =============== 尝试读取/解压/或从 H5 中获取聚类信息 ===============
    # 我们需要的聚类 CSV 路径
    cluster_path = os.path.join(
        folder, "analysis", "clustering", "gene_expression_graphclust", "clusters.csv"
    )

    # 我们需要的 UMAP CSV 路径
    umap_path = os.path.join(
        folder, "analysis", "umap", "gene_expression_2_components", "projection.csv"
    )

    # 标记是否成功从文件/压缩包中拿到了 cluster.csv 和 projection.csv
    got_csv = False

    if os.path.exists(cluster_path) and os.path.exists(umap_path):
        # 1) 直接读取 analysis 文件夹下的 csv
        print("检测到已存在 analysis 目录和相关 csv 文件，直接读取...")
        cluster = pd.read_csv(cluster_path)
        df_umap = pd.read_csv(umap_path)
        got_csv = True

    else:
        # 如果没找到 clusters.csv，就尝试解压 analysis.tar.gz
        tar_path = os.path.join(folder, "analysis.tar.gz")
        if os.path.exists(tar_path):
            print("未检测到完整的 analysis 文件夹或相应 csv，准备解压 analysis.tar.gz...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(folder)

            # 解压完成后再次检查
            if os.path.exists(cluster_path) and os.path.exists(umap_path):
                print("解压完毕，找到 clusters.csv 与 projection.csv，开始读取...")
                cluster = pd.read_csv(cluster_path)
                df_umap = pd.read_csv(umap_path)
                got_csv = True

    # 如果 got_csv = False，说明既没有现成的 csv，也没有 tar.gz 能解压出 csv
    # 这时就尝试从 analysis.h5 中获取信息
    if not got_csv:
        h5_path = os.path.join(folder, "analysis.h5")
        if not os.path.exists(h5_path):
            raise FileNotFoundError(
                "没有找到 cluster.csv，也没有 analysis.tar.gz 解压，"
                "且不存在 analysis.h5 文件，无法获取聚类和 UMAP 信息。"
            )
        print("从 analysis.h5 中读取聚类和 UMAP 信息...")

        with h5py.File(h5_path, "r") as f:
            # --- 读取聚类信息 ---
            # 这里以 gene_expression_graphclust 为例，如果你想要 kmeans 等其他聚类，
            # 可以改成相应的路径，比如 'clustering/_gene_expression_kmeans_5_clusters/clusters'
            if "clustering/_gene_expression_graphclust/clusters" not in f:
                raise ValueError("analysis.h5 中不存在 gene_expression_graphclust 的聚类信息！")
            clusters = f["clustering/_gene_expression_graphclust/clusters"][:]  # (64192, ) int64

            # 读取所有细胞的 barcode
            if "matrix/barcodes" not in f:
                raise ValueError("analysis.h5 中不存在 matrix/barcodes 信息！")
            barcodes = f["matrix/barcodes"][:]  # (64192, ) bytes类型

            # 将 bytes -> str
            barcodes = [b.decode("utf-8") for b in barcodes]
            # 将 cluster 转为字符串
            clusters_str = [str(c) for c in clusters]

            # 构建一个与原来 CSV 类似的 DataFrame，含 Barcode 和 Cluster
            cluster = pd.DataFrame({"Barcode": barcodes, "Cluster": clusters_str})

            # --- 读取 UMAP 信息 ---
            # 这里以 'umap/_gene_expression_2/transformed_umap_matrix' 为例
            if "umap/_gene_expression_2/transformed_umap_matrix" not in f:
                raise ValueError("analysis.h5 中不存在 gene_expression_2 的 UMAP 信息！")
            umap_matrix = f["umap/_gene_expression_2/transformed_umap_matrix"][:]  # (64192, 2)
            df_umap = pd.DataFrame(umap_matrix, columns=["UMAP-1", "UMAP-2"])
            df_umap["Barcode"] = barcodes

    # =============== 统一后续处理 ===============
    # 将 cluster 的 'Barcode'、'Cluster' 转为 str，保证后续合并时格式一致
    cluster["Barcode"] = cluster["Barcode"].astype(str)
    cluster["Cluster"] = cluster["Cluster"].astype(str)

    cluster_map = cluster.set_index("Barcode")["Cluster"]
    adata.obs["Cluster"] = adata.obs["cell_id"].map(cluster_map)

    df_umap["Barcode"] = df_umap["Barcode"].astype(str)
    df_umap = df_umap.set_index("Barcode")

    # ========== 关键修改：先做交集对齐 ==========
    adata_barcodes = adata.obs["cell_id"]
    umap_barcodes = df_umap.index

    # 1) 找到两者的交集
    adata_idx = pd.Index(adata_barcodes.unique())
    common_barcodes = adata_idx.intersection(umap_barcodes)

    if len(common_barcodes) == 0:
        raise ValueError("没有任何重叠的条目，无法对齐 adata.obs['cell_id'] 和 UMAP barcodes！")

    # 2) 如果希望“只保留交集内的细胞”，可以对 adata 做一个子集筛选
    #    这样 adata 仅包含在 UMAP 文件中也出现的细胞
    adata = adata[adata.obs["cell_id"].isin(common_barcodes)].copy()

    # 3) 现在 adata 和 df_umap 可以按照 adata.obs['cell_id'] 顺序安全索引
    adata.obsm["X_umap"] = df_umap.loc[adata.obs["cell_id"], ["UMAP-1", "UMAP-2"]].values

    # 将 adata 保存到 raw
    adata.raw = adata

    if normalize:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.scale(adata, max_value=10)

    return adata
