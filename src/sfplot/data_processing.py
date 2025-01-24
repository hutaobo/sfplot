import os
import tarfile
import pandas as pd
import scanpy as sc
from spatialdata_io import xenium


def load_xenium_data(folder: str, normalize: bool = True):
    """
    加载并预处理来自指定文件夹的 10X Xenium 数据，返回处理后的 AnnData 对象。

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

    # 3. 优先检查是否已经存在解压后的 clusters.csv 文件
    #    期望解压后的路径：folder/analysis/clustering/gene_expression_graphclust/clusters.csv
    cluster_path = os.path.join(
        folder, "analysis", "clustering", "gene_expression_graphclust", "clusters.csv"
    )

    if not os.path.exists(cluster_path):
        # 如果不存在，就尝试解压 analysis.tar.gz
        tar_path = os.path.join(folder, "analysis.tar.gz")
        if not os.path.exists(tar_path):
            raise FileNotFoundError(
                f"未找到已解压的 {cluster_path}，也未找到压缩文件 {tar_path}！"
            )

        print("未检测到已解压的 analysis 文件夹，准备解压 analysis.tar.gz...")

        # 解压到与 analysis 同级的位置（即解压到 folder 下）
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(folder)

        # 解压完成后再次检查 cluster_path
        if not os.path.exists(cluster_path):
            raise FileNotFoundError(
                f"解压后仍未找到 {cluster_path}，请确认压缩包内是否包含正确的文件路径。"
            )

    # 4. 读取 cluster 文件（包含每个 Barcode 对应的 Cluster 信息）
    cluster = pd.read_csv(cluster_path)

    # 5. 将 cluster 表格的 'Barcode' 和 'Cluster' 字段转换为字符串，保证后续合并时格式一致
    cluster["Barcode"] = cluster["Barcode"].astype(str)
    cluster["Cluster"] = cluster["Cluster"].astype(str)

    # 6. 将 adata.obs 中的 'cell_id' 转换为字符串，便于与 cluster 文件中的 'Barcode' 进行匹配
    adata.obs["cell_id"] = adata.obs["cell_id"].astype(str)

    # 7. 创建从 Barcode 到 Cluster 的映射字典，并映射到 AnnData 的观测值(obs)中
    cluster_map = cluster.set_index("Barcode")["Cluster"]
    adata.obs["Cluster"] = adata.obs["cell_id"].map(cluster_map)

    # 导入UMAP信息
    umap_path = os.path.join(
        folder, "analysis", "umap", "gene_expression_2_components", "projection.csv"
    )
    df_umap = pd.read_csv(umap_path)
    df_umap = df_umap.set_index("Barcode")
    # 如果你的 adata.obs_names 跟 df_umap 的 index 完全一致且顺序相同（或可以相互对应），可以直接 loc
    adata.obsm["X_umap"] = df_umap.loc[adata.obs['cell_id'], ["UMAP-1", "UMAP-2"]].values

    # 8. 将当前表达矩阵保存到 adata.raw
    adata.raw = adata

    if normalize:
        # 9. 归一化总计数，每个细胞总计数标准化为 10,000
        sc.pp.normalize_total(adata, target_sum=1e4)

        # 10. 进行对数转换（Log1p）
        sc.pp.log1p(adata)

        # 11. 对数据进行标准化（Z-score），并将绝对值超过 10 的值裁剪至 10
        sc.pp.scale(adata, max_value=10)

    return adata
