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
    # 1. Load Xenium data from the specified folder; only retrieve cell table (cells_table=True)
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

    # 2. Copy the AnnData object for this sample to avoid modifying the original sdata
    adata = sdata.tables["table"].copy()

    # Convert all cell_id values in obs to strings for easier downstream merging
    adata.obs["cell_id"] = adata.obs["cell_id"].astype(str)

    # =============== Try reading/extracting/or getting cluster info from H5 ===============
    # Path to the clustering CSV we need
    cluster_path = os.path.join(
        folder, "analysis", "clustering", "gene_expression_graphclust", "clusters.csv"
    )

    # Path to the UMAP CSV we need
    umap_path = os.path.join(
        folder, "analysis", "umap", "gene_expression_2_components", "projection.csv"
    )

    # Flag indicating whether cluster.csv and projection.csv were successfully obtained
    got_csv = False

    if os.path.exists(cluster_path) and os.path.exists(umap_path):
        # 1) Read CSVs directly from the analysis folder
        print("Detected existing analysis directory and related CSV files, reading directly...")
        cluster = pd.read_csv(cluster_path)
        df_umap = pd.read_csv(umap_path)
        got_csv = True

    else:
        # If clusters.csv is not found, try extracting analysis.tar.gz
        tar_path = os.path.join(folder, "analysis.tar.gz")
        if os.path.exists(tar_path):
            print("Complete analysis folder or CSV files not detected, preparing to extract analysis.tar.gz...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(folder)

            # Check again after extraction
            if os.path.exists(cluster_path) and os.path.exists(umap_path):
                print("Extraction complete, found clusters.csv and projection.csv, starting to read...")
                cluster = pd.read_csv(cluster_path)
                df_umap = pd.read_csv(umap_path)
                got_csv = True

    # If got_csv = False, there is no ready-made CSV and no tar.gz that can extract one
    # In this case, try to get information from analysis.h5
    if not got_csv:
        h5_path = os.path.join(folder, "analysis.h5")
        if not os.path.exists(h5_path):
            raise FileNotFoundError(
                "cluster.csv not found, analysis.tar.gz not extractable, "
                "and analysis.h5 does not exist; cannot obtain clustering and UMAP information."
            )
        print("Reading clustering and UMAP information from analysis.h5...")

        with h5py.File(h5_path, "r") as f:
            # --- Read clustering information ---
            # Example uses gene_expression_graphclust; for kmeans or other clustering,
            # change the path, e.g. 'clustering/_gene_expression_kmeans_5_clusters/clusters'
            if "clustering/_gene_expression_graphclust/clusters" not in f:
                raise ValueError("gene_expression_graphclust clustering info not found in analysis.h5!")
            clusters = f["clustering/_gene_expression_graphclust/clusters"][:]  # (64192, ) int64

            # Read barcodes for all cells
            if "matrix/barcodes" not in f:
                raise ValueError("matrix/barcodes info not found in analysis.h5!")
            barcodes = f["matrix/barcodes"][:]  # (64192,) bytes type

            # Convert bytes -> str
            barcodes = [b.decode("utf-8") for b in barcodes]
            # Convert cluster to string
            clusters_str = [str(c) for c in clusters]

            # Build a DataFrame similar to the original CSV, containing Barcode and Cluster
            cluster = pd.DataFrame({"Barcode": barcodes, "Cluster": clusters_str})

            # --- Read UMAP information ---
            # Example uses 'umap/_gene_expression_2/transformed_umap_matrix'
            if "umap/_gene_expression_2/transformed_umap_matrix" not in f:
                raise ValueError("gene_expression_2 UMAP info not found in analysis.h5!")
            umap_matrix = f["umap/_gene_expression_2/transformed_umap_matrix"][:]  # (64192, 2)
            df_umap = pd.DataFrame(umap_matrix, columns=["UMAP-1", "UMAP-2"])
            df_umap["Barcode"] = barcodes

    # =============== Unified downstream processing ===============
    # Convert 'Barcode' and 'Cluster' to str to ensure consistent format for downstream merging
    cluster["Barcode"] = cluster["Barcode"].astype(str)
    cluster["Cluster"] = cluster["Cluster"].astype(str)

    cluster_map = cluster.set_index("Barcode")["Cluster"]
    adata.obs["Cluster"] = adata.obs["cell_id"].map(cluster_map)

    df_umap["Barcode"] = df_umap["Barcode"].astype(str)
    df_umap = df_umap.set_index("Barcode")

    # ========== Key step: align by intersection first ==========
    adata_barcodes = adata.obs["cell_id"]
    umap_barcodes = df_umap.index

    # 1) Find intersection
    adata_idx = pd.Index(adata_barcodes.unique())
    common_barcodes = adata_idx.intersection(umap_barcodes)

    if len(common_barcodes) == 0:
        raise ValueError("No overlapping entries; cannot align adata.obs['cell_id'] with UMAP barcodes!")

    # 2) 如果希望“只保留交集内的细胞”，可以对 adata 做一个子集筛选
    #    这样 adata 仅包含在 UMAP 文件中也出现的细胞
    adata = adata[adata.obs["cell_id"].isin(common_barcodes)].copy()

    # 3) Now adata and df_umap can be safely indexed in the order of adata.obs['cell_id']
    adata.obsm["X_umap"] = df_umap.loc[adata.obs["cell_id"], ["UMAP-1", "UMAP-2"]].values

    # Save adata to raw
    adata.raw = adata

    if normalize:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.scale(adata, max_value=10)

    return adata
