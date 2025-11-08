import os, glob
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from typing import List, Tuple, Dict

# ========== 1.2) 工具：取表的链接键 ==========
def _get_instance_key(adata) -> str:
    """
    SpatialData 约定的链接键一般在 adata.uns['spatialdata_attrs']['instance_key'] 中。
    常见为 'instance_id' 或 'cell_id'。
    """
    sdattrs = adata.uns.get("spatialdata_attrs", {}) if hasattr(adata, "uns") else {}
    return sdattrs.get("instance_key", "instance_id")

# ========== 2) 合并 Xenium clustering 到 adata.obs ==========
def merge_xenium_clusters_into_adata(
    sdata,
    xenium_dir: str,
    table_key: str = "table",
    clustering_root: str = "analysis/clustering",
    barcode_col: str = "Barcode",
    cluster_col: str = "Cluster",
) -> Tuple["anndata.AnnData", List[str], Dict[str, float]]:
    """
    自动收集 xenium_dir/analysis/clustering/**/clusters.csv，
    并把聚类列合并进 sdata.tables[table_key].obs。
    优先用 obs['cell_id'] 连接；没有则尝试用 shapes 的索引映射。
    返回 (adata, 新增列名列表, 每列非NA命中率报告)。
    """
    adata = sdata.tables[table_key]
    obs = adata.obs
    obs_index = adata.obs_names.astype(str)

    # 找 obs 里的条形码列（优先 cell_id）
    obs_barcode_col = "cell_id" if "cell_id" in obs.columns else None

    # 若没有，尝试从 cell_boundaries 构造 label_id->barcode 的映射（作为兜底）
    label_to_barcode = None
    if obs_barcode_col is None:
        cb_gdf = _get_cell_boundaries_gdf(sdata).reset_index()
        cols = {c.lower(): c for c in cb_gdf.columns}
        if "cell_id" in cols and ("label_id" in cols or "label" in cols or "index" in cols):
            cell_id_col = cols["cell_id"]
            label_col = cols.get("label_id") or cols.get("label") or "index"
            def _norm(x):
                s = str(x);  return s[:-2] if s.endswith(".0") else s
            tmp = cb_gdf[[label_col, cell_id_col]].drop_duplicates().copy()
            tmp["label_id_norm"] = tmp[label_col].map(_norm)
            label_to_barcode = pd.Series(tmp[cell_id_col].astype(str).values,
                                         index=tmp["label_id_norm"].values)

    # 遍历 clusters.csv
    pattern = os.path.join(xenium_dir, clustering_root, "**", "clusters.csv")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No clusters.csv under {os.path.join(xenium_dir, clustering_root)}")

    added_cols, hitrate = [], {}
    for f in files:
        dirname = os.path.basename(os.path.dirname(f)).lower()

        if "graphclust" in dirname:
            colname = "xoa_graphclust"
        elif "kmeans" in dirname:
            digits = [t for t in dirname.split("_") if t.isdigit()]
            colname = f"xoa_kmeans_{digits[0]}" if digits else "xoa_kmeans"
        else:
            colname = "xoa_" + dirname.replace("gene_expression_", "").replace("_clusters", "")

        df = pd.read_csv(f)
        # 规范列
        cols = {c.lower(): c for c in df.columns}
        bcol = cols.get(barcode_col.lower()) or cols.get("barcode") or cols.get("barcodes")
        ccol = cols.get(cluster_col.lower()) or cols.get("cluster")
        if bcol is None or ccol is None:
            raise ValueError(f"{f} 缺少条形码/聚类列（需要 {barcode_col}, {cluster_col}）")

        # 三种路径：优先 cell_id；其次 label->barcode；最后用 obs_names 试试
        if obs_barcode_col is not None:
            mapper = pd.Series(df[ccol].values, index=df[bcol].astype(str))
            adata.obs[colname] = adata.obs[obs_barcode_col].astype(str).map(mapper)
        elif label_to_barcode is not None:
            bc_to_cluster = pd.Series(df[ccol].values, index=df[bcol].astype(str))
            def _norm(x):
                s = str(x);  return s[:-2] if s.endswith(".0") else s
            idx_norm = adata.obs_names.astype(str).map(_norm)
            adata.obs[colname] = idx_norm.map(label_to_barcode).map(bc_to_cluster)
        else:
            # 兜底（通常命中率低）
            mapper = pd.Series(df[ccol].values, index=df[bcol].astype(str))
            adata.obs[colname] = obs_index.map(mapper)

        adata.obs[colname] = adata.obs[colname].astype("category")
        added_cols.append(colname)
        hitrate[colname] = float((~adata.obs[colname].isna()).mean())

    return adata, added_cols, hitrate

# ========== 3) 规范化 transcripts 列名，尽量对齐 XOA 规范 ==========
def _normalize_transcript_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    if "x_location" not in df.columns:
        for c in ["x", "X", "x_um", "global_x"]:
            if c in df.columns: ren[c] = "x_location"; break
    if "y_location" not in df.columns:
        for c in ["y", "Y", "y_um", "global_y"]:
            if c in df.columns: ren[c] = "y_location"; break
    if "z_location" not in df.columns and "z" in df.columns:
        ren["z"] = "z_location"
    if "feature_name" not in df.columns:
        for c in ["gene", "FeatureName", "feature", "Feature_Name"]:
            if c in df.columns: ren[c] = "feature_name"; break
    if ren:
        df = df.rename(columns=ren)
    return df

# ========== 3.1) 工具：把 transcripts 变成 GeoDataFrame ==========
def _transcripts_to_gdf(tx_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    根据列名猜测 x/y，并转换为 GeoDataFrame（crs=None）。
    """
    x_candidates = ["x_location", "x", "X", "x_um", "global_x"]
    y_candidates = ["y_location", "y", "Y", "y_um", "global_y"]
    xcol = next((c for c in x_candidates if c in tx_df.columns), None)
    ycol = next((c for c in y_candidates if c in tx_df.columns), None)
    if xcol is None or ycol is None:
        raise KeyError("transcripts 缺少 x/y 坐标列（未能从常见列名中识别）。")
    geom = gpd.points_from_xy(tx_df[xcol], tx_df[ycol])
    return gpd.GeoDataFrame(tx_df.copy(), geometry=geom, crs=None)
