import glob
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from typing import List, Tuple, Dict

# ========== 1.2) Utility: get the table link key ==========
def _get_instance_key(adata) -> str:
    """
    The SpatialData link key is typically in adata.uns['spatialdata_attrs']['instance_key'].
    Common values are 'instance_id' or 'cell_id'.
    """
    sdattrs = adata.uns.get("spatialdata_attrs", {}) if hasattr(adata, "uns") else {}
    return sdattrs.get("instance_key", "instance_id")

# ========== 2) Merge Xenium clustering into adata.obs ==========
def merge_xenium_clusters_into_adata(
    sdata,
    xenium_dir: str,
    table_key: str = "table",
    clustering_root: str = "analysis/clustering",
    barcode_col: str = "Barcode",
    cluster_col: str = "Cluster",
) -> Tuple["anndata.AnnData", List[str], Dict[str, float]]:
    """
    Auto-collect xenium_dir/analysis/clustering/**/clusters.csv
    and merge clustering columns into sdata.tables[table_key].obs.
    Prefers linking via obs['cell_id']; falls back to shapes index mapping if unavailable.
    Returns (adata, list of new column names, per-column non-NA hit rate report).
    """
    adata = sdata.tables[table_key]
    obs = adata.obs
    obs_index = adata.obs_names.astype(str)

    # Find the barcode column in obs (prefer cell_id)
    obs_barcode_col = "cell_id" if "cell_id" in obs.columns else None

    # If not found, try building a label_id->barcode mapping from cell_boundaries (fallback)
    label_to_barcode = None
    if obs_barcode_col is None:
        cb_gdf = _get_cell_boundaries_gdf(sdata).reset_index()
        cols = {c.lower(): c for c in cb_gdf.columns}
        if "cell_id" in cols and ("label_id" in cols or "label" in cols or "index" in cols):
            cell_id_col = cols["cell_id"]
            label_col = cols.get("label_id") or cols.get("label") or "index"
            def _norm(x):
                s = str(x)
                return s[:-2] if s.endswith(".0") else s
            tmp = cb_gdf[[label_col, cell_id_col]].drop_duplicates().copy()
            tmp["label_id_norm"] = tmp[label_col].map(_norm)
            label_to_barcode = pd.Series(tmp[cell_id_col].astype(str).values,
                                         index=tmp["label_id_norm"].values)

    # Iterate over clusters.csv files
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
        # Normalize column names
        cols = {c.lower(): c for c in df.columns}
        bcol = cols.get(barcode_col.lower()) or cols.get("barcode") or cols.get("barcodes")
        ccol = cols.get(cluster_col.lower()) or cols.get("cluster")
        if bcol is None or ccol is None:
            raise ValueError(f"{f} is missing barcode/cluster columns (requires {barcode_col}, {cluster_col})")

        # Three paths: prefer cell_id; then label->barcode; finally try obs_names
        if obs_barcode_col is not None:
            mapper = pd.Series(df[ccol].values, index=df[bcol].astype(str))
            adata.obs[colname] = adata.obs[obs_barcode_col].astype(str).map(mapper)
        elif label_to_barcode is not None:
            bc_to_cluster = pd.Series(df[ccol].values, index=df[bcol].astype(str))
            def _norm(x):
                s = str(x)
                return s[:-2] if s.endswith(".0") else s
            idx_norm = adata.obs_names.astype(str).map(_norm)
            adata.obs[colname] = idx_norm.map(label_to_barcode).map(bc_to_cluster)
        else:
            # Fallback (usually low hit rate)
            mapper = pd.Series(df[ccol].values, index=df[bcol].astype(str))
            adata.obs[colname] = obs_index.map(mapper)

        adata.obs[colname] = adata.obs[colname].astype("category")
        added_cols.append(colname)
        hitrate[colname] = float((~adata.obs[colname].isna()).mean())

    return adata, added_cols, hitrate

# ========== 3) Normalize transcript column names to align with XOA conventions ==========
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

# ========== 3.1) Utility: convert transcripts to GeoDataFrame ==========
def _transcripts_to_gdf(tx_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Guess x/y column names and convert to GeoDataFrame (crs=None).
    """
    x_candidates = ["x_location", "x", "X", "x_um", "global_x"]
    y_candidates = ["y_location", "y", "Y", "y_um", "global_y"]
    xcol = next((c for c in x_candidates if c in tx_df.columns), None)
    ycol = next((c for c in y_candidates if c in tx_df.columns), None)
    if xcol is None or ycol is None:
        raise KeyError("transcripts is missing x/y coordinate columns (could not identify from common column names).")
    geom = gpd.points_from_xy(tx_df[xcol], tx_df[ycol])
    return gpd.GeoDataFrame(tx_df.copy(), geometry=geom, crs=None)
