from pathlib import Path
import pandas as pd
from spatialdata_io import visium

def read_visium_bin(base: Path, dataset_id: str, use_filtered: bool = True):
    """
    base: 形如 .../binned_outputs/square_016um 的目录
    dataset_id: 该层在 sdata 里的名字
    """
    pqt = base / "spatial" / "tissue_positions.parquet"
    csv = base / "spatial" / "tissue_positions.csv"
    pos = pd.read_parquet(pqt)

    # 统一出 barcode 列
    if "barcode" not in pos.columns:
        pos = pos.rename_axis("barcode").reset_index()

    # 智能改名
    rename = {}
    if "array_row" not in pos.columns:
        for cand in ["row", "array_y", "grid_y", "spot_row"]:
            if cand in pos.columns: rename[cand] = "array_row"; break
    if "array_col" not in pos.columns:
        for cand in ["col", "array_x", "grid_x", "spot_col"]:
            if cand in pos.columns: rename[cand] = "array_col"; break
    if "pxl_col_in_fullres" not in pos.columns:
        for cand in ["pxl_x", "pxl_col", "x", "image_x"]:
            if cand in pos.columns: rename[cand] = "pxl_col_in_fullres"; break
    if "pxl_row_in_fullres" not in pos.columns:
        for cand in ["pxl_y", "pxl_row", "y", "image_y"]:
            if cand in pos.columns: rename[cand] = "pxl_row_in_fullres"; break
    if "in_tissue" not in pos.columns:
        for cand in ["inTissue", "intissue", "in_tissue_flag", "is_tissue"]:
            if cand in pos.columns: rename[cand] = "in_tissue"; break
    pos = pos.rename(columns=rename)

    # 兜底补列，避免 KeyError（如你确认原始列名，补列会被覆盖，不影响）
    for need in ["barcode","in_tissue","array_row","array_col","pxl_col_in_fullres","pxl_row_in_fullres"]:
        if need not in pos.columns: pos[need] = 0

    cols = ["barcode","in_tissue","array_row","array_col","pxl_col_in_fullres","pxl_row_in_fullres"]
    pos[cols].to_csv(csv, index=False)

    sdata = visium(
        path=base,
        dataset_id=dataset_id,
        counts_file=("filtered_feature_bc_matrix.h5" if use_filtered else "raw_feature_bc_matrix.h5"),
        tissue_positions_file=csv,
        scalefactors_file=base / "spatial" / "scalefactors_json.json",
        # fullres_image_file=base / "spatial" / "detected_tissue_image.jpg",  # 需要时打开
    )
    return sdata
