from pathlib import Path
import pandas as pd
import tempfile, shutil, os
from spatialdata_io import visium

def read_visium_bin(base: Path, dataset_id: str, use_filtered: bool = True, keep_tmp: bool = False):
    """
    适配 spatialdata-io 0.3.0, 读取含 Parquet 坐标的 Visium HD 输出
    不向 base 写任何文件
    """
    spatial_dir = base / "spatial"
    pqt = spatial_dir / "tissue_positions.parquet"
    if not pqt.exists():
        raise FileNotFoundError(f"{pqt} 不存在")

    pos = pd.read_parquet(pqt)
    if "barcode" not in pos.columns:
        pos = pos.rename_axis("barcode").reset_index()

    # 统一列名
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
    for need in ["barcode","in_tissue","array_row","array_col","pxl_row_in_fullres","pxl_col_in_fullres"]:
        if need not in pos.columns: pos[need] = 0
    pos = pos[["barcode","in_tissue","array_row","array_col","pxl_row_in_fullres","pxl_col_in_fullres"]]

    # 创建影子目录结构
    shadow_dir = Path(tempfile.mkdtemp(prefix=f"visium_shadow_{dataset_id}_"))
    shadow_spatial = shadow_dir / "spatial"
    shadow_spatial.mkdir(parents=True, exist_ok=True)

    # 写出 tissue_positions_list.csv (无表头)
    pos.to_csv(shadow_spatial / "tissue_positions_list.csv", index=False, header=False)
    # 拷贝 scalefactors
    shutil.copy2(spatial_dir / "scalefactors_json.json", shadow_spatial / "scalefactors_json.json")

    # 链接或复制 counts
    counts_file = "filtered_feature_bc_matrix.h5" if use_filtered else "raw_feature_bc_matrix.h5"
    counts_src = base / counts_file
    counts_shadow = shadow_dir / counts_file
    try:
        os.symlink(counts_src, counts_shadow)
    except Exception:
        shutil.copy2(counts_src, counts_shadow)

    try:
        # 调用 visium：不传 tissue_positions_file，让它自动发现
        sdata = visium(
            path=shadow_dir,
            dataset_id=dataset_id,
            counts_file=counts_file,
            scalefactors_file="spatial/scalefactors_json.json",
        )
    finally:
        if not keep_tmp:
            shutil.rmtree(shadow_dir, ignore_errors=True)

    return sdata
