from pathlib import Path
import pandas as pd
import tempfile
from spatialdata_io import visium

def read_visium_bin(
    base: Path,
    dataset_id: str,
    use_filtered: bool = True,
    tmpdir: Path | None = None,
    keep_tmp: bool = False,
):
    """
    base: 形如 .../binned_outputs/square_016um 的目录（只读也可）
    dataset_id: 该层在 sdata 里的名字
    tmpdir: 可写目录；若为 None 则用系统临时目录
    keep_tmp: 是否保留临时CSV（防止下游惰性读取失败）
    """
    spatial_dir = base / "spatial"
    pqt = spatial_dir / "tissue_positions.parquet"
    csv_existing = spatial_dir / "tissue_positions.csv"

    # 优先从 parquet 读取，否则再从 csv
    if pqt.exists():
        pos = pd.read_parquet(pqt)
    elif csv_existing.exists():
        pos = pd.read_csv(csv_existing)
    else:
        raise FileNotFoundError("Neither tissue_positions.parquet nor tissue_positions.csv found in spatial/.")

    # 统一列名
    if "barcode" not in pos.columns:
        pos = pos.rename_axis("barcode").reset_index()

    rename = {}
    if "array_row" not in pos.columns:
        for cand in ["row", "array_y", "grid_y", "spot_row"]:
            if cand in pos.columns:
                rename[cand] = "array_row"; break
    if "array_col" not in pos.columns:
        for cand in ["col", "array_x", "grid_x", "spot_col"]:
            if cand in pos.columns:
                rename[cand] = "array_col"; break
    if "pxl_col_in_fullres" not in pos.columns:
        for cand in ["pxl_x", "pxl_col", "x", "image_x"]:
            if cand in pos.columns:
                rename[cand] = "pxl_col_in_fullres"; break
    if "pxl_row_in_fullres" not in pos.columns:
        for cand in ["pxl_y", "pxl_row", "y", "image_y"]:
            if cand in pos.columns:
                rename[cand] = "pxl_row_in_fullres"; break
    if "in_tissue" not in pos.columns:
        for cand in ["inTissue", "intissue", "in_tissue_flag", "is_tissue"]:
            if cand in pos.columns:
                rename[cand] = "in_tissue"; break
    pos = pos.rename(columns=rename)

    for need in ["barcode","in_tissue","array_row","array_col","pxl_col_in_fullres","pxl_row_in_fullres"]:
        if need not in pos.columns: pos[need] = 0

    cols = ["barcode","in_tissue","array_row","array_col","pxl_col_in_fullres","pxl_row_in_fullres"]

    # 总是写入临时目录
    if tmpdir is not None:
        tmpdir = Path(tmpdir)
        tmpdir.mkdir(parents=True, exist_ok=True)
        tissue_csv = tmpdir / f"{dataset_id}__tissue_positions.csv"
    else:
        ntf = tempfile.NamedTemporaryFile(prefix=f"{dataset_id}__", suffix=".csv", delete=False)
        tissue_csv = Path(ntf.name)
        ntf.close()

    pos[cols].to_csv(tissue_csv, index=False)

    try:
        sdata = visium(
            path=base,
            dataset_id=dataset_id,
            counts_file=("filtered_feature_bc_matrix.h5" if use_filtered else "raw_feature_bc_matrix.h5"),
            tissue_positions_file=tissue_csv,
            scalefactors_file=spatial_dir / "scalefactors_json.json",
        )
    finally:
        if not keep_tmp:
            try:
                tissue_csv.unlink(missing_ok=True)
            except Exception:
                pass

    return sdata
