from pathlib import Path
import pandas as pd
import shutil, os, tempfile
from spatialdata_io import visium

def read_visium_bin(
    base: Path,
    dataset_id: str,
    use_filtered: bool = True,
    workdir: Path | None = None,   # 自定义可写目录；None 用系统临时目录
    persist_tmp: bool = False,     # True 则保留影子目录，调试/规避惰性读取
):
    spatial_dir = base / "spatial"
    pqt_in = spatial_dir / "tissue_positions.parquet"
    csv_in = spatial_dir / "tissue_positions.csv"

    # 1) 读取 tissue positions（优先 parquet，其次 csv）
    if pqt_in.exists():
        pos = pd.read_parquet(pqt_in)
    elif csv_in.exists():
        pos = pd.read_csv(csv_in)
    else:
        raise FileNotFoundError(f"找不到 tissue_positions 文件：{pqt_in} 或 {csv_in}")

    # 2) 统一列名
    if "barcode" not in pos.columns:
        pos = pos.rename_axis("barcode").reset_index()

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
    if rename:
        pos = pos.rename(columns=rename)

    # 兜底补列
    for need in ["barcode","in_tissue","array_row","array_col","pxl_col_in_fullres","pxl_row_in_fullres"]:
        if need not in pos.columns:
            pos[need] = 0

    # 10x 常见顺序（顺序通常不敏感，但尽量对齐）
    cols = ["barcode","in_tissue","array_row","array_col","pxl_row_in_fullres","pxl_col_in_fullres"]
    pos = pos.loc[:, cols].copy()
    try:
        pos["in_tissue"] = pos["in_tissue"].astype(int)
    except Exception:
        pass

    # 3) 计数与 scalefactors
    counts_name = "filtered_feature_bc_matrix.h5" if use_filtered else "raw_feature_bc_matrix.h5"
    counts_src = base / counts_name
    if not counts_src.exists():
        raise FileNotFoundError(f"计数文件不存在：{counts_src}")

    sc_src = spatial_dir / "scalefactors_json.json"
    if not sc_src.exists():
        raise FileNotFoundError(f"缺少 scalefactors：{sc_src}")

    # 4) 构建影子目录结构
    tmp_root = Path(tempfile.mkdtemp(prefix=f"visium_shadow_{dataset_id}_", dir=str(workdir) if workdir else None))
    tmp_spatial = tmp_root / "spatial"
    tmp_spatial.mkdir(parents=True, exist_ok=True)

    # 写 positions 和拷贝 scalefactors（都在影子目录里）
    pos.to_csv(tmp_spatial / "tissue_positions.csv", index=False)
    shutil.copy2(sc_src, tmp_spatial / "scalefactors_json.json")

    # counts：优先做符号链接，失败则直接用绝对路径传给 visium
    dest_counts = tmp_root / counts_name
    linked = False
    try:
        os.symlink(counts_src, dest_counts)
        linked = True
    except Exception:
        # 某些平台可能禁用链接；没关系，下面直接传绝对路径
        pass
    counts_arg = dest_counts if linked else counts_src

    try:
        # 5) 让 visium 在影子目录里自动发现文件（不传 tissue_positions_file）
        sdata = visium(
            path=tmp_root,
            dataset_id=dataset_id,
            counts_file=counts_arg,
            scalefactors_file=tmp_spatial / "scalefactors_json.json",
            # fullres_image_file 可选；不需要就不拷贝原始图像，避免 I/O
        )
    finally:
        # 6) 清理影子目录（除非要求保留以兼容潜在惰性读取或调试）
        if not persist_tmp:
            try:
                shutil.rmtree(tmp_root)
            except Exception:
                pass

    return sdata
