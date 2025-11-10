from pathlib import Path
import pandas as pd
import tempfile, shutil, os
from spatialdata_io import visium

def read_visium_bin(
    base: Path,
    dataset_id: str,
    use_filtered: bool = True,
    tmpdir: Path | None = None,
    keep_tmp: bool = False,
):
    """
    读取 Visium/Visium HD 的 binned_outputs 层（只读 base 友好）。
    - 不向 base 写入任何文件
    - 兼容 spatialdata-io==0.3.0 对 tissue_positions 必须在 path/spatial 下的限制
    - 始终在影子目录 shadow_dir/spatial 下生成 tissue_positions_list.csv(无表头) 和 tissue_positions.csv(有表头)
    """

    spatial_dir = base / "spatial"
    pqt = spatial_dir / "tissue_positions.parquet"
    csv_v2 = spatial_dir / "tissue_positions.csv"
    csv_v1 = spatial_dir / "tissue_positions_list.csv"

    # ---------- 1) 读取 tissue positions 并统一为 DataFrame ----------
    def _load_positions() -> pd.DataFrame:
        if pqt.exists():
            df = pd.read_parquet(pqt)
        elif csv_v2.exists():
            # 先尝试当作有表头读取；若没有表头则降级为 header=None
            tmp = pd.read_csv(csv_v2)
            if "barcode" in tmp.columns:
                df = tmp
            else:
                tmp = pd.read_csv(csv_v2, header=None)
                tmp.columns = ["barcode","in_tissue","array_row","array_col","pxl_row_in_fullres","pxl_col_in_fullres"]
                df = tmp
        elif csv_v1.exists():
            df = pd.read_csv(csv_v1, header=None)
            df.columns = ["barcode","in_tissue","array_row","array_col","pxl_row_in_fullres","pxl_col_in_fullres"]
        else:
            raise FileNotFoundError(
                f"找不到 tissue positions：\n  - {pqt}\n  - {csv_v2}\n  - {csv_v1}"
            )

        # 统一列名（容错各种别名）
        if "barcode" not in df.columns:
            df = df.rename_axis("barcode").reset_index()

        rename = {}
        if "array_row" not in df.columns:
            for cand in ["row", "array_y", "grid_y", "spot_row"]:
                if cand in df.columns: rename[cand] = "array_row"; break
        if "array_col" not in df.columns:
            for cand in ["col", "array_x", "grid_x", "spot_col"]:
                if cand in df.columns: rename[cand] = "array_col"; break
        if "pxl_col_in_fullres" not in df.columns:
            for cand in ["pxl_x", "pxl_col", "x", "image_x", "pxl_col_fullres"]:
                if cand in df.columns: rename[cand] = "pxl_col_in_fullres"; break
        if "pxl_row_in_fullres" not in df.columns:
            for cand in ["pxl_y", "pxl_row", "y", "image_y", "pxl_row_fullres"]:
                if cand in df.columns: rename[cand] = "pxl_row_in_fullres"; break
        if "in_tissue" not in df.columns:
            for cand in ["inTissue", "intissue", "in_tissue_flag", "is_tissue"]:
                if cand in df.columns: rename[cand] = "in_tissue"; break
        if rename:
            df = df.rename(columns=rename)

        # 填补缺列并校正类型
        for need in ["barcode","in_tissue","array_row","array_col","pxl_row_in_fullres","pxl_col_in_fullres"]:
            if need not in df.columns:
                df[need] = 0
        for c in ["in_tissue","array_row","array_col","pxl_row_in_fullres","pxl_col_in_fullres"]:
            try:
                df[c] = df[c].astype(int)
            except Exception:
                pass

        # 统一列顺序（注意 list.csv 的顺序）
        df = df[["barcode","in_tissue","array_row","array_col","pxl_row_in_fullres","pxl_col_in_fullres"]]
        return df

    pos = _load_positions()

    # ---------- 2) 构建影子目录 ----------
    shadow_root = Path(
        tempfile.mkdtemp(prefix=f"visium_shadow_{dataset_id}_", dir=str(tmpdir) if tmpdir else None)
    )
    shadow_spatial = shadow_root / "spatial"
    shadow_spatial.mkdir(parents=True, exist_ok=True)

    # ---------- 3) 在影子目录下写出两种 tissue_positions 文件 ----------
    # v1: 无表头（tissue_positions_list.csv）
    pos.to_csv(shadow_spatial / "tissue_positions_list.csv", index=False, header=False)
    # v2: 有表头（tissue_positions.csv）
    pos.to_csv(shadow_spatial / "tissue_positions.csv", index=False)

    # ---------- 4) 准备 scalefactors ----------
    sc_src = spatial_dir / "scalefactors_json.json"
    if not sc_src.exists():
        # 对于大多数流程，scalefactors 是必需的；缺失时尽早报错更明确
        # 你也可以把这里改成可选：若缺失则传 None（看你工作流是否允许）
        try:
            if not keep_tmp:
                shutil.rmtree(shadow_root, ignore_errors=True)
        finally:
            raise FileNotFoundError(f"缺少 scalefactors 文件：{sc_src}")
    shutil.copy2(sc_src, shadow_spatial / "scalefactors_json.json")

    # ---------- 5) 准备 counts（软链优先，失败则复制；传字符串文件名） ----------
    counts_name = "filtered_feature_bc_matrix.h5" if use_filtered else "raw_feature_bc_matrix.h5"
    counts_src = base / counts_name
    if not counts_src.exists():
        # 某些流程名为 feature_bc_matrix.h5
        alt = base / "feature_bc_matrix.h5"
        if alt.exists():
            counts_src = alt
            counts_name = alt.name
        else:
            try:
                if not keep_tmp:
                    shutil.rmtree(shadow_root, ignore_errors=True)
            finally:
                raise FileNotFoundError(
                    f"计数文件不存在：{base/counts_name} 或 {alt}"
                )

    shadow_counts = shadow_root / counts_name
    try:
        os.symlink(counts_src, shadow_counts)
    except Exception:
        shutil.copy2(counts_src, shadow_counts)

    # ---------- 6) 调用 visium；不传 tissue_positions_file 让它自动发现 ----------
    try:
        sdata = visium(
            path=shadow_root,
            dataset_id=dataset_id,
            counts_file=shadow_counts.name,  # 传字符串，而非 Path，避免 .endswith 报错
            scalefactors_file="spatial/scalefactors_json.json",
            # fullres_image_file 如需也可处理，同样复制/软链后传 "spatial/xxx.png"
        )
    finally:
        if not keep_tmp:
            shutil.rmtree(shadow_root, ignore_errors=True)

    return sdata
