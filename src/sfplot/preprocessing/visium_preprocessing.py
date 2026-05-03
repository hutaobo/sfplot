from pathlib import Path
import importlib
import os
import shutil
import tempfile

import pandas as pd


def _load_visium_reader():
    try:
        return importlib.import_module("spatialdata_io").visium
    except ImportError as exc:
        raise ImportError(
            "read_visium_bin requires spatialdata_io and its spatialdata/ome_zarr/zarr "
            "dependency stack. Please install compatible versions before using this helper."
        ) from exc


def read_visium_bin(base: Path, dataset_id: str, use_filtered: bool = True, keep_tmp: bool = False):
    """
    Adapter for spatialdata-io 0.3.0, reads Visium HD output containing Parquet coordinates.
    Does not write any files to base.
    """
    spatial_dir = base / "spatial"
    pqt = spatial_dir / "tissue_positions.parquet"
    if not pqt.exists():
        raise FileNotFoundError(f"{pqt} does not exist")

    pos = pd.read_parquet(pqt)
    if "barcode" not in pos.columns:
        pos = pos.rename_axis("barcode").reset_index()

    # Normalize column names
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

    # Create shadow directory structure
    shadow_dir = Path(tempfile.mkdtemp(prefix=f"visium_shadow_{dataset_id}_"))
    shadow_spatial = shadow_dir / "spatial"
    shadow_spatial.mkdir(parents=True, exist_ok=True)

    # Write tissue_positions_list.csv (no header)
    pos.to_csv(shadow_spatial / "tissue_positions_list.csv", index=False, header=False)
    # Copy scalefactors
    shutil.copy2(spatial_dir / "scalefactors_json.json", shadow_spatial / "scalefactors_json.json")

    # Symlink or copy counts
    counts_file = "filtered_feature_bc_matrix.h5" if use_filtered else "raw_feature_bc_matrix.h5"
    counts_src = base / counts_file
    counts_shadow = shadow_dir / counts_file
    try:
        os.symlink(counts_src, counts_shadow)
    except Exception:
        shutil.copy2(counts_src, counts_shadow)

    try:
        # Call visium: without passing tissue_positions_file, let it auto-discover
        visium = _load_visium_reader()
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
