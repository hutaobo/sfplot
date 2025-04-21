#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tbc_analysis.py  ——  多进程、共享内存、省内存版
  • 大表 coords → multiprocessing.shared_memory，只占一份物理内存
  • _process_gene 捕获所有异常，返回 None 不中断
  • chunksize=1 + maxtasksperchild 控制内存峰值
"""

from __future__ import annotations
import os, gc, logging, traceback, warnings
from multiprocessing import Pool, shared_memory
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from sfplot import (
    load_xenium_data,
    compute_cophenetic_distances_from_df,
    plot_cophenetic_heatmap,
)
from spatialdata_io import xenium

# ─── Logging & Warnings ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)-15s | %(message)s",
)
for lib in ("sfplot", "spatialdata", "anndata", "seaborn"):
    logging.getLogger(lib).setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*uncondensed distance matrix.*")

# ─── Globals visible *inside workers*  ───────────────────────────────
_adata_obs: Optional[pd.DataFrame]       = None     # small
_row_coph_global: Optional[pd.DataFrame] = None     # small
_coords_global: Optional[pd.DataFrame]   = None     # shared‑mem view
_coph_method: str                        = "average"
_shm: Optional[shared_memory.SharedMemory] = None   # keep handle alive

# -------------------------------------------------------------------- #
def _init_worker(shm_name, shm_shape, shm_dtype,
                 adata_obs_df, row_coph_df, coph_method: str):
    """
    Attach to existing shared‑memory block and rebuild coords DataFrame.
    """
    global _coords_global, _adata_obs, _row_coph_global, _coph_method, _shm

    # 1) attach
    _shm = shared_memory.SharedMemory(name=shm_name)
    arr  = np.ndarray(shm_shape, dtype=shm_dtype, buffer=_shm.buf)

    # 2) rebuild DataFrame (zero‑copy)
    _coords_global = pd.DataFrame.from_records(
        arr, columns=["x", "y", "feature_name"]
    )
    _coords_global["feature_name"] = _coords_global["feature_name"].astype(str)
    _coords_global.flags.writeable = False   # just for safety

    # 3) small globals sent via pickle
    _adata_obs       = adata_obs_df
    _row_coph_global = row_coph_df
    _coph_method     = coph_method

    logging.info(
        f"Worker {os.getpid()} ready: coords={len(_coords_global):,}, "
        f"genes={len(_row_coph_global)}"
    )

# -------------------------------------------------------------------- #
def _process_gene(gene: str) -> Optional[pd.DataFrame]:
    """Return one‑row DataFrame or None if failure/empty."""
    if _coords_global is None:
        logging.error("Globals not initialised in worker")
        return None
    try:
        sub = _coords_global[_coords_global["feature_name"] == gene]

        # case ①: gene 无空间坐标
        if sub.empty:
            if gene not in _row_coph_global.index:
                empty_cols = [
                    col for col in _row_coph_global.columns if col != gene
                ]
                return pd.DataFrame(
                    [[np.nan] * len(empty_cols)],
                    index=[gene],
                    columns=empty_cols,
                )
            row = _row_coph_global.loc[gene].drop(labels=gene, errors="ignore")
            return pd.DataFrame([row.values], index=[gene], columns=row.index)

        # case ②: 正常计算一行 cophenetic
        df = pd.concat(
            [
                _adata_obs,
                sub[["x", "y", "feature_name"]]
                .rename(columns={"feature_name": "celltype"})
                .assign(celltype=gene),
            ],
            ignore_index=True,
        )
        row_coph, _ = compute_cophenetic_distances_from_df(
            df, "x", "y", "celltype", None, _coph_method
        )
        series = row_coph.loc[gene].drop(gene, errors="ignore")
        return pd.DataFrame([series.values], index=[gene], columns=series.index)

    except Exception as err:
        logging.error(f"Worker failed on gene {gene}: {err}")
        logging.error(traceback.format_exc())
        return None

# -------------------------------------------------------------------- #
def transcript_by_cell_analysis(
    folder: str,
    sample_name: str | None = None,
    output_folder: str | None = None,
    coph_method: str = "average",
    n_jobs: int = 32,
    maxtasks: int = 50,
):
    """
    Transcript‑by‑cell 空间分析（共享内存 + 多进程）
    """
    global _adata_obs, _row_coph_global

    # 1) 路径与输出
    sample_name  = sample_name or os.path.basename(os.path.normpath(folder))
    output_folder = output_folder or f"./t_by_c_{sample_name}"
    os.makedirs(output_folder, exist_ok=True)

    # 2) 读取数据
    adata = load_xenium_data(folder, normalize=False)
    sdata = xenium(
        folder,
        cells_boundaries=False, nucleus_boundaries=False,
        cells_as_circles=False, cells_labels=False,
        nucleus_labels=False, transcripts=True,
        morphology_mip=False, morphology_focus=False,
        aligned_images=False, cells_table=False,
    )

    # 3) 过滤转录本
    coords = sdata.points["transcripts"]
    try:
        coords = coords.compute()
    except AttributeError:
        pass
    coords["feature_name"] = coords["feature_name"].astype(str)
    coords = coords.loc[
        ~coords["feature_name"].str.contains("NegControl|Unassigned", na=False),
        ["x", "y", "feature_name", "cell_id"],
    ]

    # 4) 构建共享内存 (coords 只要 x,y,gene)
    coords_arr = coords[["x", "y", "feature_name"]].to_records(index=False)
    shm = shared_memory.SharedMemory(create=True, size=coords_arr.nbytes)
    np.ndarray(coords_arr.shape, coords_arr.dtype, buffer=shm.buf)[:] = coords_arr

    # 5) adata obs & 全局 row‑cophenetic
    adata.obs["x"], adata.obs["y"] = adata.obsm["spatial"][:, 0], adata.obsm["spatial"][:, 1]
    _adata_obs = adata.obs[["x", "y", "Cluster"]].rename(columns={"Cluster": "celltype"})
    _row_coph_global, _ = compute_cophenetic_distances_from_df(
        _adata_obs, "x", "y", "celltype", None, coph_method
    )
    plot_cophenetic_heatmap(
        _row_coph_global, "row_coph", output_folder, f"StructureMap_of_{sample_name}.pdf", sample=sample_name
    )
    _row_coph_global.to_csv(f"{output_folder}/StructureMap_table_{sample_name}.csv")

    # 6) 并行处理基因
    genes = list(adata.var.index)
    out_csv = f"{output_folder}/t_and_c_result_{sample_name}.csv"
    header_flag = [True]

    with open(out_csv, "w", newline="") as fout, Pool(
        processes=n_jobs,
        initializer=_init_worker,
        initargs=(
            shm.name, coords_arr.shape, coords_arr.dtype,
            _adata_obs, _row_coph_global, coph_method,
        ),
        maxtasksperchild=maxtasks,
    ) as pool:
        for df_gene in tqdm(
            pool.imap_unordered(_process_gene, genes, chunksize=1),
            total=len(genes),
            desc="Processing genes",
            ncols=80,
        ):
            if df_gene is None:
                continue
            df_gene.to_csv(fout, header=header_flag[0], index=True)
            header_flag[0] = False
            del df_gene
            gc.collect()

    # 7) 清理共享内存
    shm.close()
    shm.unlink()

    print(f"[DONE] Outputs saved to: {output_folder}")

# -------------------------------------------------------------------- #
if __name__ == "__main__":   # 简单 CLI demo
    import sys
    transcript_by_cell_analysis(sys.argv[1] if len(sys.argv) > 1 else ".")
