"""
tbc_analysis.py
改进点：
  • _process_gene 增加异常捕获，避免 KeyError / OOM 后挂死
  • chunksize 固定为 1，及时回收内存
  • maxtasksperchild 参数可调
  • 主循环忽略返回 None 的任务，保证稳健
"""

import os
import gc
import traceback
import logging
import warnings
from multiprocessing import Pool
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

# ─── Logging & Warnings ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)-15s | %(message)s",
)
for lib in ("sfplot", "spatialdata", "anndata", "seaborn"):
    logging.getLogger(lib).setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*uncondensed distance matrix.*")

# ─── Globals for Fork‑COW Sharing ─────────────────────────────────────────────
_adata_obs: Optional[pd.DataFrame]      = None
_coords_global: Optional[pd.DataFrame]  = None
_row_coph_global: Optional[pd.DataFrame] = None
_coph_method: str                      = "average"

# --------------------------------------------------------------------------- #
#                               Worker helpers                                #
# --------------------------------------------------------------------------- #
def _init_worker(coph_method: str):
    """Initialize each worker with only the method string."""
    global _coph_method
    _coph_method = coph_method


def _process_gene(gene: str) -> Optional[pd.DataFrame]:
    """
    Compute per‑gene cophenetic row.
    如果出现异常（KeyError/OOM 等），返回 None 并把错误写进日志。
    """
    try:
        sub = _coords_global[_coords_global["feature_name"] == gene]

        # --- case ①: 无空间坐标 --------------------------------------------------
        if sub.empty:
            if gene not in _row_coph_global.index:
                # 完全没有全局行，用 NaN 占位，保持列数对齐
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

        # --- case ②: 正常计算 ----------------------------------------------------
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
            df,
            x_col="x",
            y_col="y",
            celltype_col="celltype",
            output_dir=None,
            method=_coph_method,
        )
        series = row_coph.loc[gene].drop(labels=gene, errors="ignore")
        return pd.DataFrame([series.values], index=[gene], columns=series.index)

    except Exception as err:
        logging.error(f"Worker failed on gene {gene}: {err}")
        logging.error(traceback.format_exc())
        return None


# --------------------------------------------------------------------------- #
#                       Top‑level analysis entry point                         #
# --------------------------------------------------------------------------- #
def transcript_by_cell_analysis(
    folder: str,
    sample_name: str | None = None,
    output_folder: str | None = None,
    coph_method: str = "average",
    n_jobs: int = 32,
    maxtasks: int = 20,
):
    """
    Transcript‑by‑cell 空间分析（多进程稳健版）
    """
    global _adata_obs, _coords_global, _row_coph_global

    # 1) 采样名称与输出目录 ------------------------------------------------------
    sample_name = sample_name or os.path.basename(os.path.normpath(folder))
    output_folder = output_folder or f"./t_by_c_{sample_name}"
    os.makedirs(output_folder, exist_ok=True)

    # 2) 预加载数据，利用 Fork‑COW ----------------------------------------------
    adata = load_xenium_data(folder, normalize=False)
    sdata = xenium(
        folder,
        cells_boundaries=False,
        nucleus_boundaries=False,
        cells_as_circles=False,
        cells_labels=False,
        nucleus_labels=False,
        transcripts=True,
        morphology_mip=False,
        morphology_focus=False,
        aligned_images=False,
        cells_table=False,
    )

    # 3) 转录本计数表 -----------------------------------------------------------
    clusters = adata.obs["Cluster"].values
    uniq = np.unique(clusters)
    counts = {
        cl: (
            adata.X[clusters == cl].sum(axis=0).A.flatten()
            if hasattr(adata.X, "A")
            else np.asarray(adata.X[clusters == cl].sum(axis=0)).flatten()
        )
        for cl in uniq
    }
    transcript_df = pd.DataFrame(counts, index=adata.var.index)

    # 4) 过滤 Neg/Unassigned 并统计 --------------------------------------------
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
    summary = (
        coords.groupby("feature_name")
        .agg(unassigned_count=("cell_id", lambda x: (x == "UNASSIGNED").sum()))
        .reset_index()
    )
    transcript_df = (
        transcript_df.merge(
            summary[["feature_name", "unassigned_count"]],
            left_index=True,
            right_on="feature_name",
            how="left",
        )
        .fillna(0)
        .assign(unassigned_count=lambda df: df["unassigned_count"].astype(int))
        .set_index("feature_name")
    )

    # 5) 输出 count & 百分比表 ---------------------------------------------------
    transcript_df.to_csv(f"{output_folder}/transcript_table_{sample_name}.csv")
    (
        transcript_df.div(transcript_df.sum(axis=1), axis=0) * 100
    ).to_csv(f"{output_folder}/transcript_table_percentage_{sample_name}.csv")

    # 6) 全局 cophenetic & heatmap ---------------------------------------------
    adata.obs["x"], adata.obs["y"] = (
        adata.obsm["spatial"][:, 0],
        adata.obsm["spatial"][:, 1],
    )
    _adata_obs = adata.obs[["x", "y", "Cluster"]].rename(
        columns={"Cluster": "celltype"}
    )
    _coords_global = coords[["x", "y", "feature_name"]]
    _row_coph_global, _ = compute_cophenetic_distances_from_df(
        _adata_obs,
        x_col="x",
        y_col="y",
        celltype_col="celltype",
        output_dir=None,
        method=coph_method,
    )
    plot_cophenetic_heatmap(
        matrix=_row_coph_global,
        matrix_name="row_coph",
        sample=sample_name,
        output_dir=output_folder,
        output_filename=f"StructureMap_of_{sample_name}.pdf",
    )
    _row_coph_global.to_csv(f"{output_folder}/StructureMap_table_{sample_name}.csv")

    # 7) 并行逐基因 cophenetic，按行流式写 CSV ----------------------------------
    genes = list(adata.var.index)
    out_csv = f"{output_folder}/t_and_c_result_{sample_name}.csv"
    header_flag = [True]  # 可变引用，写一次表头

    with open(out_csv, "w", newline="") as fout, Pool(
        processes=n_jobs,
        initializer=_init_worker,
        initargs=(coph_method,),
        maxtasksperchild=maxtasks,
    ) as pool:
        for df_gene in tqdm(
            pool.imap_unordered(_process_gene, genes, chunksize=1),
            total=len(genes),
            desc="Processing genes",
            ncols=80,
        ):
            if df_gene is None:  # 该任务失败，已记录日志
                continue
            df_gene.to_csv(fout, header=header_flag[0], index=True)
            header_flag[0] = False
            del df_gene
            gc.collect()

    print(f"[DONE] Outputs saved to: {output_folder}")
