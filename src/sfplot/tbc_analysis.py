#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tbc_analysis.py — Multiprocess, shared memory, memory-efficient version
  • shared memory for coords to save physical memory
  • _process_gene catches exceptions and returns None without interrupting
  • chunksize=1 + maxtasksperchild controls memory peak
"""

from __future__ import annotations

import warnings
import logging
import os
import sys
import traceback
import gc
from multiprocessing import Pool, shared_memory
from typing import Optional

import numpy as np
import pandas as pd
from sfplot import (
    load_xenium_data,
    compute_cophenetic_distances_from_df,
    plot_cophenetic_heatmap,
)
from spatialdata_io import xenium
from tqdm import tqdm

# global logging and warnings configuration
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s %(levelname)-8s %(name)-15s | %(message)s"
)
# suppress logs from specific libraries
for lib in ("sfplot", "spatialdata", "anndata", "seaborn"):
    logging.getLogger(lib).setLevel(logging.ERROR)

# Globals visible inside worker processes
_adata_obs: Optional[pd.DataFrame] = None
_row_coph_global: Optional[pd.DataFrame] = None
_coords_global: Optional[pd.DataFrame] = None
_coph_method: str = "average"
_shm: Optional[shared_memory.SharedMemory] = None


def _init_worker(shm_name, shm_shape, shm_dtype, adata_obs_df, row_coph_df, coph_method: str):
    """
    Attach to existing shared memory block and rebuild coords DataFrame.
    """
    global _coords_global, _adata_obs, _row_coph_global, _coph_method, _shm

    # attach to the shared memory segment
    _shm = shared_memory.SharedMemory(name=shm_name)
    arr = np.ndarray(shm_shape, dtype=shm_dtype, buffer=_shm.buf)

    # rebuild DataFrame without copying
    _coords_global = pd.DataFrame.from_records(
        arr, columns=["x", "y", "feature_name"]
    )
    # ensure feature_name is string type
    _coords_global["feature_name"] = _coords_global["feature_name"].astype(str)
    _coords_global.flags.writeable = False

    # set small globals from parent
    _adata_obs = adata_obs_df
    _row_coph_global = row_coph_df
    _coph_method = coph_method

    logging.info(
        f"Worker {os.getpid()} ready: coords={len(_coords_global):,}, "
        f"celltypes={len(_row_coph_global)}"
    )


def _process_gene(gene: str) -> Optional[pd.DataFrame]:
    """Return one-row DataFrame of cophenetic distances for a gene, or None if failure/empty."""
    if _coords_global is None:
        logging.error("Globals not initialised in worker")
        return None
    try:
        # select transcripts for this gene
        sub = _coords_global[_coords_global["feature_name"] == gene]

        # case 1: no spatial coordinates for gene
        if sub.empty:
            if gene not in _row_coph_global.index:
                # build empty row if gene not in initial distances
                empty_cols = [col for col in _row_coph_global.columns if col != gene]
                return pd.DataFrame(
                    [[np.nan] * len(empty_cols)],
                    index=[gene],
                    columns=empty_cols,
                )
            # return global cophenetic row for gene
            row = _row_coph_global.loc[gene].drop(labels=gene, errors="ignore")
            return pd.DataFrame([row.values], index=[gene], columns=row.index)

        # case 2: compute cophenetic distances for gene-transcripts
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
            df=df,
            x_col="x",
            y_col="y",
            celltype_col="celltype",
            output_dir=None,
            method=_coph_method
        )
        series = row_coph.loc[gene].drop(gene, errors="ignore")
        return pd.DataFrame([series.values], index=[gene], columns=series.index)

    except Exception as err:
        # catch and log any error without interrupting
        logging.error(f"Worker failed on gene {gene}: {err}")
        logging.error(traceback.format_exc())
        return None


def transcript_by_cell_analysis(
    folder: str,
    sample_name: Optional[str] = None,
    output_folder: Optional[str] = None,
    coph_method: str = "average",
    n_jobs: int = 32,
    maxtasks: int = 50,
    df: Optional[pd.DataFrame] = None,
):
    """
    Perform transcript-by-cell spatial analysis using shared memory and multiprocessing.
    """
    global _adata_obs, _row_coph_global

    # prepare output directory
    sample = sample_name or os.path.basename(os.path.normpath(folder))
    out_dir = output_folder or f"./t_by_c_{sample}"
    os.makedirs(out_dir, exist_ok=True)

    # load Xenium data and transcripts
    adata = load_xenium_data(folder, normalize=False)
    sdata = xenium(
        folder,
        cells_boundaries=False, nucleus_boundaries=False,
        cells_as_circles=False, cells_labels=False,
        nucleus_labels=False, transcripts=True,
        morphology_mip=False, morphology_focus=False,
        aligned_images=False, cells_table=False,
    )

    # filter transcripts and convert feature_name to string
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

    # create shared memory block for transcript coords
    coords_arr = coords[["x", "y", "feature_name"]].to_records(index=False)
    shm = shared_memory.SharedMemory(create=True, size=coords_arr.nbytes)
    np.ndarray(coords_arr.shape, coords_arr.dtype, buffer=shm.buf)[:] = coords_arr

    # prepare cell coordinates and celltype annotations
    adata.obs["x"] = adata.obsm["spatial"][:, 0]
    adata.obs["y"] = adata.obsm["spatial"][:, 1]
    if df is not None:
        adata.obs = adata.obs.merge(df[["cell_id", "group"]], on="cell_id", how="left")
        _adata_obs = adata.obs[["x", "y", "group"]].rename(columns={"group": "celltype"})
    else:
        _adata_obs = adata.obs[["x", "y", "Cluster"]].rename(columns={"Cluster": "celltype"})

    # compute initial cophenetic distances between cell types
    _row_coph_global, _ = compute_cophenetic_distances_from_df(
        df=_adata_obs,
        x_col="x",
        y_col="y",
        celltype_col="celltype",
        output_dir=None,
        method=coph_method
    )
    plot_cophenetic_heatmap(
        _row_coph_global, "row_coph", out_dir, f"StructureMap_of_{sample}.pdf", sample=sample
    )
    _row_coph_global.to_csv(f"{out_dir}/StructureMap_table_{sample}.csv")

    # parallel processing of genes
    genes = list(adata.var.index)
    out_csv = f"{out_dir}/t_and_c_result_{sample}.csv"
    header_written = False

    try:
        # initialize process pool
        pool = Pool(
            processes=n_jobs,
            initializer=_init_worker,
            initargs=(
                shm.name, coords_arr.shape, coords_arr.dtype,
                _adata_obs, _row_coph_global, coph_method
            ),
            maxtasksperchild=maxtasks
        )
        # open output file and iterate with progress bar
        with open(out_csv, "w", newline="") as fout:
            print("Workers initialized, start processing genes ...")
            # compute chunksize for fewer dispatch calls
            chunks = max(1, len(genes) // (n_jobs * 2))
            iterable = pool.imap_unordered(_process_gene, genes, chunksize=chunks)
            for df_gene in tqdm(
                iterable,
                total=len(genes),
                desc=f"Processing genes ({sample})",
                ncols=80,
                file=sys.stdout,
                mininterval=1.0,
            ):
                if df_gene is None:
                    continue
                df_gene.to_csv(fout, header=not header_written, index=True)
                header_written = True
                del df_gene
                gc.collect()
    finally:
        # clean up pool and shared memory
        try:
            pool.close()
            pool.join()
        except Exception:
            pass
        shm.close()
        try:
            shm.unlink()
        except FileNotFoundError:
            pass

    print(f"[DONE] Outputs saved to: {out_dir}")


if __name__ == "__main__":
    import sys
    transcript_by_cell_analysis(sys.argv[1] if len(sys.argv) > 1 else ".")
