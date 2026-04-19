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
from .transcript_batching import (
    iter_gene_batches,
    load_transcript_batch,
    open_transcript_batch_source,
)

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

    # Rebuild a compact numeric DataFrame without copying Python string objects.
    _coords_global = pd.DataFrame.from_records(
        arr, columns=["x", "y", "gene_code"]
    )
    _coords_global.flags.writeable = False

    # set small globals from parent
    _adata_obs = adata_obs_df
    _row_coph_global = row_coph_df
    _coph_method = coph_method

    logging.info(
        f"Worker {os.getpid()} ready: coords={len(_coords_global):,}, "
        f"celltypes={len(_row_coph_global)}"
    )


def _default_gene_row(gene: str) -> pd.DataFrame:
    """Return the fallback output row for genes with no transcript coordinates."""
    if gene not in _row_coph_global.index:
        empty_cols = [col for col in _row_coph_global.columns if col != gene]
        return pd.DataFrame(
            [[np.nan] * len(empty_cols)],
            index=[gene],
            columns=empty_cols,
        )

    row = _row_coph_global.loc[gene].drop(labels=gene, errors="ignore")
    return pd.DataFrame([row.values], index=[gene], columns=row.index)


def _process_gene(gene_item: tuple[str, int]) -> Optional[pd.DataFrame]:
    """Return one-row DataFrame of cophenetic distances for a gene, or None if failure/empty."""
    if _coords_global is None:
        logging.error("Globals not initialised in worker")
        return None

    gene, gene_code = gene_item
    try:
        # Select transcripts for this gene within the current streamed batch.
        sub = _coords_global[_coords_global["gene_code"] == gene_code]

        # case 1: no spatial coordinates for gene
        if sub.empty:
            return _default_gene_row(gene)

        # case 2: compute cophenetic distances for gene-transcripts
        df = pd.concat(
            [
                _adata_obs,
                sub[["x", "y"]]
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


def _load_all_transcripts(folder: str) -> pd.DataFrame:
    """
    Fallback path for datasets that do not expose `transcripts.parquet`.

    This keeps the previous behavior available, but the preferred path is the
    parquet-backed streaming reader used for large Xenium outputs.
    """
    logging.warning(
        "Falling back to eager transcript loading because `transcripts.parquet` "
        "is unavailable. Large datasets may require substantial memory."
    )
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

    coords = sdata.points["transcripts"]
    try:
        coords = coords.compute()
    except AttributeError:
        pass
    coords["feature_name"] = coords["feature_name"].astype(str)
    return coords.loc[
        ~coords["feature_name"].str.contains("NegControl|Unassigned", na=False),
        ["x", "y", "feature_name"],
    ]


def transcript_by_cell_analysis(
    folder: str,
    sample_name: Optional[str] = None,
    output_folder: Optional[str] = None,
    coph_method: str = "average",
    n_jobs: int = 32,
    maxtasks: int = 50,
    df: Optional[pd.DataFrame] = None,
    gene_batch_size: int = 64,
):
    """
    Perform transcript-by-cell spatial analysis using shared memory and multiprocessing.

    Transcript coordinates are loaded in streamed gene batches when
    `transcripts.parquet` is available, which avoids materializing the entire
    transcript table in memory at once.
    """
    global _adata_obs, _row_coph_global

    # prepare output directory
    sample = sample_name or os.path.basename(os.path.normpath(folder))
    out_dir = output_folder or f"./t_by_c_{sample}"
    os.makedirs(out_dir, exist_ok=True)

    if gene_batch_size <= 0:
        raise ValueError("gene_batch_size must be a positive integer.")

    # load Xenium data and configure transcript source
    adata = load_xenium_data(folder, normalize=False)
    transcript_source = open_transcript_batch_source(folder)
    coords_fallback: Optional[pd.DataFrame] = None
    if transcript_source is None:
        coords_fallback = _load_all_transcripts(folder)

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

    with open(out_csv, "w", newline="") as fout, tqdm(
        total=len(genes),
        desc=f"Processing genes ({sample})",
        ncols=80,
        file=sys.stdout,
        mininterval=1.0,
    ) as progress:
        for gene_batch in iter_gene_batches(genes, gene_batch_size):
            if transcript_source is not None:
                coords = load_transcript_batch(transcript_source, gene_batch)
            else:
                coords = coords_fallback[coords_fallback["feature_name"].isin(gene_batch)].copy()

            if coords.empty:
                for gene in gene_batch:
                    df_gene = _default_gene_row(gene)
                    df_gene.to_csv(fout, header=not header_written, index=True)
                    header_written = True
                    del df_gene
                progress.update(len(gene_batch))
                gc.collect()
                continue

            gene_codes = {gene: code for code, gene in enumerate(gene_batch)}
            coords = coords.copy()
            coords["gene_code"] = coords["feature_name"].map(gene_codes)
            coords = coords.dropna(subset=["gene_code"])
            coords["gene_code"] = coords["gene_code"].astype(np.int32)
            coords_arr = coords[["x", "y", "gene_code"]].to_records(index=False)

            shm = shared_memory.SharedMemory(create=True, size=coords_arr.nbytes)
            try:
                np.ndarray(coords_arr.shape, coords_arr.dtype, buffer=shm.buf)[:] = coords_arr
                gene_tasks = [(gene, gene_codes[gene]) for gene in gene_batch]
                batch_results: dict[str, pd.DataFrame] = {}
                with Pool(
                    processes=n_jobs,
                    initializer=_init_worker,
                    initargs=(
                        shm.name,
                        coords_arr.shape,
                        coords_arr.dtype,
                        _adata_obs,
                        _row_coph_global,
                        coph_method,
                    ),
                    maxtasksperchild=maxtasks,
                ) as pool:
                    chunks = max(1, len(gene_tasks) // max(1, n_jobs * 2))
                    for df_gene in pool.imap_unordered(_process_gene, gene_tasks, chunksize=chunks):
                        if df_gene is None:
                            continue
                        batch_results[str(df_gene.index[0])] = df_gene

                for gene in gene_batch:
                    if gene in batch_results:
                        df_gene = batch_results[gene]
                    else:
                        df_gene = _default_gene_row(gene)
                    df_gene.to_csv(fout, header=not header_written, index=True)
                    header_written = True
                    del df_gene
            finally:
                shm.close()
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass

            progress.update(len(gene_batch))
            gc.collect()

    print(f"[DONE] Outputs saved to: {out_dir}")


if __name__ == "__main__":
    import sys
    transcript_by_cell_analysis(sys.argv[1] if len(sys.argv) > 1 else ".")
