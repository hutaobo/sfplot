#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tbc_analysis_serial.py — single-process version of transcript-by-cell analysis.

This module implements `transcript_by_cell_analysis_serial`, which performs the
same analysis as the original multi-processing function but *sequentially* to
avoid possible dead-locks or hangs in fork/spawn environments.

Author: Your Name
Date  : 2025-06-27
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from spatialdata_io import xenium
from .transcript_batching import (
    iter_gene_batches,
    load_transcript_batch,
    open_transcript_batch_source,
)

# --- sfplot utilities -------------------------------------------------
from sfplot import (
    load_xenium_data,
    compute_cophenetic_distances_from_df,   # core metric computation
    plot_cophenetic_heatmap,                # nice clustered heat-map
)

# --------------------------------------------------------------------- #
def _prepare_obs_df(
    adata,
    group_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build a small DataFrame with columns ['x','y','celltype'] for distance
    computation. If *group_df* is supplied, its 'group' column will be used
    as *celltype*; otherwise `adata.obs['Cluster']` is used.
    """
    adata.obs["x"] = adata.obsm["spatial"][:, 0]
    adata.obs["y"] = adata.obsm["spatial"][:, 1]

    if group_df is not None:
        adata.obs = adata.obs.merge(
            group_df[["cell_id", "group"]], on="cell_id", how="left"
        )
        celltype_col = "group"
    else:
        celltype_col = "Cluster"

    obs_df = (
        adata.obs[["x", "y", celltype_col]]
        .rename(columns={celltype_col: "celltype"})
        .copy()
    )
    return obs_df


def _compute_structure_map(
    obs_df: pd.DataFrame,
    out_dir: str,
    sample: str,
    method: str = "average",
):
    """
    Compute row-wise cophenetic distance matrix (StructureMap) and save both
    heat-map PDF and numeric CSV.
    """
    row_coph, _ = compute_cophenetic_distances_from_df(      # heavy math
        df=obs_df,
        x_col="x",
        y_col="y",
        celltype_col="celltype",
        output_dir=None,
        method=method,
    )
    # plot & export
    plot_cophenetic_heatmap(
        row_coph,
        matrix_name="row_coph",
        output_dir=out_dir,
        output_filename=f"StructureMap_of_{sample}.pdf",
        sample=sample,
    )
    row_coph.to_csv(f"{out_dir}/StructureMap_table_{sample}.csv")
    return row_coph


def _default_gene_row(row_coph_global: pd.DataFrame, gene: str) -> pd.DataFrame:
    """Return the fallback output row for genes with no transcript coordinates."""
    if gene in row_coph_global.index:
        row = row_coph_global.loc[gene].drop(gene, errors="ignore")
        return pd.DataFrame([row.values], index=[gene], columns=row.index)

    empty_cols = [c for c in row_coph_global.columns if c != gene]
    return pd.DataFrame([[np.nan] * len(empty_cols)], index=[gene], columns=empty_cols)


def _load_all_transcripts(folder: str) -> pd.DataFrame:
    """
    Fallback path for datasets that do not expose `transcripts.parquet`.

    The preferred path is the parquet-backed streaming reader, but keeping this
    fallback preserves compatibility with older Xenium outputs.
    """
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
        coords = coords.compute()              # dask -> pandas if necessary
    except AttributeError:
        pass
    coords["feature_name"] = coords["feature_name"].astype(str)
    return coords.loc[
        ~coords["feature_name"].str.contains("NegControl|Unassigned", na=False),
        ["x", "y", "feature_name"],
    ]


# ============================  PUBLIC API  ============================ #
def transcript_by_cell_analysis_serial(
    folder: str,
    sample_name: Optional[str] = None,
    output_folder: Optional[str] = None,
    coph_method: str = "average",
    df: Optional[pd.DataFrame] = None,
    gene_batch_size: int = 64,
):
    """
    Run transcript-by-cell spatial analysis *sequentially* (single process).

    Parameters
    ----------
    folder : str
        Path to a Xenium sample directory.
    sample_name : str, optional
        Name shown in output files; defaults to folder basename.
    output_folder : str, optional
        Where to write results; defaults to `./t_by_c_<sample>`.
    coph_method : str, optional
        Linkage method for cophenetic distance; default 'average'.
    df : pd.DataFrame, optional
        Optional DataFrame with columns ['cell_id','group'] to override default
        cluster annotations.

    Returns
    -------
    None
    """
    # ---- 0. paths & I/O ----------------------------------------------
    sample = sample_name or os.path.basename(os.path.normpath(folder))
    out_dir = output_folder or f"./t_by_c_{sample}"
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1. load data -------------------------------------------------
    print(f"[{sample}] Loading data …")
    adata = load_xenium_data(folder, normalize=False)
    if gene_batch_size <= 0:
        raise ValueError("gene_batch_size must be a positive integer.")

    transcript_source = open_transcript_batch_source(folder)
    coords_fallback: Optional[pd.DataFrame] = None
    if transcript_source is None:
        coords_fallback = _load_all_transcripts(folder)

    # ---- 3. prepare observation table & structure map ----------------
    obs_df = _prepare_obs_df(adata, df)
    row_coph_global = _compute_structure_map(obs_df, out_dir, sample, coph_method)

    # ---- 4. iterate genes sequentially -------------------------------
    genes = list(adata.var.index)
    result_csv = os.path.join(out_dir, f"t_and_c_result_{sample}.csv")
    header_written = False

    print(f"[{sample}] Processing {len(genes):,} genes in streamed batches …")
    with open(result_csv, "w", newline="") as fout, tqdm(total=len(genes), ncols=80, desc="Genes") as progress:
        for gene_batch in iter_gene_batches(genes, gene_batch_size):
            if transcript_source is not None:
                coords = load_transcript_batch(transcript_source, gene_batch)
            else:
                coords = coords_fallback[coords_fallback["feature_name"].isin(gene_batch)].copy()

            batch_lookup = {
                gene: coords.loc[coords["feature_name"] == gene, ["x", "y"]]
                for gene in gene_batch
            }

            for gene in gene_batch:
                sub = batch_lookup[gene]
                if sub.empty:
                    df_gene = _default_gene_row(row_coph_global, gene)
                else:
                    tmp_df = pd.concat(
                        [
                            obs_df,
                            sub.assign(celltype=gene),
                        ],
                        ignore_index=True,
                    )
                    row_new, _ = compute_cophenetic_distances_from_df(
                        df=tmp_df,
                        x_col="x",
                        y_col="y",
                        celltype_col="celltype",
                        output_dir=None,
                        method=coph_method,
                    )
                    series = row_new.loc[gene].drop(gene, errors="ignore")
                    df_gene = pd.DataFrame([series.values], index=[gene], columns=series.index)

                df_gene.to_csv(fout, header=not header_written, index=True)
                header_written = True

            progress.update(len(gene_batch))

    print(f"[DONE] Outputs saved to: {out_dir}")


# ------------------------- CLI helper ---------------------------------
if __name__ == "__main__":
    folder_arg = sys.argv[1] if len(sys.argv) > 1 else "."
    transcript_by_cell_analysis_serial(folder_arg)
