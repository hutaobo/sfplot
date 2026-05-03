from __future__ import annotations

import importlib
import os
import tarfile
from pathlib import Path
from typing import Optional

import h5py
import pandas as pd


def _load_scanpy_module():
    try:
        return importlib.import_module("scanpy")
    except ImportError as exc:
        raise ImportError(
            "scanpy is required for Xenium table-bundle loading. Install scanpy to use this helper."
        ) from exc


def _normalize_if_requested(adata, normalize: bool):
    if not normalize:
        return adata
    sc = _load_scanpy_module()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    return adata


def _resolve_single_path(folder: Path, explicit_path: Optional[str | os.PathLike[str]], pattern: str) -> Path:
    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Required file does not exist: {path}")
        return path

    matches = sorted(folder.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find a file matching {pattern!r} under {folder}")
    if len(matches) > 1:
        names = ", ".join(str(path.name) for path in matches[:5])
        raise FileExistsError(f"Expected one file matching {pattern!r} under {folder}, found: {names}")
    return matches[0]


def _pick_existing_column(frame: pd.DataFrame, candidates: list[str], label: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"Could not resolve {label}; tried columns: {candidates}")


def load_xenium_table_bundle(
    folder: str | os.PathLike[str],
    *,
    cells_path: Optional[str | os.PathLike[str]] = None,
    cell_groups_path: Optional[str | os.PathLike[str]] = None,
    feature_matrix_path: Optional[str | os.PathLike[str]] = None,
    normalize: bool = False,
    cluster_col: str = "Clusters",
    cell_id_col: str = "Barcode",
    x_col: str = "x_centroid",
    y_col: str = "y_centroid",
):
    """
    Load a Xenium run from the stable table-bundle route used by the Atera benchmark.

    This helper avoids the ``spatialdata_io`` stack and instead assembles an
    ``AnnData`` object from:

    - ``cells.parquet``
    - ``*_cell_groups.csv``
    - ``cell_feature_matrix.h5``

    The returned object keeps the original official cluster labels in
    ``adata.obs[cluster_col]`` and mirrors them into ``adata.obs["Cluster"]`` for
    backward compatibility with the existing ``sfplot`` API.
    """

    folder = Path(folder)
    cells_path = _resolve_single_path(folder, cells_path, "cells.parquet")
    cell_groups_path = _resolve_single_path(folder, cell_groups_path, "*_cell_groups.csv")
    feature_matrix_path = _resolve_single_path(folder, feature_matrix_path, "cell_feature_matrix.h5")

    sc = _load_scanpy_module()
    cells = pd.read_parquet(cells_path, columns=["cell_id", x_col, y_col])
    cells["cell_id"] = cells["cell_id"].astype(str)

    cell_groups = pd.read_csv(cell_groups_path)
    resolved_cell_id_col = cell_id_col if cell_id_col in cell_groups.columns else _pick_existing_column(
        cell_groups, ["Barcode", "cell_id"], "cell-id column"
    )
    resolved_cluster_col = cluster_col if cluster_col in cell_groups.columns else _pick_existing_column(
        cell_groups, ["Clusters", "group", "Cluster"], "cluster column"
    )
    cell_groups = cell_groups.copy()
    cell_groups[resolved_cell_id_col] = cell_groups[resolved_cell_id_col].astype(str)

    adata = sc.read_10x_h5(str(feature_matrix_path))
    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)
    adata.obs["cell_id"] = adata.obs_names.astype(str)

    merged = (
        adata.obs[["cell_id"]]
        .merge(cells.rename(columns={"cell_id": "cell_id"}), on="cell_id", how="left")
        .merge(
            cell_groups.rename(columns={resolved_cell_id_col: "cell_id", resolved_cluster_col: cluster_col}),
            on="cell_id",
            how="left",
        )
    )
    if merged[[x_col, y_col]].isna().any().any():
        missing = int(merged[[x_col, y_col]].isna().any(axis=1).sum())
        raise ValueError(
            f"{missing} cells from cell_feature_matrix.h5 are missing centroid coordinates in {cells_path}."
        )

    adata.obs = merged.set_index(pd.Index(adata.obs_names))
    adata.obsm["spatial"] = adata.obs[[x_col, y_col]].to_numpy(dtype=float)
    adata.obs[cluster_col] = adata.obs[cluster_col].astype("string")
    adata.obs["Cluster"] = adata.obs[cluster_col].astype("string")
    if "color" in adata.obs.columns:
        adata.obs["cluster_color"] = adata.obs["color"].astype("string")
    adata.raw = adata.copy()
    return _normalize_if_requested(adata, normalize=normalize)


def load_xenium_data(folder: str, normalize: bool = True):
    """
    Load and preprocess a Xenium run through ``spatialdata_io.xenium``.

    Notes
    -----
    This legacy loader depends on a compatible ``spatialdata_io`` / ``spatialdata`` /
    ``ome_zarr`` / ``zarr`` stack. On environments where those packages are version
    mismatched, prefer :func:`load_xenium_table_bundle`, which assembles the same
    benchmark inputs from ``cells.parquet`` + ``*_cell_groups.csv`` +
    ``cell_feature_matrix.h5`` without the ``spatialdata_io`` dependency chain.
    """
    try:
        spatialdata_io = importlib.import_module("spatialdata_io")
    except ImportError as exc:
        raise ImportError(
            "load_xenium_data requires spatialdata_io and its Xenium reader dependencies. "
            "If that stack is unavailable in your environment, use load_xenium_table_bundle instead."
        ) from exc

    # Load Xenium data from the specified folder; only retrieve cell table.
    sdata = spatialdata_io.xenium(
        folder,
        cells_boundaries=False,
        nucleus_boundaries=False,
        cells_as_circles=False,
        cells_labels=False,
        nucleus_labels=False,
        transcripts=False,
        morphology_mip=False,
        morphology_focus=False,
        aligned_images=False,
        cells_table=True,
    )

    # 2. Copy the AnnData object for this sample to avoid modifying the original sdata
    adata = sdata.tables["table"].copy()

    # Convert all cell_id values in obs to strings for easier downstream merging
    adata.obs["cell_id"] = adata.obs["cell_id"].astype(str)

    # =============== Try reading/extracting/or getting cluster info from H5 ===============
    # Path to the clustering CSV we need
    cluster_path = os.path.join(
        folder, "analysis", "clustering", "gene_expression_graphclust", "clusters.csv"
    )

    # Path to the UMAP CSV we need
    umap_path = os.path.join(
        folder, "analysis", "umap", "gene_expression_2_components", "projection.csv"
    )

    # Flag indicating whether cluster.csv and projection.csv were successfully obtained
    got_csv = False

    if os.path.exists(cluster_path) and os.path.exists(umap_path):
        # 1) Read CSVs directly from the analysis folder
        print("Detected existing analysis directory and related CSV files, reading directly...")
        cluster = pd.read_csv(cluster_path)
        df_umap = pd.read_csv(umap_path)
        got_csv = True

    else:
        # If clusters.csv is not found, try extracting analysis.tar.gz
        tar_path = os.path.join(folder, "analysis.tar.gz")
        if os.path.exists(tar_path):
            print("Complete analysis folder or CSV files not detected, preparing to extract analysis.tar.gz...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(folder)

            # Check again after extraction
            if os.path.exists(cluster_path) and os.path.exists(umap_path):
                print("Extraction complete, found clusters.csv and projection.csv, starting to read...")
                cluster = pd.read_csv(cluster_path)
                df_umap = pd.read_csv(umap_path)
                got_csv = True

    # If got_csv = False, there is no ready-made CSV and no tar.gz that can extract one
    # In this case, try to get information from analysis.h5
    if not got_csv:
        h5_path = os.path.join(folder, "analysis.h5")
        if not os.path.exists(h5_path):
            raise FileNotFoundError(
                "cluster.csv not found, analysis.tar.gz not extractable, "
                "and analysis.h5 does not exist; cannot obtain clustering and UMAP information."
            )
        print("Reading clustering and UMAP information from analysis.h5...")

        with h5py.File(h5_path, "r") as f:
            # --- Read clustering information ---
            # Example uses gene_expression_graphclust; for kmeans or other clustering,
            # change the path, e.g. 'clustering/_gene_expression_kmeans_5_clusters/clusters'
            if "clustering/_gene_expression_graphclust/clusters" not in f:
                raise ValueError("gene_expression_graphclust clustering info not found in analysis.h5!")
            clusters = f["clustering/_gene_expression_graphclust/clusters"][:]  # (64192, ) int64

            # Read barcodes for all cells
            if "matrix/barcodes" not in f:
                raise ValueError("matrix/barcodes info not found in analysis.h5!")
            barcodes = f["matrix/barcodes"][:]  # (64192,) bytes type

            # Convert bytes -> str
            barcodes = [b.decode("utf-8") for b in barcodes]
            # Convert cluster to string
            clusters_str = [str(c) for c in clusters]

            # Build a DataFrame similar to the original CSV, containing Barcode and Cluster
            cluster = pd.DataFrame({"Barcode": barcodes, "Cluster": clusters_str})

            # --- Read UMAP information ---
            # Example uses 'umap/_gene_expression_2/transformed_umap_matrix'
            if "umap/_gene_expression_2/transformed_umap_matrix" not in f:
                raise ValueError("gene_expression_2 UMAP info not found in analysis.h5!")
            umap_matrix = f["umap/_gene_expression_2/transformed_umap_matrix"][:]  # (64192, 2)
            df_umap = pd.DataFrame(umap_matrix, columns=["UMAP-1", "UMAP-2"])
            df_umap["Barcode"] = barcodes

    # =============== Unified downstream processing ===============
    # Convert 'Barcode' and 'Cluster' to str to ensure consistent format for downstream merging
    cluster["Barcode"] = cluster["Barcode"].astype(str)
    cluster["Cluster"] = cluster["Cluster"].astype(str)

    cluster_map = cluster.set_index("Barcode")["Cluster"]
    adata.obs["Cluster"] = adata.obs["cell_id"].map(cluster_map)

    df_umap["Barcode"] = df_umap["Barcode"].astype(str)
    df_umap = df_umap.set_index("Barcode")

    # ========== Key step: align by intersection first ==========
    adata_barcodes = adata.obs["cell_id"]
    umap_barcodes = df_umap.index

    # 1) Find intersection
    adata_idx = pd.Index(adata_barcodes.unique())
    common_barcodes = adata_idx.intersection(umap_barcodes)

    if len(common_barcodes) == 0:
        raise ValueError("No overlapping entries; cannot align adata.obs['cell_id'] with UMAP barcodes!")

    # 2) 如果希望“只保留交集内的细胞”，可以对 adata 做一个子集筛选
    #    这样 adata 仅包含在 UMAP 文件中也出现的细胞
    adata = adata[adata.obs["cell_id"].isin(common_barcodes)].copy()

    # 3) Now adata and df_umap can be safely indexed in the order of adata.obs['cell_id']
    adata.obsm["X_umap"] = df_umap.loc[adata.obs["cell_id"], ["UMAP-1", "UMAP-2"]].values

    # Save adata to raw
    adata.raw = adata
    return _normalize_if_requested(adata, normalize=normalize)
