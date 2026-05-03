from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import issparse
from sklearn.neighbors import NearestNeighbors

from .searcher_findee_score import compute_cophenetic_distances_from_df, compute_cophenetic_from_distance_matrix


def _ensure_output_dir(output_dir: Optional[str | os.PathLike[str]]) -> Optional[Path]:
    if output_dir is None:
        return None
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _coerce_nonnegative(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    min_val = float(values.min()) if len(values) else 0.0
    if min_val < 0:
        values = values - min_val
    return values.clip(lower=0.0)


def _normalize_series(values: pd.Series) -> pd.Series:
    if values.empty:
        return values.astype(float)
    values = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    min_val = float(values.min())
    max_val = float(values.max())
    if math.isclose(min_val, max_val):
        if math.isclose(max_val, 0.0):
            return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
        return pd.Series(np.ones(len(values)), index=values.index, dtype=float)
    return (values - min_val) / (max_val - min_val)


def _normalize_frame_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    values = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    mins = values.min(axis=1)
    maxs = values.max(axis=1)
    spans = (maxs - mins).replace(0.0, np.nan)
    normalized = values.sub(mins, axis=0).div(spans, axis=0)
    normalized = normalized.fillna(0.0)
    constant_nonzero = spans.isna() & (maxs > 0)
    if constant_nonzero.any():
        normalized.loc[constant_nonzero] = 1.0
    return normalized


def _winsorized_minmax(values: np.ndarray, upper_quantile: float = 0.99) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if np.allclose(arr, arr.flat[0]):
        return np.zeros_like(arr) if math.isclose(float(arr.flat[0]), 0.0) else np.ones_like(arr)
    clipped = np.clip(arr, a_min=float(np.min(arr)), a_max=float(np.quantile(arr, upper_quantile)))
    min_val = float(np.min(clipped))
    max_val = float(np.max(clipped))
    if math.isclose(min_val, max_val):
        return np.zeros_like(clipped) if math.isclose(max_val, 0.0) else np.ones_like(clipped)
    return (clipped - min_val) / (max_val - min_val)


def _winsorized_normalize_series(values: pd.Series, upper_quantile: float = 0.99) -> pd.Series:
    if values.empty:
        return values.astype(float)
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    normalized = _winsorized_minmax(numeric.to_numpy(dtype=float), upper_quantile=upper_quantile)
    return pd.Series(normalized, index=numeric.index, dtype=float)


def _winsorized_normalize_frame(frame: pd.DataFrame, upper_quantile: float = 0.99) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    normalized = frame.copy().astype(float)
    for col in normalized.columns:
        normalized[col] = _winsorized_normalize_series(normalized[col], upper_quantile=upper_quantile)
    return normalized


def _robust_scale_columns(frame: pd.DataFrame, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    scaled = frame.copy().astype(float)
    for col in scaled.columns:
        values = scaled[col].to_numpy(dtype=float)
        if values.size == 0:
            continue
        lo = float(np.quantile(values, lower_quantile))
        hi = float(np.quantile(values, upper_quantile))
        if math.isclose(lo, hi):
            scaled[col] = 0.0 if math.isclose(hi, 0.0) else 1.0
            continue
        clipped = np.clip(values, lo, hi)
        scaled[col] = (clipped - lo) / (hi - lo)
    return scaled.fillna(0.0)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if len(values) == 0:
        return float("nan")
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights)
    if not valid.any():
        return float("nan")
    values = values[valid]
    weights = np.clip(weights[valid], 0.0, None)
    if np.allclose(weights.sum(), 0.0):
        return float(np.quantile(values, quantile))
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cutoff = float(np.clip(quantile, 0.0, 1.0)) * float(cumulative[-1])
    idx = int(np.searchsorted(cumulative, cutoff, side="left"))
    idx = min(idx, len(values) - 1)
    return float(values[idx])


def _aggregate_weighted_values(values: np.ndarray, weights: np.ndarray, method: str = "weighted_median") -> float:
    valid = np.isfinite(values) & np.isfinite(weights)
    if not valid.any():
        return float("nan")
    values = np.asarray(values[valid], dtype=float)
    weights = np.clip(np.asarray(weights[valid], dtype=float), 0.0, None)
    if method == "weighted_median":
        return _weighted_quantile(values, weights, 0.5)
    if method == "weighted_trimmed_mean":
        if np.allclose(weights.sum(), 0.0):
            return float(np.mean(values))
        low = _weighted_quantile(values, weights, 0.1)
        high = _weighted_quantile(values, weights, 0.9)
        keep = (values >= low) & (values <= high)
        if not keep.any():
            keep = np.ones(len(values), dtype=bool)
        return _weighted_average(values[keep], weights[keep])
    if method == "mean":
        return _weighted_average(values, weights)
    raise ValueError("method must be one of: weighted_median, weighted_trimmed_mean, mean")


def _safe_row_cophenetic(distance_matrix: pd.DataFrame, method: str = "average") -> pd.DataFrame:
    if distance_matrix.empty:
        return pd.DataFrame()
    if len(distance_matrix.index) == 1:
        idx = distance_matrix.index.astype(str)
        return pd.DataFrame([[0.0]], index=idx, columns=idx)
    row_coph, _ = compute_cophenetic_from_distance_matrix(distance_matrix, method=method, show_corr=False)
    return row_coph


def _weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights)
    if not valid.any():
        return float("nan")
    values_valid = values[valid]
    weights_valid = np.clip(weights[valid], 0.0, None)
    if np.allclose(weights_valid.sum(), 0.0):
        return float(np.mean(values_valid))
    return float(np.average(values_valid, weights=weights_valid))


def _safe_to_parquet(df: pd.DataFrame, path: Path) -> bool:
    try:
        df.to_parquet(path, index=False)
        return True
    except Exception:
        return False


def _save_heatmap(matrix: pd.DataFrame, title: str, output_prefix: Path, cmap: str = "mako") -> list[str]:
    fig_w = max(6.0, 0.45 * max(1, matrix.shape[1]))
    fig_h = max(4.0, 0.35 * max(1, matrix.shape[0]))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(matrix, cmap=cmap, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(matrix.columns.name or "")
    ax.set_ylabel(matrix.index.name or "")
    fig.tight_layout()
    paths: list[str] = []
    for ext in ("png", "pdf"):
        out = output_prefix.with_suffix(f".{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        paths.append(str(out))
    plt.close(fig)
    return paths


def _save_hotspot_overlay(
    reference_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    sender_mask: pd.Series,
    receiver_mask: pd.Series,
    sender_score: pd.Series,
    receiver_score: pd.Series,
    title: str,
    output_prefix: Path,
) -> list[str]:
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    ax.scatter(
        reference_df[x_col],
        reference_df[y_col],
        s=8,
        c="#D0D0D0",
        alpha=0.45,
        linewidths=0.0,
        label="Background cells",
    )

    if sender_mask.any():
        sender_cells = reference_df.loc[sender_mask]
        ax.scatter(
            sender_cells[x_col],
            sender_cells[y_col],
            s=24 + 36 * sender_score.loc[sender_mask].to_numpy(),
            c=sender_score.loc[sender_mask],
            cmap="Reds",
            alpha=0.85,
            linewidths=0.0,
            label="Sender hotspot",
        )

    if receiver_mask.any():
        receiver_cells = reference_df.loc[receiver_mask]
        ax.scatter(
            receiver_cells[x_col],
            receiver_cells[y_col],
            s=24 + 36 * receiver_score.loc[receiver_mask].to_numpy(),
            c=receiver_score.loc[receiver_mask],
            cmap="Blues",
            alpha=0.85,
            linewidths=0.0,
            label="Receiver hotspot",
            marker="s",
        )

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    paths: list[str] = []
    for ext in ("png", "pdf"):
        out = output_prefix.with_suffix(f".{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        paths.append(str(out))
    plt.close(fig)
    return paths


def _reference_from_adata(
    adata: Any,
    *,
    cluster_col: str = "Cluster",
    cell_id_col: str = "cell_id",
    x_col: str = "x",
    y_col: str = "y",
) -> pd.DataFrame:
    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] is required to derive the reference table.")
    if cluster_col not in adata.obs.columns:
        raise ValueError(f"{cluster_col!r} is missing from adata.obs.")
    if cell_id_col not in adata.obs.columns:
        raise ValueError(f"{cell_id_col!r} is missing from adata.obs.")

    spatial = np.asarray(adata.obsm["spatial"])
    if spatial.shape[1] < 2:
        raise ValueError("adata.obsm['spatial'] must contain at least two dimensions.")

    reference_df = pd.DataFrame(
        {
            cell_id_col: adata.obs[cell_id_col].astype(str).to_numpy(),
            x_col: spatial[:, 0],
            y_col: spatial[:, 1],
            "celltype": adata.obs[cluster_col].astype(str).to_numpy(),
        }
    )
    return reference_df


def _expression_from_adata(
    adata: Any,
    genes: Iterable[str],
    *,
    cell_id_col: str = "cell_id",
    use_raw: bool = False,
) -> pd.DataFrame:
    genes = list(dict.fromkeys(str(g) for g in genes))
    gene_index = adata.raw.var_names if use_raw and getattr(adata, "raw", None) is not None else adata.var_names
    present = [gene for gene in genes if gene in set(gene_index)]
    if not present:
        return pd.DataFrame(index=adata.obs[cell_id_col].astype(str).to_numpy())

    if use_raw and getattr(adata, "raw", None) is not None:
        matrix = adata.raw[:, present].X
    else:
        matrix = adata[:, present].X

    if issparse(matrix):
        matrix = matrix.toarray()
    else:
        matrix = np.asarray(matrix)

    return pd.DataFrame(matrix, index=adata.obs[cell_id_col].astype(str).to_numpy(), columns=present)


def _coerce_reference_df(
    reference_df: Optional[pd.DataFrame] = None,
    *,
    adata: Any = None,
    cluster_col: str = "Cluster",
    cell_id_col: str = "cell_id",
    x_col: str = "x",
    y_col: str = "y",
    celltype_col: str = "celltype",
) -> pd.DataFrame:
    if reference_df is None:
        if adata is None:
            raise ValueError("Either reference_df or adata must be provided.")
        reference_df = _reference_from_adata(adata, cluster_col=cluster_col, cell_id_col=cell_id_col, x_col=x_col, y_col=y_col)
    else:
        reference_df = reference_df.copy()
        if cell_id_col not in reference_df.columns:
            reference_df[cell_id_col] = [f"cell_{i}" for i in range(len(reference_df))]
        if celltype_col not in reference_df.columns:
            raise ValueError(f"reference_df must contain {celltype_col!r}.")
        rename_map = {}
        if celltype_col != "celltype":
            rename_map[celltype_col] = "celltype"
        if rename_map:
            reference_df = reference_df.rename(columns=rename_map)
    required = {cell_id_col, x_col, y_col, "celltype"}
    missing = required.difference(reference_df.columns)
    if missing:
        raise ValueError(f"reference_df is missing required columns: {sorted(missing)}")
    reference_df[cell_id_col] = reference_df[cell_id_col].astype(str)
    reference_df["celltype"] = reference_df["celltype"].astype(str)
    return reference_df[[cell_id_col, x_col, y_col, "celltype"]].copy()


def _coerce_expression_df(
    reference_df: pd.DataFrame,
    expression_df: Optional[pd.DataFrame] = None,
    *,
    adata: Any = None,
    genes: Optional[Iterable[str]] = None,
    cell_id_col: str = "cell_id",
    use_raw: bool = False,
) -> pd.DataFrame:
    if expression_df is None:
        if adata is None:
            raise ValueError("Either expression_df or adata must be provided.")
        expression_df = _expression_from_adata(adata, genes or [], cell_id_col=cell_id_col, use_raw=use_raw)
    else:
        expression_df = expression_df.copy()
        if cell_id_col in expression_df.columns:
            expression_df[cell_id_col] = expression_df[cell_id_col].astype(str)
            expression_df = expression_df.set_index(cell_id_col)
        expression_df.index = expression_df.index.astype(str)
        if genes is not None:
            present = [gene for gene in genes if gene in expression_df.columns]
            expression_df = expression_df.loc[:, present]

    aligned = expression_df.reindex(reference_df[cell_id_col]).fillna(0.0)
    aligned.index = reference_df[cell_id_col].astype(str)
    aligned = aligned.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return aligned


def _pick_matching_file(base: Path, patterns: Sequence[str]) -> Optional[Path]:
    for pattern in patterns:
        matches = sorted(base.glob(pattern))
        if matches:
            return matches[0]
    return None


def _load_matrix_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df.apply(pd.to_numeric, errors="coerce")


def _resolve_precomputed_tables(
    *,
    tbc_results: Optional[str | os.PathLike[str]] = None,
    t_and_c_df: Optional[pd.DataFrame] = None,
    structure_map: Optional[pd.DataFrame] = None,
    structure_map_df: Optional[pd.DataFrame] = None,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    topology_df = t_and_c_df.copy() if t_and_c_df is not None else None
    structure_df = structure_map_df.copy() if structure_map_df is not None else None
    if structure_df is None and structure_map is not None and isinstance(structure_map, pd.DataFrame):
        structure_df = structure_map.copy()

    if tbc_results is not None:
        base = Path(tbc_results)
        if base.is_file():
            if topology_df is None:
                topology_df = _load_matrix_csv(base)
            if structure_df is None:
                sibling = _pick_matching_file(
                    base.parent,
                    ["StructureMap_table*.csv", "*StructureMap*.csv"],
                )
                if sibling is not None:
                    structure_df = _load_matrix_csv(sibling)
        elif base.is_dir():
            if topology_df is None:
                topology_path = _pick_matching_file(
                    base,
                    ["t_and_c_result*.csv", "*t_and_c*.csv"],
                )
                if topology_path is not None:
                    topology_df = _load_matrix_csv(topology_path)
            if structure_df is None:
                structure_path = _pick_matching_file(
                    base,
                    ["StructureMap_table*.csv", "*StructureMap*.csv"],
                )
                if structure_path is not None:
                    structure_df = _load_matrix_csv(structure_path)

    if topology_df is not None:
        topology_df.index = topology_df.index.astype(str)
        topology_df.columns = topology_df.columns.astype(str)
        topology_df = topology_df.apply(pd.to_numeric, errors="coerce")
    if structure_df is not None:
        structure_df.index = structure_df.index.astype(str)
        structure_df.columns = structure_df.columns.astype(str)
        structure_df = structure_df.apply(pd.to_numeric, errors="coerce")

    return topology_df, structure_df


def _recompute_gene_topology(
    reference_df: pd.DataFrame,
    expression_df: pd.DataFrame,
    genes: Sequence[str],
    *,
    entity_points_df: Optional[pd.DataFrame] = None,
    cell_id_col: str,
    x_col: str,
    y_col: str,
    entity_min_weight: float,
    topology_method: str,
) -> pd.DataFrame:
    if not genes:
        return pd.DataFrame(columns=reference_df["celltype"].astype(str).drop_duplicates().tolist())
    present = [gene for gene in genes if gene in expression_df.columns]
    if entity_points_df is not None:
        entity_points = entity_points_df.loc[entity_points_df["entity"].astype(str).isin(genes)].copy()
    else:
        if not present:
            return pd.DataFrame(columns=reference_df["celltype"].astype(str).drop_duplicates().tolist())
        entity_points = build_entity_points_from_expression(
            reference_df,
            expression_df,
            entities=present,
            cell_id_col=cell_id_col,
            x_col=x_col,
            y_col=y_col,
            min_weight=entity_min_weight,
        )
    if entity_points.empty:
        return pd.DataFrame(index=present, columns=reference_df["celltype"].astype(str).drop_duplicates().tolist())
    return compute_entity_to_cell_topology(
        reference_df,
        entity_points,
        x_col=x_col,
        y_col=y_col,
        celltype_col="celltype",
        entity_col="entity",
        weight_col="weight",
        min_weight=entity_min_weight,
        method=topology_method,
    )


def _resolve_gene_topology_anchors(
    reference_df: pd.DataFrame,
    expression_df: pd.DataFrame,
    genes: Sequence[str],
    *,
    tbc_results: Optional[str | os.PathLike[str]] = None,
    t_and_c_df: Optional[pd.DataFrame] = None,
    structure_map: Optional[pd.DataFrame] = None,
    structure_map_df: Optional[pd.DataFrame] = None,
    anchor_mode: str = "precomputed",
    entity_points_df: Optional[pd.DataFrame] = None,
    cell_id_col: str = "cell_id",
    x_col: str = "x",
    y_col: str = "y",
    entity_min_weight: float = 0.0,
    topology_method: str = "average",
) -> tuple[pd.DataFrame, dict[str, str], Optional[pd.DataFrame], str]:
    if anchor_mode not in {"precomputed", "recompute", "hybrid"}:
        raise ValueError("anchor_mode must be one of: precomputed, recompute, hybrid")

    celltypes = list(dict.fromkeys(reference_df["celltype"].astype(str).tolist()))
    genes = list(dict.fromkeys(str(gene) for gene in genes))
    precomputed_topology, precomputed_structure = _resolve_precomputed_tables(
        tbc_results=tbc_results,
        t_and_c_df=t_and_c_df,
        structure_map=structure_map,
        structure_map_df=structure_map_df,
    )

    topology_parts: list[pd.DataFrame] = []
    source_by_gene: dict[str, str] = {}

    use_precomputed = anchor_mode in {"precomputed", "hybrid"} and precomputed_topology is not None
    if use_precomputed:
        available = [gene for gene in genes if gene in precomputed_topology.index]
        if available:
            topology_parts.append(precomputed_topology.reindex(available).reindex(columns=celltypes))
            source_by_gene.update({gene: "precomputed" for gene in available})

    needs_recompute = []
    if anchor_mode == "recompute" or precomputed_topology is None:
        needs_recompute = genes
    else:
        needs_recompute = [gene for gene in genes if gene not in source_by_gene]

    if needs_recompute:
        recomputed = _recompute_gene_topology(
            reference_df,
            expression_df,
            needs_recompute,
            entity_points_df=entity_points_df,
            cell_id_col=cell_id_col,
            x_col=x_col,
            y_col=y_col,
            entity_min_weight=entity_min_weight,
            topology_method=topology_method,
        )
        if not recomputed.empty:
            topology_parts.append(recomputed.reindex(columns=celltypes))
        source_by_gene.update({gene: "recompute" for gene in needs_recompute})

    if topology_parts:
        topology = pd.concat(topology_parts, axis=0)
        topology = topology[~topology.index.duplicated(keep="first")]
        topology = topology.reindex(genes).reindex(columns=celltypes)
    else:
        topology = pd.DataFrame(index=genes, columns=celltypes, dtype=float)

    if precomputed_structure is not None and anchor_mode in {"precomputed", "hybrid"}:
        structure_source = "precomputed"
        resolved_structure = precomputed_structure.reindex(index=celltypes, columns=celltypes)
    elif structure_map is not None and isinstance(structure_map, pd.DataFrame):
        structure_source = "provided"
        resolved_structure = structure_map.copy().reindex(index=celltypes, columns=celltypes)
    else:
        structure_source = "recompute"
        resolved_structure, _ = compute_cophenetic_distances_from_df(
            reference_df,
            x_col=x_col,
            y_col=y_col,
            celltype_col="celltype",
            method=topology_method,
        )
        resolved_structure = resolved_structure.reindex(index=celltypes, columns=celltypes)
    resolved_structure = resolved_structure.apply(pd.to_numeric, errors="coerce")
    resolved_structure = resolved_structure.fillna(1.0)
    for celltype in celltypes:
        if celltype in resolved_structure.index and celltype in resolved_structure.columns:
            resolved_structure.loc[celltype, celltype] = 0.0

    topology.index.name = "gene"
    topology.columns.name = "celltype"
    return topology, source_by_gene, resolved_structure, structure_source


def build_entity_points_from_expression(
    reference_df: pd.DataFrame,
    expression_df: pd.DataFrame,
    *,
    entities: Optional[Iterable[str]] = None,
    cell_id_col: str = "cell_id",
    x_col: str = "x",
    y_col: str = "y",
    min_weight: float = 0.0,
    entity_col: str = "entity",
    weight_col: str = "weight",
) -> pd.DataFrame:
    entities = list(entities) if entities is not None else list(expression_df.columns)
    records: list[pd.DataFrame] = []
    aligned_expr = expression_df.reindex(reference_df[cell_id_col]).fillna(0.0)
    aligned_expr.index = aligned_expr.index.astype(str)

    for entity in entities:
        if entity not in aligned_expr.columns:
            continue
        weights = _coerce_nonnegative(aligned_expr[entity])
        keep = weights > float(min_weight)
        if not keep.any():
            continue
        entity_points = reference_df.loc[keep.to_numpy(), [cell_id_col, x_col, y_col, "celltype"]].copy()
        entity_points[entity_col] = str(entity)
        entity_points[weight_col] = weights.loc[keep].to_numpy()
        records.append(entity_points)

    if not records:
        return pd.DataFrame(columns=[cell_id_col, x_col, y_col, "celltype", entity_col, weight_col])
    return pd.concat(records, ignore_index=True)


def compute_weighted_searcher_findee_distance_matrix_from_df(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    z_col: Optional[str] = None,
    group_col: str = "celltype",
    weight_col: Optional[str] = "weight",
    min_weight: float = 0.0,
) -> pd.DataFrame:
    """
    Compute a weighted directed searcher→findee average nearest-neighbor matrix.

    The weighting scheme is intentionally conservative to preserve backward
    compatibility with the original ``t_and_c`` logic: the nearest-neighbor
    geometry is unchanged, while the row-wise aggregation becomes a weighted
    average over source/searcher points. When every point has unit weight, the
    result is exactly equivalent to ``compute_searcher_findee_distance_matrix_from_df``.
    """

    required = {x_col, y_col, group_col}
    if z_col is not None:
        required.add(z_col)
    if weight_col is not None:
        required.add(weight_col)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {sorted(missing)}")

    work = df.copy()
    if weight_col is None:
        work["__weight"] = 1.0
        weight_col = "__weight"

    work[group_col] = work[group_col].astype("category").cat.remove_unused_categories()
    work[weight_col] = _coerce_nonnegative(work[weight_col])
    work = work.loc[work[weight_col] > float(min_weight)].copy()
    if work.empty:
        raise ValueError("No weighted points remain after filtering; cannot compute weighted distances.")

    coord_cols = [x_col, y_col] + ([z_col] if z_col is not None else [])
    coords = work[coord_cols].to_numpy(dtype=float)
    groups = work[group_col].astype("category").cat.remove_unused_categories()
    unique_groups = list(groups.cat.categories)

    df_nearest = pd.DataFrame(index=work.index, columns=unique_groups, dtype=float)
    for target in unique_groups:
        target_mask = (groups == target).to_numpy()
        coords_target = coords[target_mask]
        if coords_target.shape[0] == 0:
            df_nearest.loc[:, target] = np.nan
            continue
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nbrs.fit(coords_target)
        distances, _ = nbrs.kneighbors(coords)
        df_nearest[target] = distances[:, 0]

    weights = work[weight_col].to_numpy(dtype=float)
    result_rows: list[pd.Series] = []
    for group in unique_groups:
        source_mask = (groups == group).to_numpy()
        source_values = df_nearest.loc[source_mask, unique_groups]
        source_weights = weights[source_mask]
        row = {
            target: _weighted_average(source_values[target].to_numpy(dtype=float), source_weights)
            for target in unique_groups
        }
        result_rows.append(pd.Series(row, name=str(group)))

    return pd.DataFrame(result_rows, index=unique_groups, columns=unique_groups)


def compute_weighted_cophenetic_distances_from_df(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    z_col: Optional[str] = None,
    group_col: str = "celltype",
    weight_col: Optional[str] = "weight",
    min_weight: float = 0.0,
    method: str = "average",
    show_corr: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_mean = compute_weighted_searcher_findee_distance_matrix_from_df(
        df=df,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        group_col=group_col,
        weight_col=weight_col,
        min_weight=min_weight,
    )
    return compute_cophenetic_from_distance_matrix(group_mean, method=method, show_corr=show_corr)


def compute_entity_to_cell_topology(
    reference_df: pd.DataFrame,
    entity_points_df: pd.DataFrame,
    *,
    x_col: str = "x",
    y_col: str = "y",
    z_col: Optional[str] = None,
    celltype_col: str = "celltype",
    entity_col: str = "entity",
    weight_col: str = "weight",
    min_weight: float = 0.0,
    method: str = "average",
) -> pd.DataFrame:
    """
    Generalize transcript-by-cell topology to arbitrary weighted entities.

    ``reference_df`` contains the fixed cell-type template. ``entity_points_df``
    contains an entity label plus spatial points and weights. For every entity we
    temporarily append its weighted point cloud to the reference template, compute
    a weighted StructureMap, and extract the entity→celltype row.
    """

    reference = reference_df.copy()
    if celltype_col not in reference.columns:
        raise ValueError(f"reference_df must contain {celltype_col!r}.")
    required_entity = {x_col, y_col, entity_col, weight_col}
    if z_col is not None:
        required_entity.add(z_col)
    missing_entity = required_entity.difference(entity_points_df.columns)
    if missing_entity:
        raise ValueError(f"entity_points_df is missing required columns: {sorted(missing_entity)}")

    reference = reference.rename(columns={celltype_col: "__group"})
    reference["__weight"] = 1.0
    reference_cols = [x_col, y_col] + ([z_col] if z_col is not None else []) + ["__group", "__weight"]
    reference = reference.loc[:, reference_cols]
    celltypes = reference["__group"].astype(str).tolist()
    unique_celltypes = list(dict.fromkeys(celltypes))

    entity_points = entity_points_df.copy()
    entity_points[entity_col] = entity_points[entity_col].astype(str)
    entity_points[weight_col] = _coerce_nonnegative(entity_points[weight_col])
    unique_entities = list(dict.fromkeys(entity_points[entity_col].tolist()))

    rows: list[pd.Series] = []
    for entity in unique_entities:
        sub = entity_points.loc[
            (entity_points[entity_col] == entity) & (entity_points[weight_col] > float(min_weight))
        ].copy()
        if sub.empty:
            rows.append(pd.Series(np.nan, index=unique_celltypes, name=entity))
            continue

        sub = sub.rename(columns={entity_col: "__group", weight_col: "__weight"})
        sub["__group"] = entity
        sub = sub.loc[:, reference_cols]
        combined = pd.concat([reference, sub], ignore_index=True)
        row_coph, _ = compute_weighted_cophenetic_distances_from_df(
            combined,
            x_col=x_col,
            y_col=y_col,
            z_col=z_col,
            group_col="__group",
            weight_col="__weight",
            min_weight=min_weight,
            method=method,
        )
        row = row_coph.loc[entity].reindex(unique_celltypes)
        rows.append(pd.Series(row, name=entity))

    if not rows:
        return pd.DataFrame(columns=unique_celltypes)
    out = pd.DataFrame(rows)
    out.index.name = entity_col
    out.columns.name = celltype_col
    return out


def compute_entity_structuremap(
    entity_points_df: pd.DataFrame,
    *,
    x_col: str = "x",
    y_col: str = "y",
    z_col: Optional[str] = None,
    entity_col: str = "entity",
    weight_col: str = "weight",
    min_weight: float = 0.0,
    method: str = "average",
) -> pd.DataFrame:
    if entity_points_df.empty:
        return pd.DataFrame()
    work = entity_points_df.copy()
    work[entity_col] = work[entity_col].astype(str)
    work[weight_col] = _coerce_nonnegative(work[weight_col])
    row_coph, _ = compute_weighted_cophenetic_distances_from_df(
        work,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        group_col=entity_col,
        weight_col=weight_col,
        min_weight=min_weight,
        method=method,
    )
    row_coph.index.name = entity_col
    row_coph.columns.name = entity_col
    return row_coph


def _build_neighbor_index(
    reference_df: pd.DataFrame,
    *,
    x_col: str = "x",
    y_col: str = "y",
    k_neighbors: int = 8,
    radius: Optional[float] = None,
) -> list[np.ndarray]:
    coords = reference_df[[x_col, y_col]].to_numpy(dtype=float)
    if len(coords) == 0:
        return []

    if radius is not None:
        model = NearestNeighbors(radius=radius, algorithm="auto")
        model.fit(coords)
        neighbors = model.radius_neighbors(coords, return_distance=False)
        return [arr[arr != idx] for idx, arr in enumerate(neighbors)]

    n_neighbors = min(len(coords), max(2, int(k_neighbors) + 1))
    model = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
    model.fit(coords)
    indices = model.kneighbors(coords, return_distance=False)
    return [arr[arr != idx] for idx, arr in enumerate(indices)]


def _smooth_matrix_by_neighbors(
    matrix: pd.DataFrame,
    neighbor_index: list[np.ndarray],
    *,
    include_self: bool = True,
) -> pd.DataFrame:
    if matrix.empty:
        return matrix.copy()
    values = matrix.to_numpy(dtype=float)
    smoothed = np.zeros_like(values)
    for idx, neighbors in enumerate(neighbor_index):
        if include_self:
            neighbors = np.unique(np.append(neighbors, idx))
        if len(neighbors) == 0:
            smoothed[idx] = values[idx]
        else:
            smoothed[idx] = values[neighbors].mean(axis=0)
    return pd.DataFrame(smoothed, index=matrix.index, columns=matrix.columns)


def _normalize_matrix_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    values = frame.copy().astype(float)
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    spans = (maxs - mins).replace(0.0, np.nan)
    normalized = values.sub(mins, axis=1).div(spans, axis=1).fillna(0.0)
    constant_nonzero = spans.isna() & (maxs > 0)
    for col in values.columns[constant_nonzero]:
        normalized[col] = 1.0
    return normalized


def summarize_expression_by_celltype(
    expression_df: pd.DataFrame,
    celltypes: pd.Series,
    *,
    detection_threshold: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return raw and normalized gene×celltype summaries using
    pseudobulk share × sqrt(detection fraction).
    """

    expr = expression_df.copy()
    expr.index = expr.index.astype(str)
    celltypes = celltypes.astype(str)
    pseudobulk = expr.groupby(celltypes).sum()
    share = pseudobulk.div(pseudobulk.sum(axis=0).replace(0.0, np.nan), axis=1).fillna(0.0)
    detected = (expr > float(detection_threshold)).groupby(celltypes).mean()
    combined = share.mul(np.sqrt(detected))
    raw = combined.T.astype(float)
    raw.index.name = "gene"
    raw.columns.name = "celltype"
    normalized = _normalize_frame_rows(raw)
    normalized.index.name = "gene"
    normalized.columns.name = "celltype"
    return raw, normalized


def _normalize_lr_prior(lr_pairs: pd.DataFrame, prior_col: Optional[str]) -> pd.Series:
    if prior_col is None or prior_col not in lr_pairs.columns:
        return pd.Series(np.ones(len(lr_pairs)), index=lr_pairs.index, name="prior_confidence")
    prior = pd.to_numeric(lr_pairs[prior_col], errors="coerce").fillna(0.0).astype(float)
    return _normalize_series(prior).rename("prior_confidence")


def _compute_local_contact_matrix(
    reference_df: pd.DataFrame,
    ligand_values: pd.Series,
    receptor_values: pd.Series,
    neighbor_index: list[np.ndarray],
    *,
    celltype_col: str = "celltype",
    min_cross_edges: int = 50,
    contact_expr_threshold: str | float = "q75_nonzero",
    winsor_quantile: float = 0.99,
) -> dict[str, pd.DataFrame]:
    celltypes = reference_df[celltype_col].astype(str).tolist()
    unique_celltypes = list(dict.fromkeys(celltypes))
    index_by_type = {celltype: [] for celltype in unique_celltypes}
    for idx, celltype in enumerate(celltypes):
        index_by_type[celltype].append(idx)

    ligand_raw = pd.to_numeric(ligand_values.reindex(reference_df.index).fillna(0.0), errors="coerce").fillna(0.0).astype(float)
    receptor_raw = pd.to_numeric(receptor_values.reindex(reference_df.index).fillna(0.0), errors="coerce").fillna(0.0).astype(float)
    ligand_norm = _winsorized_normalize_series(ligand_raw, upper_quantile=winsor_quantile)
    receptor_norm = _winsorized_normalize_series(receptor_raw, upper_quantile=winsor_quantile)

    def _resolve_threshold(series: pd.Series) -> float:
        if isinstance(contact_expr_threshold, (int, float)):
            return float(contact_expr_threshold)
        if contact_expr_threshold == "q75_nonzero":
            positive = series.loc[series > 0]
            return float(positive.quantile(0.75)) if not positive.empty else float("inf")
        raise ValueError("contact_expr_threshold must be a float or 'q75_nonzero'")

    ligand_threshold = _resolve_threshold(ligand_raw)
    receptor_threshold = _resolve_threshold(receptor_raw)

    strength = pd.DataFrame(0.0, index=unique_celltypes, columns=unique_celltypes, dtype=float)
    coverage = pd.DataFrame(0.0, index=unique_celltypes, columns=unique_celltypes, dtype=float)
    cross_edges = pd.DataFrame(0, index=unique_celltypes, columns=unique_celltypes, dtype=int)
    for sender in unique_celltypes:
        sender_indices = index_by_type[sender]
        for receiver in unique_celltypes:
            edge_scores: list[float] = []
            active_edges = 0
            total_edges = 0
            for idx in sender_indices:
                for nbr in neighbor_index[idx]:
                    if celltypes[nbr] != receiver:
                        continue
                    total_edges += 1
                    edge_scores.append(float(ligand_norm.iloc[idx] * receptor_norm.iloc[nbr]))
                    if ligand_raw.iloc[idx] > ligand_threshold and receptor_raw.iloc[nbr] > receptor_threshold:
                        active_edges += 1
            cross_edges.loc[sender, receiver] = total_edges
            strength.loc[sender, receiver] = float(np.mean(edge_scores)) if edge_scores else 0.0
            coverage.loc[sender, receiver] = (active_edges / total_edges) if total_edges else 0.0

    strength_norm = _winsorized_normalize_frame(strength, upper_quantile=winsor_quantile)
    off_diag_mask = ~np.eye(len(unique_celltypes), dtype=bool)
    off_diag_values = cross_edges.to_numpy(dtype=float)[off_diag_mask]
    off_diag_values = off_diag_values[off_diag_values > 0]
    if off_diag_values.size == 0:
        edge_scale = 1.0
    else:
        edge_scale = float(np.quantile(off_diag_values, 0.99))
        if math.isclose(edge_scale, 0.0):
            edge_scale = 1.0
    edge_support = (cross_edges.astype(float) / edge_scale).clip(lower=0.0, upper=1.0)

    local_contact = pd.DataFrame(0.0, index=unique_celltypes, columns=unique_celltypes, dtype=float)
    enough_edges = cross_edges >= int(min_cross_edges)
    local_contact = (np.sqrt(strength_norm.mul(coverage)) * edge_support).where(enough_edges, other=0.0)

    return {
        "local_contact": local_contact.astype(float),
        "contact_strength_raw": strength.astype(float),
        "contact_strength_normalized": strength_norm.astype(float),
        "contact_coverage": coverage.astype(float),
        "cross_edge_count": cross_edges.astype(int),
        "edge_support": edge_support.astype(float),
        "ligand_threshold": pd.DataFrame(ligand_threshold, index=unique_celltypes, columns=unique_celltypes),
        "receptor_threshold": pd.DataFrame(receptor_threshold, index=unique_celltypes, columns=unique_celltypes),
    }


def _geometric_mean(values: Iterable[float], eps: float = 1e-8) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = np.clip(arr, 0.0, None)
    return float(np.exp(np.mean(np.log(arr + eps))))


def _prepare_hotspot_table(
    reference_df: pd.DataFrame,
    *,
    sender_mask: pd.Series,
    receiver_mask: pd.Series,
    sender_score: pd.Series,
    receiver_score: pd.Series,
    ligand: str,
    receptor: str,
    sender_celltype: str,
    receiver_celltype: str,
    x_col: str,
    y_col: str,
) -> pd.DataFrame:
    sender_tbl = reference_df.loc[sender_mask, ["cell_id", x_col, y_col, "celltype"]].copy()
    sender_tbl["role"] = "sender"
    sender_tbl["feature"] = ligand
    sender_tbl["score"] = sender_score.loc[sender_mask].to_numpy()
    sender_tbl["sender_celltype"] = sender_celltype
    sender_tbl["receiver_celltype"] = receiver_celltype
    sender_tbl["ligand"] = ligand
    sender_tbl["receptor"] = receptor

    receiver_tbl = reference_df.loc[receiver_mask, ["cell_id", x_col, y_col, "celltype"]].copy()
    receiver_tbl["role"] = "receiver"
    receiver_tbl["feature"] = receptor
    receiver_tbl["score"] = receiver_score.loc[receiver_mask].to_numpy()
    receiver_tbl["sender_celltype"] = sender_celltype
    receiver_tbl["receiver_celltype"] = receiver_celltype
    receiver_tbl["ligand"] = ligand
    receiver_tbl["receptor"] = receptor
    return pd.concat([sender_tbl, receiver_tbl], ignore_index=True)


def ligand_receptor_topology_analysis(
    *,
    reference_df: Optional[pd.DataFrame] = None,
    expression_df: Optional[pd.DataFrame] = None,
    lr_pairs: pd.DataFrame,
    output_dir: Optional[str | os.PathLike[str]] = None,
    adata: Any = None,
    entity_points_df: Optional[pd.DataFrame] = None,
    tbc_results: Optional[str | os.PathLike[str]] = None,
    t_and_c_df: Optional[pd.DataFrame] = None,
    cluster_col: str = "Cluster",
    cell_id_col: str = "cell_id",
    x_col: str = "x",
    y_col: str = "y",
    celltype_col: str = "celltype",
    ligand_col: str = "ligand",
    receptor_col: str = "receptor",
    prior_col: str = "evidence_weight",
    structure_map: Optional[pd.DataFrame] = None,
    structure_map_df: Optional[pd.DataFrame] = None,
    anchor_mode: str = "precomputed",
    expression_support_mode: str = "pseudobulk_detection",
    contact_mode: str = "strength_coverage",
    entity_min_weight: float = 0.0,
    detection_threshold: float = 0.0,
    k_neighbors: int = 8,
    radius: Optional[float] = None,
    topology_method: str = "average",
    top_n_pairs: int = 12,
    hotspot_quantile: float = 0.9,
    min_cross_edges: int = 50,
    contact_expr_threshold: str | float = "q75_nonzero",
    use_raw: bool = False,
) -> dict[str, Any]:
    if expression_support_mode != "pseudobulk_detection":
        raise ValueError("expression_support_mode currently supports only 'pseudobulk_detection'.")
    if contact_mode != "strength_coverage":
        raise ValueError("contact_mode currently supports only 'strength_coverage'.")

    reference = _coerce_reference_df(
        reference_df,
        adata=adata,
        cluster_col=cluster_col,
        cell_id_col=cell_id_col,
        x_col=x_col,
        y_col=y_col,
        celltype_col=celltype_col,
    )
    reference.index = reference[cell_id_col].astype(str)

    if ligand_col not in lr_pairs.columns or receptor_col not in lr_pairs.columns:
        raise ValueError(f"lr_pairs must contain {ligand_col!r} and {receptor_col!r}.")
    lr_pairs = lr_pairs.copy()
    lr_pairs[ligand_col] = lr_pairs[ligand_col].astype(str)
    lr_pairs[receptor_col] = lr_pairs[receptor_col].astype(str)
    lr_pairs["prior_confidence"] = _normalize_lr_prior(lr_pairs, prior_col)

    genes = list(dict.fromkeys(lr_pairs[ligand_col].tolist() + lr_pairs[receptor_col].tolist()))
    expression = _coerce_expression_df(
        reference,
        expression_df,
        adata=adata,
        genes=genes,
        cell_id_col=cell_id_col,
        use_raw=use_raw,
    )
    expression.index = reference.index

    topology, anchor_sources, resolved_structure_map, structure_map_source = _resolve_gene_topology_anchors(
        reference,
        expression,
        genes,
        tbc_results=tbc_results,
        t_and_c_df=t_and_c_df,
        structure_map=structure_map,
        structure_map_df=structure_map_df,
        anchor_mode=anchor_mode,
        entity_points_df=entity_points_df,
        cell_id_col=cell_id_col,
        x_col=x_col,
        y_col=y_col,
        entity_min_weight=entity_min_weight,
        topology_method=topology_method,
    )
    ligand_to_cell = topology.reindex(lr_pairs[ligand_col].drop_duplicates())
    receptor_to_cell = topology.reindex(lr_pairs[receptor_col].drop_duplicates())
    structure_map = resolved_structure_map

    _, expression_summary = summarize_expression_by_celltype(
        expression,
        reference["celltype"],
        detection_threshold=detection_threshold,
    )
    neighbor_index = _build_neighbor_index(reference, x_col=x_col, y_col=y_col, k_neighbors=k_neighbors, radius=radius)
    celltypes = list(dict.fromkeys(reference["celltype"].astype(str).tolist()))
    score_rows: list[dict[str, Any]] = []

    for row in lr_pairs.itertuples(index=False):
        ligand = getattr(row, ligand_col)
        receptor = getattr(row, receptor_col)
        prior = float(getattr(row, "prior_confidence"))
        if ligand not in topology.index or receptor not in topology.index:
            continue
        sender_anchor = (1.0 - topology.loc[ligand].reindex(celltypes).fillna(1.0)).clip(lower=0.0, upper=1.0)
        receiver_anchor = (1.0 - topology.loc[receptor].reindex(celltypes).fillna(1.0)).clip(lower=0.0, upper=1.0)
        sender_expr = (
            expression_summary.loc[ligand].reindex(celltypes).fillna(0.0)
            if ligand in expression_summary.index
            else pd.Series(0.0, index=celltypes)
        )
        receiver_expr = (
            expression_summary.loc[receptor].reindex(celltypes).fillna(0.0)
            if receptor in expression_summary.index
            else pd.Series(0.0, index=celltypes)
        )
        local_contact_parts = _compute_local_contact_matrix(
            reference,
            expression[ligand] if ligand in expression.columns else pd.Series(0.0, index=reference.index),
            expression[receptor] if receptor in expression.columns else pd.Series(0.0, index=reference.index),
            neighbor_index,
            celltype_col="celltype",
            min_cross_edges=min_cross_edges,
            contact_expr_threshold=contact_expr_threshold,
        )
        local_contact = local_contact_parts["local_contact"].reindex(index=celltypes, columns=celltypes, fill_value=0.0)
        contact_strength_raw = local_contact_parts["contact_strength_raw"].reindex(
            index=celltypes,
            columns=celltypes,
            fill_value=0.0,
        )
        contact_coverage = local_contact_parts["contact_coverage"].reindex(
            index=celltypes,
            columns=celltypes,
            fill_value=0.0,
        )
        contact_strength_normalized = local_contact_parts["contact_strength_normalized"].reindex(
            index=celltypes,
            columns=celltypes,
            fill_value=0.0,
        )
        cross_edge_count = local_contact_parts["cross_edge_count"].reindex(
            index=celltypes,
            columns=celltypes,
            fill_value=0,
        )

        for sender in celltypes:
            for receiver in celltypes:
                bridge = 1.0 - float(structure_map.loc[sender, receiver])
                score = _geometric_mean(
                    [
                        float(sender_anchor.loc[sender]),
                        float(receiver_anchor.loc[receiver]),
                        bridge,
                        float(sender_expr.loc[sender]),
                        float(receiver_expr.loc[receiver]),
                        float(local_contact.loc[sender, receiver]),
                    ]
                ) * prior
                score_rows.append(
                    {
                        "ligand": ligand,
                        "receptor": receptor,
                        "sender_celltype": sender,
                        "receiver_celltype": receiver,
                        "anchor_source_ligand": anchor_sources.get(ligand, "recompute"),
                        "anchor_source_receptor": anchor_sources.get(receptor, "recompute"),
                        "structure_map_source": structure_map_source,
                        "sender_anchor": float(sender_anchor.loc[sender]),
                        "receiver_anchor": float(receiver_anchor.loc[receiver]),
                        "structure_bridge": bridge,
                        "sender_expr": float(sender_expr.loc[sender]),
                        "receiver_expr": float(receiver_expr.loc[receiver]),
                        "local_contact": float(local_contact.loc[sender, receiver]),
                        "contact_strength_raw": float(contact_strength_raw.loc[sender, receiver]),
                        "contact_strength_normalized": float(contact_strength_normalized.loc[sender, receiver]),
                        "contact_coverage": float(contact_coverage.loc[sender, receiver]),
                        "cross_edge_count": int(cross_edge_count.loc[sender, receiver]),
                        "prior_confidence": prior,
                        "LR_score": float(score),
                    }
                )

    scores = pd.DataFrame(score_rows)
    if not scores.empty:
        scores = scores.sort_values("LR_score", ascending=False).reset_index(drop=True)
    component_diagnostics = scores.copy()
    out_dir = _ensure_output_dir(output_dir)
    output_files: dict[str, Any] = {}
    if out_dir is not None:
        ligand_path = out_dir / "ligand_to_cell.csv"
        receptor_path = out_dir / "receptor_to_cell.csv"
        scores_path = out_dir / "lr_sender_receiver_scores.csv"
        diagnostics_path = out_dir / "lr_component_diagnostics.csv"
        ligand_to_cell.to_csv(ligand_path)
        receptor_to_cell.to_csv(receptor_path)
        scores.to_csv(scores_path, index=False)
        component_diagnostics.to_csv(diagnostics_path, index=False)
        output_files["ligand_to_cell"] = str(ligand_path)
        output_files["receptor_to_cell"] = str(receptor_path)
        output_files["lr_sender_receiver_scores"] = str(scores_path)
        output_files["lr_component_diagnostics"] = str(diagnostics_path)

        summary = scores.copy()
        summary["lr_pair"] = summary["ligand"] + "→" + summary["receptor"]
        summary["sender_receiver"] = summary["sender_celltype"] + "→" + summary["receiver_celltype"]
        top_pairs = summary.groupby("lr_pair")["LR_score"].max().sort_values(ascending=False).head(int(top_n_pairs)).index.tolist()
        summary_matrix = summary.loc[summary["lr_pair"].isin(top_pairs)].pivot_table(
            index="lr_pair",
            columns="sender_receiver",
            values="LR_score",
            aggfunc="max",
            fill_value=0.0,
        )
        if not summary_matrix.empty:
            output_files["lr_summary_heatmap"] = _save_heatmap(
                summary_matrix,
                title="Ligand-receptor topology summary",
                output_prefix=out_dir / "lr_summary_heatmap",
                cmap="rocket_r",
            )

        if not scores.empty:
            best = scores.iloc[0]
            ligand = str(best["ligand"])
            receptor = str(best["receptor"])
            sender = str(best["sender_celltype"])
            receiver = str(best["receiver_celltype"])
            ligand_cell = (
                _winsorized_normalize_series(expression[ligand]) if ligand in expression.columns else pd.Series(0.0, index=reference.index)
            )
            receptor_cell = (
                _winsorized_normalize_series(expression[receptor]) if receptor in expression.columns else pd.Series(0.0, index=reference.index)
            )
            sender_mask = (reference["celltype"] == sender) & (ligand_cell >= ligand_cell.quantile(float(hotspot_quantile)))
            receiver_mask = (reference["celltype"] == receiver) & (receptor_cell >= receptor_cell.quantile(float(hotspot_quantile)))
            hotspot_df = _prepare_hotspot_table(
                reference,
                sender_mask=sender_mask,
                receiver_mask=receiver_mask,
                sender_score=ligand_cell,
                receiver_score=receptor_cell,
                ligand=ligand,
                receptor=receptor,
                sender_celltype=sender,
                receiver_celltype=receiver,
                x_col=x_col,
                y_col=y_col,
            )
            hotspot_csv = out_dir / "lr_hotspot_cells.csv"
            hotspot_df.to_csv(hotspot_csv, index=False)
            output_files["lr_hotspot_cells_csv"] = str(hotspot_csv)
            parquet_path = out_dir / "lr_hotspot_cells.parquet"
            if _safe_to_parquet(hotspot_df, parquet_path):
                output_files["lr_hotspot_cells_parquet"] = str(parquet_path)
            output_files["lr_hotspot_overlay"] = _save_hotspot_overlay(
                reference,
                x_col=x_col,
                y_col=y_col,
                sender_mask=sender_mask,
                receiver_mask=receiver_mask,
                sender_score=ligand_cell,
                receiver_score=receptor_cell,
                title=f"{ligand}→{receptor} hotspot ({sender}→{receiver})",
                output_prefix=out_dir / "lr_hotspot_overlay",
            )

    return {
        "ligand_to_cell": ligand_to_cell,
        "receptor_to_cell": receptor_to_cell,
        "structure_map": structure_map,
        "scores": scores,
        "component_diagnostics": component_diagnostics,
        "anchor_sources": anchor_sources,
        "files": output_files,
    }


def _standardize_pathway_definitions(pathway_definitions: Mapping[str, Any] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(pathway_definitions, pd.DataFrame):
        required = {"pathway", "gene"}
        missing = required.difference(pathway_definitions.columns)
        if missing:
            raise ValueError(f"pathway_definitions DataFrame is missing required columns: {sorted(missing)}")
        out = pathway_definitions.copy()
        if "weight" not in out.columns:
            out["weight"] = 1.0
        out["pathway"] = out["pathway"].astype(str)
        out["gene"] = out["gene"].astype(str)
        out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(1.0).astype(float)
        return out[["pathway", "gene", "weight"]]

    rows: list[dict[str, Any]] = []
    for pathway, genes in pathway_definitions.items():
        if isinstance(genes, Mapping):
            for gene, weight in genes.items():
                rows.append({"pathway": str(pathway), "gene": str(gene), "weight": float(weight)})
        else:
            for gene in genes:
                rows.append({"pathway": str(pathway), "gene": str(gene), "weight": 1.0})
    return pd.DataFrame(rows, columns=["pathway", "gene", "weight"])


def compute_pathway_activity_matrix(
    expression_df: pd.DataFrame,
    pathway_definitions: Mapping[str, Any] | pd.DataFrame,
    *,
    method: str = "rank_mean",
    normalize: bool = True,
) -> pd.DataFrame:
    pathway_table = _standardize_pathway_definitions(pathway_definitions)
    expr = expression_df.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    activity = pd.DataFrame(index=expr.index)

    if method not in {"rank_mean", "ucell", "aucell", "weighted_sum", "progeny"}:
        raise ValueError("method must be one of: rank_mean, ucell, aucell, weighted_sum, progeny")

    if method in {"rank_mean", "ucell", "aucell"}:
        ranked = expr.rank(axis=1, method="average", ascending=True, pct=True)
        for pathway, group in pathway_table.groupby("pathway", sort=False):
            present = [gene for gene in group["gene"] if gene in ranked.columns]
            if not present:
                activity[pathway] = 0.0
                continue
            weights = group.set_index("gene").loc[present, "weight"].astype(float)
            values = ranked[present].to_numpy(dtype=float)
            activity[pathway] = np.average(values, axis=1, weights=np.abs(weights.to_numpy(dtype=float)))
    else:
        for pathway, group in pathway_table.groupby("pathway", sort=False):
            present = [gene for gene in group["gene"] if gene in expr.columns]
            if not present:
                activity[pathway] = 0.0
                continue
            weights = group.set_index("gene").loc[present, "weight"].astype(float)
            denom = float(np.sum(np.abs(weights.to_numpy(dtype=float)))) or 1.0
            activity[pathway] = expr[present].to_numpy(dtype=float) @ weights.to_numpy(dtype=float) / denom

    if normalize:
        activity = _normalize_matrix_columns(activity)
    return activity


def _aggregate_pathway_gene_topology(
    gene_topology: pd.DataFrame,
    pathway_table: pd.DataFrame,
    *,
    aggregate: str = "weighted_median",
) -> pd.DataFrame:
    celltypes = gene_topology.columns.astype(str).tolist()
    rows: list[pd.Series] = []
    for pathway, group in pathway_table.groupby("pathway", sort=False):
        pathway_genes = [gene for gene in group["gene"].astype(str) if gene in gene_topology.index]
        weights = group.set_index("gene").reindex(pathway_genes)["weight"].fillna(1.0).astype(float)
        if not pathway_genes:
            rows.append(pd.Series(np.nan, index=celltypes, name=str(pathway)))
            continue
        aggregated = {
            celltype: _aggregate_weighted_values(
                gene_topology.loc[pathway_genes, celltype].to_numpy(dtype=float),
                weights.to_numpy(dtype=float),
                method=aggregate,
            )
            for celltype in celltypes
        }
        rows.append(pd.Series(aggregated, name=str(pathway)))
    out = pd.DataFrame(rows, columns=celltypes)
    out.index.name = "pathway"
    out.columns.name = "celltype"
    return out


def _build_pathway_activity_points(
    reference_df: pd.DataFrame,
    pathway_activity: pd.DataFrame,
    *,
    cell_id_col: str,
    x_col: str,
    y_col: str,
    activity_mode: str,
    activity_threshold_schedule: Sequence[float],
    min_activity_cells: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []

    for pathway in pathway_activity.columns:
        values = pd.to_numeric(pathway_activity[pathway], errors="coerce").fillna(0.0).astype(float)
        retained_quantile: Optional[float] = None
        retained_mask = pd.Series(False, index=values.index)
        positive_values = values.loc[values > 0]
        quantile_pool = positive_values if not positive_values.empty else values
        for q in activity_threshold_schedule:
            threshold = float(quantile_pool.quantile(float(q)))
            if positive_values.empty:
                mask = values >= threshold
            else:
                mask = (values >= threshold) & (values > 0)
            if int(mask.sum()) >= int(min_activity_cells):
                retained_mask = mask
                retained_quantile = float(q)
                break
        if retained_quantile is None:
            top_n = min(int(min_activity_cells), len(values))
            if top_n > 0:
                top_idx = values.nlargest(top_n).index
                retained_mask.loc[top_idx] = True
            retained_quantile = float(activity_threshold_schedule[-1]) if activity_threshold_schedule else 0.5

        retained_values = values.loc[retained_mask]
        if not retained_values.empty:
            points = reference_df.loc[retained_mask, [cell_id_col, x_col, y_col, "celltype"]].copy()
            points["entity"] = str(pathway)
            points["weight"] = retained_values.to_numpy(dtype=float)
            records.append(points)

        diagnostics.append(
            {
                "pathway": str(pathway),
                "retained_cell_count": int(retained_mask.sum()),
                "retained_quantile": float(retained_quantile),
                "activity_mode": str(activity_mode),
            }
        )

    if records:
        entity_points = pd.concat(records, ignore_index=True)
    else:
        entity_points = pd.DataFrame(columns=[cell_id_col, x_col, y_col, "celltype", "entity", "weight"])
    diagnostics_df = pd.DataFrame(diagnostics)
    return entity_points, diagnostics_df


def _pathway_mode_summary(
    pathway_to_cell: pd.DataFrame,
    *,
    mode_name: str,
) -> pd.DataFrame:
    if pathway_to_cell.empty:
        return pd.DataFrame(columns=["pathway", f"{mode_name}_best_celltype", f"{mode_name}_best_distance"])
    best = pd.DataFrame(
        {
            "pathway": pathway_to_cell.index.astype(str),
            f"{mode_name}_best_celltype": pathway_to_cell.idxmin(axis=1).astype(str).to_numpy(),
            f"{mode_name}_best_distance": pathway_to_cell.min(axis=1).to_numpy(dtype=float),
        }
    )
    return best


def pathway_topology_analysis(
    *,
    pathway_definitions: Mapping[str, Any] | pd.DataFrame,
    reference_df: Optional[pd.DataFrame] = None,
    expression_df: Optional[pd.DataFrame] = None,
    output_dir: Optional[str | os.PathLike[str]] = None,
    adata: Any = None,
    tbc_results: Optional[str | os.PathLike[str]] = None,
    t_and_c_df: Optional[pd.DataFrame] = None,
    cluster_col: str = "Cluster",
    cell_id_col: str = "cell_id",
    x_col: str = "x",
    y_col: str = "y",
    celltype_col: str = "celltype",
    scoring_method: str = "weighted_sum",
    view: str = "intrinsic",
    structure_map: Optional[pd.DataFrame] = None,
    structure_map_df: Optional[pd.DataFrame] = None,
    anchor_mode: str = "precomputed",
    pathway_modes: Sequence[str] = ("gene_topology_aggregate", "activity_point_cloud"),
    primary_pathway_mode: str = "gene_topology_aggregate",
    pathway_aggregate: str = "weighted_median",
    activity_threshold_schedule: Sequence[float] = (0.95, 0.90, 0.80, 0.70, 0.60, 0.50),
    min_activity_cells: int = 50,
    entity_min_weight: float = 0.0,
    k_neighbors: int = 8,
    radius: Optional[float] = None,
    topology_method: str = "average",
    hotspot_quantile: float = 0.9,
    use_raw: bool = False,
) -> dict[str, Any]:
    if view not in {"intrinsic", "niche_smoothed"}:
        raise ValueError("view must be either 'intrinsic' or 'niche_smoothed'.")
    valid_modes = {"gene_topology_aggregate", "activity_point_cloud"}
    if any(mode not in valid_modes for mode in pathway_modes):
        raise ValueError("pathway_modes must contain only 'gene_topology_aggregate' and/or 'activity_point_cloud'.")
    if primary_pathway_mode not in valid_modes:
        raise ValueError("primary_pathway_mode must be 'gene_topology_aggregate' or 'activity_point_cloud'.")
    if primary_pathway_mode not in set(pathway_modes):
        raise ValueError("primary_pathway_mode must also be present in pathway_modes.")

    pathway_table = _standardize_pathway_definitions(pathway_definitions)
    genes = pathway_table["gene"].drop_duplicates().tolist()
    reference = _coerce_reference_df(
        reference_df,
        adata=adata,
        cluster_col=cluster_col,
        cell_id_col=cell_id_col,
        x_col=x_col,
        y_col=y_col,
        celltype_col=celltype_col,
    )
    reference.index = reference[cell_id_col].astype(str)
    expression = _coerce_expression_df(
        reference,
        expression_df,
        adata=adata,
        genes=genes,
        cell_id_col=cell_id_col,
        use_raw=use_raw,
    )
    expression.index = reference.index

    pathway_activity = compute_pathway_activity_matrix(expression, pathway_table, method=scoring_method, normalize=False)
    pathway_activity = _robust_scale_columns(pathway_activity)
    if view == "niche_smoothed":
        neighbor_index = _build_neighbor_index(reference, x_col=x_col, y_col=y_col, k_neighbors=k_neighbors, radius=radius)
        pathway_activity = _robust_scale_columns(_smooth_matrix_by_neighbors(pathway_activity, neighbor_index, include_self=True))

    gene_topology, anchor_sources, resolved_structure_map, structure_map_source = _resolve_gene_topology_anchors(
        reference,
        expression,
        genes,
        tbc_results=tbc_results,
        t_and_c_df=t_and_c_df,
        structure_map=structure_map,
        structure_map_df=structure_map_df,
        anchor_mode=anchor_mode,
        cell_id_col=cell_id_col,
        x_col=x_col,
        y_col=y_col,
        entity_min_weight=entity_min_weight,
        topology_method=topology_method,
    )

    gene_topology_aggregate = _aggregate_pathway_gene_topology(
        gene_topology,
        pathway_table,
        aggregate=pathway_aggregate,
    )
    gene_topology_structuremap = _safe_row_cophenetic(gene_topology_aggregate, method=topology_method)

    activity_entity_points, activity_diagnostics = _build_pathway_activity_points(
        reference,
        pathway_activity,
        cell_id_col=cell_id_col,
        x_col=x_col,
        y_col=y_col,
        activity_mode=view,
        activity_threshold_schedule=activity_threshold_schedule,
        min_activity_cells=min_activity_cells,
    )
    pathway_activity_to_cell = compute_entity_to_cell_topology(
        reference,
        activity_entity_points,
        x_col=x_col,
        y_col=y_col,
        celltype_col="celltype",
        entity_col="entity",
        weight_col="weight",
        min_weight=entity_min_weight,
        method=topology_method,
    )
    pathway_activity_structuremap = compute_entity_structuremap(
        activity_entity_points,
        x_col=x_col,
        y_col=y_col,
        entity_col="entity",
        weight_col="weight",
        min_weight=entity_min_weight,
        method=topology_method,
    )

    primary_lookup = {
        "gene_topology_aggregate": (gene_topology_aggregate, gene_topology_structuremap),
        "activity_point_cloud": (pathway_activity_to_cell, pathway_activity_structuremap),
    }
    pathway_to_cell, pathway_structuremap = primary_lookup[primary_pathway_mode]

    mode_comparison = (
        _pathway_mode_summary(gene_topology_aggregate, mode_name="aggregate")
        .merge(
            _pathway_mode_summary(pathway_activity_to_cell, mode_name="activity"),
            on="pathway",
            how="outer",
        )
        .merge(activity_diagnostics, on="pathway", how="left")
    )
    mode_comparison["primary_pathway_mode"] = primary_pathway_mode
    mode_comparison["structure_map_source"] = structure_map_source

    out_dir = _ensure_output_dir(output_dir)
    output_files: dict[str, Any] = {}
    if out_dir is not None:
        p_to_c_path = out_dir / "pathway_to_cell.csv"
        p_to_p_path = out_dir / "pathway_structuremap.csv"
        p_activity_to_c_path = out_dir / "pathway_activity_to_cell.csv"
        p_activity_to_p_path = out_dir / "pathway_activity_structuremap.csv"
        comparison_path = out_dir / "pathway_mode_comparison.csv"
        pathway_to_cell.to_csv(p_to_c_path)
        pathway_structuremap.to_csv(p_to_p_path)
        pathway_activity_to_cell.to_csv(p_activity_to_c_path)
        pathway_activity_structuremap.to_csv(p_activity_to_p_path)
        mode_comparison.to_csv(comparison_path, index=False)
        output_files["pathway_to_cell"] = str(p_to_c_path)
        output_files["pathway_structuremap"] = str(p_to_p_path)
        output_files["pathway_activity_to_cell"] = str(p_activity_to_c_path)
        output_files["pathway_activity_structuremap"] = str(p_activity_to_p_path)
        output_files["pathway_mode_comparison"] = str(comparison_path)

        if not pathway_to_cell.empty:
            output_files["pathway_to_cell_heatmap"] = _save_heatmap(
                pathway_to_cell,
                title=f"Pathway-to-cell topology ({primary_pathway_mode})",
                output_prefix=out_dir / "pathway_to_cell_heatmap",
                cmap="viridis_r",
            )

        if (not pathway_activity.empty) and (not pathway_activity_to_cell.empty):
            best_pathway = pathway_activity_to_cell.min(axis=1).sort_values().index[0]
            best_score = pathway_activity[best_pathway]
            threshold = float(best_score.quantile(float(hotspot_quantile)))
            hotspot_mask = best_score >= threshold
            hotspot_df = reference.loc[hotspot_mask, ["cell_id", x_col, y_col, "celltype"]].copy()
            hotspot_df["pathway"] = best_pathway
            hotspot_df["score"] = best_score.loc[hotspot_mask].to_numpy()
            activity_meta = activity_diagnostics.set_index("pathway").reindex([best_pathway])
            if not activity_meta.empty:
                hotspot_df["retained_quantile"] = float(activity_meta["retained_quantile"].iloc[0])
                hotspot_df["retained_cell_count"] = int(activity_meta["retained_cell_count"].iloc[0])
                hotspot_df["activity_mode"] = str(activity_meta["activity_mode"].iloc[0])
            hotspot_csv = out_dir / "pathway_hotspot_cells.csv"
            hotspot_df.to_csv(hotspot_csv, index=False)
            output_files["pathway_hotspot_cells_csv"] = str(hotspot_csv)
            parquet_path = out_dir / "pathway_hotspot_cells.parquet"
            if _safe_to_parquet(hotspot_df, parquet_path):
                output_files["pathway_hotspot_cells_parquet"] = str(parquet_path)

            output_files["pathway_roi_overlay"] = _save_hotspot_overlay(
                reference,
                x_col=x_col,
                y_col=y_col,
                sender_mask=hotspot_mask,
                receiver_mask=pd.Series(False, index=reference.index),
                sender_score=_winsorized_normalize_series(best_score),
                receiver_score=pd.Series(0.0, index=reference.index),
                title=f"{best_pathway} hotspot ({view}, activity-point-cloud)",
                output_prefix=out_dir / "pathway_roi_overlay",
            )

    return {
        "pathway_activity": pathway_activity,
        "gene_topology": gene_topology,
        "gene_topology_aggregate": gene_topology_aggregate,
        "gene_topology_structuremap": gene_topology_structuremap,
        "pathway_activity_to_cell": pathway_activity_to_cell,
        "pathway_activity_structuremap": pathway_activity_structuremap,
        "pathway_to_cell": pathway_to_cell,
        "pathway_structuremap": pathway_structuremap,
        "pathway_mode_comparison": mode_comparison,
        "activity_diagnostics": activity_diagnostics,
        "anchor_sources": anchor_sources,
        "structure_map": resolved_structure_map,
        "files": output_files,
    }


def ligand_receptor_target_consistency(
    lr_scores: pd.DataFrame,
    receiver_signatures: Mapping[str, Any] | pd.DataFrame,
    ligand_target_prior: pd.DataFrame,
    *,
    ligand_col: str = "ligand",
    receiver_col: str = "receiver_celltype",
    target_col: str = "target",
    prior_weight_col: str = "weight",
    signature_gene_col: str = "gene",
    signature_weight_col: str = "score",
) -> pd.DataFrame:
    """
    Compute a NicheNet-like downstream target consistency layer.

    The default scoring is intentionally lightweight: for each ligand and receiver
    cell type we compute the weighted overlap between the ligand prior targets and
    the receiver signature genes. The output can be merged back onto the
    ``ligand_receptor_topology_analysis`` result table.
    """

    if not {ligand_col, receiver_col, "LR_score"}.issubset(lr_scores.columns):
        raise ValueError("lr_scores must contain ligand, receiver_celltype, and LR_score columns.")
    required_prior = {ligand_col, target_col}
    missing_prior = required_prior.difference(ligand_target_prior.columns)
    if missing_prior:
        raise ValueError(f"ligand_target_prior is missing required columns: {sorted(missing_prior)}")
    prior = ligand_target_prior.copy()
    if prior_weight_col not in prior.columns:
        prior[prior_weight_col] = 1.0
    prior[ligand_col] = prior[ligand_col].astype(str)
    prior[target_col] = prior[target_col].astype(str)
    prior[prior_weight_col] = pd.to_numeric(prior[prior_weight_col], errors="coerce").fillna(1.0).astype(float)

    if isinstance(receiver_signatures, pd.DataFrame):
        required_sig = {receiver_col, signature_gene_col}
        missing_sig = required_sig.difference(receiver_signatures.columns)
        if missing_sig:
            raise ValueError(f"receiver_signatures is missing required columns: {sorted(missing_sig)}")
        sig_df = receiver_signatures.copy()
        if signature_weight_col not in sig_df.columns:
            sig_df[signature_weight_col] = 1.0
        sig_df[receiver_col] = sig_df[receiver_col].astype(str)
        sig_df[signature_gene_col] = sig_df[signature_gene_col].astype(str)
        sig_df[signature_weight_col] = pd.to_numeric(sig_df[signature_weight_col], errors="coerce").fillna(1.0).astype(float)
    else:
        rows: list[dict[str, Any]] = []
        for receiver, signature in receiver_signatures.items():
            if isinstance(signature, Mapping):
                for gene, score in signature.items():
                    rows.append({receiver_col: str(receiver), signature_gene_col: str(gene), signature_weight_col: float(score)})
            else:
                for gene in signature:
                    rows.append({receiver_col: str(receiver), signature_gene_col: str(gene), signature_weight_col: 1.0})
        sig_df = pd.DataFrame(rows, columns=[receiver_col, signature_gene_col, signature_weight_col])

    signature_lookup = {
        receiver: group.set_index(signature_gene_col)[signature_weight_col].astype(float)
        for receiver, group in sig_df.groupby(receiver_col, sort=False)
    }
    prior_lookup = {
        ligand: group.groupby(target_col)[prior_weight_col].sum().astype(float)
        for ligand, group in prior.groupby(ligand_col, sort=False)
    }

    out = lr_scores.copy()
    target_support: list[float] = []
    for row in out.itertuples(index=False):
        ligand = getattr(row, ligand_col)
        receiver = getattr(row, receiver_col)
        if ligand not in prior_lookup or receiver not in signature_lookup:
            target_support.append(0.0)
            continue
        target_weights = prior_lookup[ligand]
        signature_weights = signature_lookup[receiver]
        overlap = target_weights.index.intersection(signature_weights.index)
        if len(overlap) == 0:
            target_support.append(0.0)
            continue
        numerator = float((target_weights.loc[overlap] * signature_weights.loc[overlap]).sum())
        denominator = float(target_weights.sum()) or 1.0
        target_support.append(numerator / denominator)

    out["target_support"] = _normalize_series(pd.Series(target_support, index=out.index))
    out["topology_supported"] = out["LR_score"].astype(float)
    out["target_supported"] = out["target_support"].astype(float)
    out["topology_and_target_supported"] = np.sqrt(out["LR_score"].astype(float) * out["target_support"].astype(float))
    return out
