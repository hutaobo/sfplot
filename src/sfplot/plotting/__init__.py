"""Plotting subpackage: heatmaps, dendrograms, and cluster visualisations."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "circle_heatmap": (".circle_heatmap", "circle_heatmap"),
    "generate_cluster_distance_heatmap_from_adata": (
        ".plotting",
        "generate_cluster_distance_heatmap_from_adata",
    ),
    "generate_cluster_distance_heatmap_from_df": (
        ".plotting",
        "generate_cluster_distance_heatmap_from_df",
    ),
    "generate_cluster_distance_heatmap_from_path": (
        ".plotting",
        "generate_cluster_distance_heatmap_from_path",
    ),
    "generate_TCR_distance_heatmap_from_df": (
        ".tcr_distance_heatmap",
        "generate_TCR_distance_heatmap_from_df",
    ),
    "plot_cophenetic_heatmap": (
        "..analysis.searcher_findee_score",
        "plot_cophenetic_heatmap",
    ),
    "circular_dendrogram": (".circular_dendrogram", None),
    "plot_circular_dendrogram_pycirclize": (
        ".circular_dendrogram",
        "plot_circular_dendrogram_pycirclize",
    ),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
