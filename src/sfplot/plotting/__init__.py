"""Plotting subpackage: heatmaps, dendrograms, and cluster visualisations."""

from .circle_heatmap import circle_heatmap
from .plotting import (
    generate_cluster_distance_heatmap_from_adata,
    generate_cluster_distance_heatmap_from_df,
    generate_cluster_distance_heatmap_from_path,
)
from .tcr_distance_heatmap import generate_TCR_distance_heatmap_from_df
from ..analysis.searcher_findee_score import plot_cophenetic_heatmap

__all__ = [
    "circle_heatmap",
    "generate_cluster_distance_heatmap_from_adata",
    "generate_cluster_distance_heatmap_from_df",
    "generate_cluster_distance_heatmap_from_path",
    "generate_TCR_distance_heatmap_from_df",
    "plot_cophenetic_heatmap",
]

# pycirclize is an optional dependency
try:
    from . import circular_dendrogram
    from .circular_dendrogram import plot_circular_dendrogram_pycirclize
    __all__ += ["circular_dendrogram", "plot_circular_dendrogram_pycirclize"]
except ImportError:
    circular_dendrogram = None  # type: ignore[assignment]
