"""Top-level package for sfplot."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__author__ = "Taobo Hu"
__email__ = "taobo.hu@scilifelab.se"

_LAZY_EXPORTS = {
    "merge_xenium_clusters_into_adata": (".xenium_preprocessing", "merge_xenium_clusters_into_adata"),
    "calculate_gene_distance_matrix_ewnn": (".binned_analysis", "calculate_gene_distance_matrix_ewnn"),
    "calculate_gene_distance_matrix_wmda": (".binned_analysis", "calculate_gene_distance_matrix_wmda"),
    "calculate_gene_distance_matrix_visium": (".binned_analysis", "calculate_gene_distance_matrix_visium"),
    "circle_heatmap": (".circle_heatmap", "circle_heatmap"),
    "compute_col_dendrogram_scores": (".compute_col_dendrogram_scores", "compute_col_dendrogram_scores"),
    "compute_cophenetic_distances_from_adata": (".Searcher_Findee_Score", "compute_cophenetic_distances_from_adata"),
    "compute_cophenetic_distances_from_df": (".Searcher_Findee_Score", "compute_cophenetic_distances_from_df"),
    "compute_cophenetic_from_distance_matrix": (".Searcher_Findee_Score", "compute_cophenetic_from_distance_matrix"),
    "compute_searcher_findee_distance_matrix_from_df": (
        ".Searcher_Findee_Score",
        "compute_searcher_findee_distance_matrix_from_df",
    ),
    "compute_cophenetic_distances_from_df_memory_opt": (
        ".compute_cophenetic_distances_from_df_memory_opt",
        "compute_cophenetic_distances_from_df_memory_opt",
    ),
    "pick_batch_size": (".compute_cophenetic_distances_from_df_memory_opt", "pick_batch_size"),
    "compute_cophenetic_distances_from_group_mean_matrix": (
        ".binned_analysis",
        "compute_cophenetic_distances_from_group_mean_matrix",
    ),
    "plot_cophenetic_heatmap": (".Searcher_Findee_Score", "plot_cophenetic_heatmap"),
    "generate_TCR_distance_heatmap_from_df": (".TCR_distance_heatmap", "generate_TCR_distance_heatmap_from_df"),
    "generate_cluster_distance_heatmap_from_adata": (".plotting", "generate_cluster_distance_heatmap_from_adata"),
    "generate_cluster_distance_heatmap_from_df": (".plotting", "generate_cluster_distance_heatmap_from_df"),
    "generate_cluster_distance_heatmap_from_path": (".plotting", "generate_cluster_distance_heatmap_from_path"),
    "load_xenium_data": (".data_processing", "load_xenium_data"),
    "read_visium_bin": (".visium_preprocesssing", "read_visium_bin"),
    "split_B_by_distance_to_A": (".sfplot", "split_B_by_distance_to_A"),
    "transcript_by_cell_analysis": (".tbc_analysis", "transcript_by_cell_analysis"),
    "transcript_by_cell_analysis_serial": (".tbc_analysis_serial", "transcript_by_cell_analysis_serial"),
    "plot_circular_dendrogram_pycirclize": (".circular_dendrogram", "plot_circular_dendrogram_pycirclize"),
    "circular_dendrogram": (".circular_dendrogram", None),
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
