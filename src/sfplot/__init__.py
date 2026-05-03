"""Top-level package for sfplot."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__author__ = "Taobo Hu"
__email__ = "taobo.hu@scilifelab.se"

_LAZY_EXPORTS = {
    # subpackages
    "preprocessing": (".preprocessing", None),
    "analysis": (".analysis", None),
    "plotting": (".plotting", None),
    # preprocessing
    "load_xenium_data": (".preprocessing.data_processing", "load_xenium_data"),
    "load_xenium_table_bundle": (".preprocessing.data_processing", "load_xenium_table_bundle"),
    "read_visium_bin": (".preprocessing.visium_preprocessing", "read_visium_bin"),
    "merge_xenium_clusters_into_adata": (
        ".preprocessing.xenium_preprocessing",
        "merge_xenium_clusters_into_adata",
    ),
    # analysis
    "calculate_gene_distance_matrix_ewnn": (
        ".analysis.binned_analysis",
        "calculate_gene_distance_matrix_ewnn",
    ),
    "calculate_gene_distance_matrix_wmda": (
        ".analysis.binned_analysis",
        "calculate_gene_distance_matrix_wmda",
    ),
    "calculate_gene_distance_matrix_visium": (
        ".analysis.binned_analysis",
        "calculate_gene_distance_matrix_visium",
    ),
    "calculate_gene_distance_matrix_ewnn_gpu": (
        ".analysis.binned_analysis_gpu",
        "calculate_gene_distance_matrix_ewnn_gpu",
    ),
    "calculate_gene_distance_matrix_wmda_gpu": (
        ".analysis.binned_analysis_gpu",
        "calculate_gene_distance_matrix_wmda_gpu",
    ),
    "compute_col_dendrogram_scores": (
        ".analysis.compute_col_dendrogram_scores",
        "compute_col_dendrogram_scores",
    ),
    "compute_cophenetic_distances_from_adata": (
        ".analysis.searcher_findee_score",
        "compute_cophenetic_distances_from_adata",
    ),
    "compute_cophenetic_distances_from_df": (
        ".analysis.searcher_findee_score",
        "compute_cophenetic_distances_from_df",
    ),
    "compute_cophenetic_from_distance_matrix": (
        ".analysis.searcher_findee_score",
        "compute_cophenetic_from_distance_matrix",
    ),
    "compute_searcher_findee_distance_matrix_from_df": (
        ".analysis.searcher_findee_score",
        "compute_searcher_findee_distance_matrix_from_df",
    ),
    "compute_weighted_searcher_findee_distance_matrix_from_df": (
        ".analysis.topology_extensions",
        "compute_weighted_searcher_findee_distance_matrix_from_df",
    ),
    "compute_weighted_cophenetic_distances_from_df": (
        ".analysis.topology_extensions",
        "compute_weighted_cophenetic_distances_from_df",
    ),
    "compute_entity_to_cell_topology": (
        ".analysis.topology_extensions",
        "compute_entity_to_cell_topology",
    ),
    "compute_entity_structuremap": (
        ".analysis.topology_extensions",
        "compute_entity_structuremap",
    ),
    "build_entity_points_from_expression": (
        ".analysis.topology_extensions",
        "build_entity_points_from_expression",
    ),
    "compute_pathway_activity_matrix": (
        ".analysis.topology_extensions",
        "compute_pathway_activity_matrix",
    ),
    "ligand_receptor_topology_analysis": (
        ".analysis.topology_extensions",
        "ligand_receptor_topology_analysis",
    ),
    "ligand_receptor_target_consistency": (
        ".analysis.topology_extensions",
        "ligand_receptor_target_consistency",
    ),
    "pathway_topology_analysis": (
        ".analysis.topology_extensions",
        "pathway_topology_analysis",
    ),
    "compute_cophenetic_distances_from_df_memory_opt": (
        ".analysis.compute_cophenetic_distances_from_df_memory_opt",
        "compute_cophenetic_distances_from_df_memory_opt",
    ),
    "pick_batch_size": (
        ".analysis.compute_cophenetic_distances_from_df_memory_opt",
        "pick_batch_size",
    ),
    "compute_cophenetic_distances_from_group_mean_matrix": (
        ".analysis.binned_analysis",
        "compute_cophenetic_distances_from_group_mean_matrix",
    ),
    "compute_groupwise_average_distance_between_two_dfs": (
        ".analysis.ghost_searcher_with_findee",
        "compute_groupwise_average_distance_between_two_dfs",
    ),
    "split_B_by_distance_to_A": (".analysis.split_utils", "split_B_by_distance_to_A"),
    "transcript_by_cell_analysis": (".analysis.tbc_analysis", "transcript_by_cell_analysis"),
    "transcript_by_cell_analysis_serial": (
        ".analysis.tbc_analysis_serial",
        "transcript_by_cell_analysis_serial",
    ),
    # plotting
    "circle_heatmap": (".plotting.circle_heatmap", "circle_heatmap"),
    "generate_TCR_distance_heatmap_from_df": (
        ".plotting.tcr_distance_heatmap",
        "generate_TCR_distance_heatmap_from_df",
    ),
    "generate_cluster_distance_heatmap_from_adata": (
        ".plotting.plotting",
        "generate_cluster_distance_heatmap_from_adata",
    ),
    "generate_cluster_distance_heatmap_from_df": (
        ".plotting.plotting",
        "generate_cluster_distance_heatmap_from_df",
    ),
    "generate_cluster_distance_heatmap_from_path": (
        ".plotting.plotting",
        "generate_cluster_distance_heatmap_from_path",
    ),
    "plot_cophenetic_heatmap": (
        ".analysis.searcher_findee_score",
        "plot_cophenetic_heatmap",
    ),
    "plot_circular_dendrogram_pycirclize": (
        ".plotting.circular_dendrogram",
        "plot_circular_dendrogram_pycirclize",
    ),
    "circular_dendrogram": (".plotting.circular_dendrogram", None),
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
