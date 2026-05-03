"""Analysis subpackage: distance-matrix computation and cophenetic scoring."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "compute_cophenetic_distances_from_adata": (
        ".searcher_findee_score",
        "compute_cophenetic_distances_from_adata",
    ),
    "compute_cophenetic_distances_from_df": (
        ".searcher_findee_score",
        "compute_cophenetic_distances_from_df",
    ),
    "compute_cophenetic_from_distance_matrix": (
        ".searcher_findee_score",
        "compute_cophenetic_from_distance_matrix",
    ),
    "compute_searcher_findee_distance_matrix_from_df": (
        ".searcher_findee_score",
        "compute_searcher_findee_distance_matrix_from_df",
    ),
    "compute_cophenetic_distances_from_group_mean_matrix": (
        ".binned_analysis",
        "compute_cophenetic_distances_from_group_mean_matrix",
    ),
    "calculate_gene_distance_matrix_wmda": (
        ".binned_analysis",
        "calculate_gene_distance_matrix_wmda",
    ),
    "calculate_gene_distance_matrix_ewnn": (
        ".binned_analysis",
        "calculate_gene_distance_matrix_ewnn",
    ),
    "calculate_gene_distance_matrix_visium": (
        ".binned_analysis",
        "calculate_gene_distance_matrix_visium",
    ),
    "compute_col_dendrogram_scores": (
        ".compute_col_dendrogram_scores",
        "compute_col_dendrogram_scores",
    ),
    "compute_cophenetic_distances_from_df_memory_opt": (
        ".compute_cophenetic_distances_from_df_memory_opt",
        "compute_cophenetic_distances_from_df_memory_opt",
    ),
    "pick_batch_size": (
        ".compute_cophenetic_distances_from_df_memory_opt",
        "pick_batch_size",
    ),
    "compute_groupwise_average_distance_between_two_dfs": (
        ".ghost_searcher_with_findee",
        "compute_groupwise_average_distance_between_two_dfs",
    ),
    "split_B_by_distance_to_A": (".split_utils", "split_B_by_distance_to_A"),
    "transcript_by_cell_analysis": (".tbc_analysis", "transcript_by_cell_analysis"),
    "transcript_by_cell_analysis_serial": (
        ".tbc_analysis_serial",
        "transcript_by_cell_analysis_serial",
    ),
    "build_entity_points_from_expression": (
        ".topology_extensions",
        "build_entity_points_from_expression",
    ),
    "compute_entity_structuremap": (
        ".topology_extensions",
        "compute_entity_structuremap",
    ),
    "compute_entity_to_cell_topology": (
        ".topology_extensions",
        "compute_entity_to_cell_topology",
    ),
    "compute_pathway_activity_matrix": (
        ".topology_extensions",
        "compute_pathway_activity_matrix",
    ),
    "compute_weighted_cophenetic_distances_from_df": (
        ".topology_extensions",
        "compute_weighted_cophenetic_distances_from_df",
    ),
    "compute_weighted_searcher_findee_distance_matrix_from_df": (
        ".topology_extensions",
        "compute_weighted_searcher_findee_distance_matrix_from_df",
    ),
    "ligand_receptor_target_consistency": (
        ".topology_extensions",
        "ligand_receptor_target_consistency",
    ),
    "ligand_receptor_topology_analysis": (
        ".topology_extensions",
        "ligand_receptor_topology_analysis",
    ),
    "pathway_topology_analysis": (
        ".topology_extensions",
        "pathway_topology_analysis",
    ),
    "calculate_gene_distance_matrix_wmda_gpu": (
        ".binned_analysis_gpu",
        "calculate_gene_distance_matrix_wmda_gpu",
    ),
    "calculate_gene_distance_matrix_ewnn_gpu": (
        ".binned_analysis_gpu",
        "calculate_gene_distance_matrix_ewnn_gpu",
    ),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
