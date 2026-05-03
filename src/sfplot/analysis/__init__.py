"""Analysis subpackage: distance-matrix computation and cophenetic scoring."""

from .searcher_findee_score import (
    compute_cophenetic_distances_from_adata,
    compute_cophenetic_distances_from_df,
    compute_cophenetic_from_distance_matrix,
    compute_searcher_findee_distance_matrix_from_df,
)
from .binned_analysis import (
    compute_cophenetic_distances_from_group_mean_matrix,
    calculate_gene_distance_matrix_wmda,
    calculate_gene_distance_matrix_ewnn,
    calculate_gene_distance_matrix_visium,
)
from .compute_col_dendrogram_scores import compute_col_dendrogram_scores
from .compute_cophenetic_distances_from_df_memory_opt import (
    compute_cophenetic_distances_from_df_memory_opt,
    pick_batch_size,
)
from .ghost_searcher_with_findee import (
    compute_groupwise_average_distance_between_two_dfs,
)
from .split_utils import split_B_by_distance_to_A
from .tbc_analysis import transcript_by_cell_analysis
from .tbc_analysis_serial import transcript_by_cell_analysis_serial
from .topology_extensions import (
    build_entity_points_from_expression,
    compute_entity_structuremap,
    compute_entity_to_cell_topology,
    compute_pathway_activity_matrix,
    compute_weighted_cophenetic_distances_from_df,
    compute_weighted_searcher_findee_distance_matrix_from_df,
    ligand_receptor_target_consistency,
    ligand_receptor_topology_analysis,
    pathway_topology_analysis,
)

__all__ = [
    "compute_cophenetic_distances_from_adata",
    "compute_cophenetic_distances_from_df",
    "compute_cophenetic_from_distance_matrix",
    "compute_searcher_findee_distance_matrix_from_df",
    "compute_cophenetic_distances_from_group_mean_matrix",
    "calculate_gene_distance_matrix_wmda",
    "calculate_gene_distance_matrix_ewnn",
    "calculate_gene_distance_matrix_visium",
    "compute_col_dendrogram_scores",
    "compute_cophenetic_distances_from_df_memory_opt",
    "pick_batch_size",
    "compute_groupwise_average_distance_between_two_dfs",
    "split_B_by_distance_to_A",
    "transcript_by_cell_analysis",
    "transcript_by_cell_analysis_serial",
    "compute_weighted_searcher_findee_distance_matrix_from_df",
    "compute_weighted_cophenetic_distances_from_df",
    "compute_entity_to_cell_topology",
    "compute_entity_structuremap",
    "build_entity_points_from_expression",
    "compute_pathway_activity_matrix",
    "ligand_receptor_topology_analysis",
    "ligand_receptor_target_consistency",
    "pathway_topology_analysis",
]

# GPU variants — only available when PyTorch is installed
try:
    from .binned_analysis_gpu import (
        calculate_gene_distance_matrix_wmda_gpu,
        calculate_gene_distance_matrix_ewnn_gpu,
    )
    __all__ += [
        "calculate_gene_distance_matrix_wmda_gpu",
        "calculate_gene_distance_matrix_ewnn_gpu",
    ]
except ImportError:
    pass
