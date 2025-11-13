"""Top-level package for sfplot."""

__author__ = "Taobo Hu"
__email__ = "taobo.hu@scilifelab.se"

# ---- public APIs (functions) ----
from .circle_heatmap import circle_heatmap
from .compute_col_dendrogram_scores import compute_col_dendrogram_scores
from .compute_cophenetic_distances_from_df_memory_opt import (
    compute_cophenetic_distances_from_df_memory_opt,
    pick_batch_size,
)
from .Searcher_Findee_Score import (
    compute_cophenetic_distances_from_adata,
    compute_cophenetic_distances_from_df,
    compute_cophenetic_from_distance_matrix,
    compute_searcher_findee_distance_matrix_from_df,
    plot_cophenetic_heatmap,
)
from .TCR_distance_heatmap import generate_TCR_distance_heatmap_from_df
from .plotting import (
    generate_cluster_distance_heatmap_from_adata,
    generate_cluster_distance_heatmap_from_df,
    generate_cluster_distance_heatmap_from_path,
)
from .data_processing import load_xenium_data
from .sfplot import split_B_by_distance_to_A
from .tbc_analysis import transcript_by_cell_analysis
from .tbc_analysis_serial import transcript_by_cell_analysis_serial
from .binned_analysis import (
    compute_cophenetic_distances_from_group_mean_matrix,
    calculate_gene_distance_matrix_wmda,
    calculate_gene_distance_matrix_ewnn,
    calculate_gene_distance_matrix_visium,
)
from .xenium_preprocessing import merge_xenium_clusters_into_adata
from .visium_preprocesssing import read_visium_bin

# ---- expose submodule so `sfplot.circular_dendrogram.*` works ----
from . import circular_dendrogram
from .circular_dendrogram import plot_circular_dendrogram_pycirclize

__all__ = [
    # functions
    "merge_xenium_clusters_into_adata",
    "calculate_gene_distance_matrix_ewnn",
    "calculate_gene_distance_matrix_wmda",
    "calculate_gene_distance_matrix_visium",
    "circle_heatmap",
    "compute_col_dendrogram_scores",
    "compute_cophenetic_distances_from_adata",
    "compute_cophenetic_distances_from_df",
    "compute_cophenetic_from_distance_matrix",
    "compute_searcher_findee_distance_matrix_from_df",
    "compute_cophenetic_distances_from_df_memory_opt",
    "compute_cophenetic_distances_from_group_mean_matrix",
    "plot_cophenetic_heatmap",
    "generate_TCR_distance_heatmap_from_df",
    "generate_cluster_distance_heatmap_from_adata",
    "generate_cluster_distance_heatmap_from_df",
    "generate_cluster_distance_heatmap_from_path",
    "load_xenium_data",
    "read_visium_bin",
    "split_B_by_distance_to_A",
    "transcript_by_cell_analysis",
    "transcript_by_cell_analysis_serial",
    "plot_circular_dendrogram_pycirclize",
    # submodules
    "circular_dendrogram",
]
