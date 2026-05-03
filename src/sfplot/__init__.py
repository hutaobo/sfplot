"""Top-level package for sfplot."""

__author__ = "Taobo Hu"
__email__ = "taobo.hu@scilifelab.se"

# ---- preprocessing ----
from .preprocessing import load_xenium_data, read_visium_bin

_xenium_exports = []
try:
    from .preprocessing import merge_xenium_clusters_into_adata
    _xenium_exports = ["merge_xenium_clusters_into_adata"]
except ImportError:
    pass

# ---- analysis ----
from .analysis import (
    compute_cophenetic_distances_from_adata,
    compute_cophenetic_distances_from_df,
    compute_cophenetic_from_distance_matrix,
    compute_searcher_findee_distance_matrix_from_df,
    compute_cophenetic_distances_from_group_mean_matrix,
    calculate_gene_distance_matrix_wmda,
    calculate_gene_distance_matrix_ewnn,
    calculate_gene_distance_matrix_visium,
    compute_col_dendrogram_scores,
    compute_cophenetic_distances_from_df_memory_opt,
    pick_batch_size,
    compute_groupwise_average_distance_between_two_dfs,
    split_B_by_distance_to_A,
    transcript_by_cell_analysis,
    transcript_by_cell_analysis_serial,
)

# ---- plotting ----
from .plotting import (
    circle_heatmap,
    generate_cluster_distance_heatmap_from_adata,
    generate_cluster_distance_heatmap_from_df,
    generate_cluster_distance_heatmap_from_path,
    generate_TCR_distance_heatmap_from_df,
    plot_cophenetic_heatmap,
)

# ---- expose submodules so `sfplot.preprocessing.*` etc. work ----
from . import preprocessing, analysis, plotting

# pycirclize is optional — expose gracefully
from .plotting import circular_dendrogram  # may be None when pycirclize absent
try:
    from .plotting import plot_circular_dendrogram_pycirclize
    _pycirclize_exports = ["plot_circular_dendrogram_pycirclize", "circular_dendrogram"]
except ImportError:
    _pycirclize_exports = []

# GPU variants — only available when PyTorch is installed
try:
    from .analysis import (
        calculate_gene_distance_matrix_wmda_gpu,
        calculate_gene_distance_matrix_ewnn_gpu,
    )
    _gpu_exports = [
        "calculate_gene_distance_matrix_wmda_gpu",
        "calculate_gene_distance_matrix_ewnn_gpu",
    ]
except ImportError:
    _gpu_exports = []

__all__ = [
    # preprocessing
    "load_xenium_data",
    "read_visium_bin",
    # analysis
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
    # plotting
    "circle_heatmap",
    "generate_cluster_distance_heatmap_from_adata",
    "generate_cluster_distance_heatmap_from_df",
    "generate_cluster_distance_heatmap_from_path",
    "generate_TCR_distance_heatmap_from_df",
    "plot_cophenetic_heatmap",
    # submodules
    "preprocessing",
    "analysis",
    "plotting",
] + _xenium_exports + _pycirclize_exports + _gpu_exports

