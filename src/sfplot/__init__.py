"""Top-level package for sfplot."""

__author__ = """Taobo Hu"""
__email__ = 'taobo.hu@scilifelab.se'

from .TCR_distance_heatmap import generate_TCR_distance_heatmap_from_df
from .data_processing import load_xenium_data
from .plotting import generate_cluster_distance_heatmap_from_adata
from .plotting import generate_cluster_distance_heatmap_from_path
from .sfplot import split_B_by_distance_to_A
from .Searcher_Findee_Score import compute_cophenetic_distances_from_adata

__all__ = [
    "load_xenium_data",
    "generate_cluster_distance_heatmap_from_adata",
    "generate_cluster_distance_heatmap_from_path",
    "split_B_by_distance_to_A",
    "generate_TCR_distance_heatmap_from_df",
    "compute_cophenetic_distances_from_adata"
    # 其他导出的函数或类
]
