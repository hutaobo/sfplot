"""Preprocessing subpackage: data loading and spatial-data wrangling."""

from .data_processing import load_xenium_data, load_xenium_table_bundle
from .visium_preprocessing import read_visium_bin

__all__ = [
    "load_xenium_data",
    "load_xenium_table_bundle",
    "read_visium_bin",
]

# geopandas / shapely are optional
try:
    from .xenium_preprocessing import merge_xenium_clusters_into_adata
    __all__ += ["merge_xenium_clusters_into_adata"]
except ImportError:
    pass
