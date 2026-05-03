"""Preprocessing subpackage: data loading and spatial-data wrangling."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "load_xenium_data": (".data_processing", "load_xenium_data"),
    "load_xenium_table_bundle": (".data_processing", "load_xenium_table_bundle"),
    "read_visium_bin": (".visium_preprocessing", "read_visium_bin"),
    "merge_xenium_clusters_into_adata": (
        ".xenium_preprocessing",
        "merge_xenium_clusters_into_adata",
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
