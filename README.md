# sfplot

`sfplot` is a Python package for spatial structure analysis in spatial omics data. It implements the Search-and-Find Plot (SFplot) / Cell-GPS workflow used in our manuscript to quantify multiscale tissue architecture, compute cophenetic distance-based structure maps, and analyze cell-cell or transcript-cell spatial relationships.

This repository is being maintained as the code companion for manuscript review and future reuse.

## What the package does

- Computes searcher-findee distance matrices from spatial coordinates and cell labels.
- Builds cophenetic distance matrices and StructureMap heatmaps from `AnnData` objects or plain `pandas` tables.
- Loads 10x Xenium outputs and prepares them for downstream spatial analysis.
- Provides a table-bundle Xenium loader that assembles `cells.parquet`, official `*_cell_groups.csv`, and `cell_feature_matrix.h5` without requiring the `spatialdata_io` dependency chain.
- Supports transcript-by-cell analysis for locating transcripts relative to cell types.
- Includes memory-optimized workflows for large datasets.
- Provides plotting utilities such as clustered heatmaps, circular dendrograms, and related summary figures.

## Repository layout

- `src/sfplot/`: core implementation.
- `tests/`: package tests and smoke checks.
- `docs/`: Sphinx documentation.
- `sfplot-manuscript/`: manuscript-specific notebooks, figures, and derived outputs.
- `benchmarking/`: benchmarking-related material.
- `segmentation_methods/`: supporting segmentation workflows.

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/hutaobo/sfplot.git
```

For local development or reviewer inspection:

```bash
git clone https://github.com/hutaobo/sfplot.git
cd sfplot
pip install -e .
```

The package requires Python 3.9 or later.

## Quick start from a coordinate table

The minimal input is a table with spatial coordinates and a cell-type column.

```python
import pandas as pd
from sfplot import compute_cophenetic_distances_from_df, plot_cophenetic_heatmap

df = pd.DataFrame(
    {
        "x": [0, 1, 5, 6],
        "y": [0, 1, 5, 6],
        "celltype": ["A", "A", "B", "B"],
    }
)

row_coph, col_coph = compute_cophenetic_distances_from_df(
    df=df,
    x_col="x",
    y_col="y",
    celltype_col="celltype",
)

plot_cophenetic_heatmap(
    row_coph,
    matrix_name="row_coph",
    output_dir="output",
    output_filename="StructureMap_example.pdf",
    sample="Example",
)
```

## Quick start from Xenium output

```python
from sfplot import load_xenium_data, load_xenium_table_bundle, compute_cophenetic_distances_from_adata

# Legacy `spatialdata_io` route
adata = load_xenium_data("/path/to/xenium/run", normalize=False)

# Stable table-bundle route used for the Atera Xenium benchmark
adata = load_xenium_table_bundle("/path/to/xenium/run", normalize=False)

row_coph, col_coph = compute_cophenetic_distances_from_adata(
    adata,
    cluster_col="Cluster",
    output_dir="output",
)
```

## Useful public entry points

- `load_xenium_data`: load and preprocess Xenium data.
- `load_xenium_table_bundle`: load Xenium data from `cells.parquet` + `*_cell_groups.csv` + `cell_feature_matrix.h5` when `spatialdata_io` is unavailable or unstable.
- `compute_cophenetic_distances_from_df`: compute structure matrices from a coordinate table.
- `compute_weighted_searcher_findee_distance_matrix_from_df`: weighted searcher-findee kernel for entity, pathway, or LR analysis.
- `compute_weighted_cophenetic_distances_from_df`: weighted StructureMap wrapper over the weighted kernel.
- `compute_cophenetic_distances_from_adata`: compute structure matrices from `AnnData`.
- `compute_entity_to_cell_topology`: generalize `t_and_c` from transcripts to arbitrary weighted entities.
- `compute_entity_structuremap`: build StructureMap-style topology among arbitrary weighted entities.
- `plot_cophenetic_heatmap`: generate StructureMap and related clustered heatmaps.
- `transcript_by_cell_analysis`: analyze transcript-to-cell spatial structure at scale.
- `ligand_receptor_topology_analysis`: score sender→receiver ligand-receptor candidates using topology, structure compatibility, and local contact.
- `ligand_receptor_target_consistency`: add a NicheNet-style downstream target-consistency layer.
- `compute_pathway_activity_matrix`: compute rank-based or weighted pathway activities per cell.
- `pathway_topology_analysis`: analyze pathway-to-cell and pathway-to-pathway spatial topology.
- `compute_cophenetic_distances_from_df_memory_opt`: memory-aware alternative for large tables.
- `plot_circular_dendrogram_pycirclize`: circular dendrogram visualization.

## Notes for reviewers

- The package code used for the manuscript is in `src/sfplot/`.
- Manuscript-facing notebooks and generated figure assets are kept in `sfplot-manuscript/`.
- Raw experimental datasets are not bundled in this repository because of size and distribution constraints. The code expects standard spatial omics outputs such as Xenium folders or tabular coordinate inputs.
- When an `sfplot_tbc_formal_wta/results`-style directory is already available, the LR and pathway topology extensions are designed to reuse its `t_and_c_result_*.csv` and `StructureMap_table_*.csv` outputs as the preferred gene-level topology anchors before falling back to recomputation.
- `load_xenium_data` depends on a compatible `spatialdata_io` / `spatialdata` / `ome_zarr` / `zarr` stack. For reviewer reproduction on environments where that stack is mismatched, prefer `load_xenium_table_bundle`.
- A short repository walkthrough is available in [REVIEWER_GUIDE.md](REVIEWER_GUIDE.md).

## Documentation

Sphinx documentation sources are available in `docs/`.

## Citation

If you use this repository in connection with manuscript review, please cite the associated manuscript:

> Cophenetic Spatial Topology Embedding reveals multiscale tissue architecture in spatial omics

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
