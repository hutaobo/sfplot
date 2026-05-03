# Extended Notes for `sfplot`

This file provides a compact English overview of the main modules in the repository. It replaces an earlier internal draft note so that reviewer-facing documentation remains consistent.

## Core modules

### `data_processing.py`

- `load_xenium_data(folder, normalize=True)`
  Loads 10x Xenium outputs into an `AnnData` object and attaches spatial coordinates and cluster annotations when available.
- `load_xenium_table_bundle(folder, ...)`
  Builds an `AnnData` object directly from `cells.parquet`, official `*_cell_groups.csv`, and `cell_feature_matrix.h5`, which is the recommended fallback when the `spatialdata_io` stack is not compatible with the local environment.

### `Searcher_Findee_Score.py`

- `compute_cophenetic_distances_from_adata(...)`
  Computes row and column cophenetic distance matrices from an `AnnData` object.
- `compute_cophenetic_distances_from_df(...)`
  Computes the same quantities from a plain coordinate table.
- `compute_searcher_findee_distance_matrix_from_df(...)`
  Builds the underlying searcher-findee distance matrix before cophenetic transformation.
- `plot_cophenetic_heatmap(...)`
  Produces StructureMap-style clustered heatmaps and can return figures or save publication-ready files.

### `topology_extensions.py`

- `compute_weighted_searcher_findee_distance_matrix_from_df(...)`
  Weighted version of the searcher-findee kernel that preserves backward compatibility when all weights equal one.
- `compute_weighted_cophenetic_distances_from_df(...)`
  Weighted StructureMap wrapper over the weighted kernel.
- `compute_entity_to_cell_topology(...)`
  Generalizes transcript-by-cell topology to arbitrary weighted entities.
- `compute_entity_structuremap(...)`
  Builds StructureMap-style topology among weighted entities such as pathways.
- `ligand_receptor_topology_analysis(...)`
  Scores sender→receiver ligand-receptor candidates using preferred precomputed `t_and_c` anchors, StructureMap compatibility, pseudobulk+detection expression support, and de-saturated local-contact diagnostics.
- `ligand_receptor_target_consistency(...)`
  Adds a lightweight downstream target-consistency layer for ligand prioritization.
- `compute_pathway_activity_matrix(...)`
  Computes rank-based or weighted pathway activities per cell.
- `pathway_topology_analysis(...)`
  Computes dual pathway topology views: a primary gene-topology aggregate and a secondary activity-point-cloud view.

### `compute_cophenetic_distances_from_df_memory_opt.py`

- `pick_batch_size(...)`
  Suggests a batch size for large analyses based on available memory.
- `compute_cophenetic_distances_from_df_memory_opt(...)`
  Memory-aware version of the DataFrame workflow for larger datasets.

### `plotting.py`

- `generate_cluster_distance_heatmap_from_path(...)`
- `generate_cluster_distance_heatmap_from_adata(...)`
- `generate_cluster_distance_heatmap_from_df(...)`

These are convenience wrappers for quick heatmap generation from common input types.

### `tbc_analysis.py`

- `transcript_by_cell_analysis(...)`
  Runs transcript-by-cell analysis with multiprocessing and shared memory for large Xenium datasets.

### `circular_dendrogram.py`

- `plot_circular_dendrogram_pycirclize(...)`
  Draws circular dendrograms from structure matrices.

## Manuscript-related material

- `sfplot-manuscript/` contains notebooks, generated figures, and intermediate outputs used during manuscript preparation.
- `benchmarking/` contains benchmarking-related artifacts.
- `segmentation_methods/` contains supporting segmentation workflows.

## Recommended reading order

1. `README.md`
2. `REVIEWER_GUIDE.md`
3. `src/sfplot/Searcher_Findee_Score.py`
4. `src/sfplot/data_processing.py`
5. `src/sfplot/tbc_analysis.py`

## Practical notes

- If an `sfplot_tbc_formal_wta/results` directory already exists for a sample, use it as the preferred anchor input for ligand-receptor and pathway topology analyses.
- `load_xenium_data` remains available for the legacy `spatialdata_io` route, but the table-bundle loader is more robust for environments with `zarr` / `ome_zarr` / `spatialdata` version mismatches.
