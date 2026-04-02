# Extended Notes for `sfplot`

This file provides a compact English overview of the main modules in the repository. It replaces an earlier internal draft note so that reviewer-facing documentation remains consistent.

## Core modules

### `data_processing.py`

- `load_xenium_data(folder, normalize=True)`
  Loads 10x Xenium outputs into an `AnnData` object and attaches spatial coordinates and cluster annotations when available.

### `Searcher_Findee_Score.py`

- `compute_cophenetic_distances_from_adata(...)`
  Computes row and column cophenetic distance matrices from an `AnnData` object.
- `compute_cophenetic_distances_from_df(...)`
  Computes the same quantities from a plain coordinate table.
- `compute_searcher_findee_distance_matrix_from_df(...)`
  Builds the underlying searcher-findee distance matrix before cophenetic transformation.
- `plot_cophenetic_heatmap(...)`
  Produces StructureMap-style clustered heatmaps and can return figures or save publication-ready files.

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
