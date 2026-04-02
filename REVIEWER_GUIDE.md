# Reviewer Guide

This repository contains the code used to develop the `sfplot` / Cell-GPS workflow described in the manuscript.

## Where to start

If you only have a few minutes, read the files in this order:

1. `README.md`
2. `src/sfplot/Searcher_Findee_Score.py`
3. `src/sfplot/data_processing.py`
4. `src/sfplot/tbc_analysis.py`

## What each area contains

- `src/sfplot/`
  Core package implementation.
- `sfplot-manuscript/`
  Manuscript-facing notebooks, generated figures, and intermediate outputs.
- `benchmarking/`
  Benchmarking material.
- `docs/`
  Lightweight package documentation.

## Minimal install

```bash
git clone https://github.com/hutaobo/sfplot.git
cd sfplot
pip install -e .
```

## Minimal input contract

For the plain DataFrame workflow, the minimal required columns are:

- `x`
- `y`
- `celltype`

With those columns you can run:

```python
from sfplot import compute_cophenetic_distances_from_df

row_coph, col_coph = compute_cophenetic_distances_from_df(
    df,
    x_col="x",
    y_col="y",
    celltype_col="celltype",
)
```

## Xenium workflow

The Xenium workflow expects a standard Xenium output directory and uses:

- `load_xenium_data`
- `compute_cophenetic_distances_from_adata`
- `transcript_by_cell_analysis`

## Reproducibility note

Raw datasets are not included in this repository because of size and data-distribution constraints. The repository is intended to expose the analysis code, plotting functions, and manuscript-facing workflow structure used in the study.
