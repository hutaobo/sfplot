=====
Usage
=====

From a coordinate table
-----------------------

The minimal table-based workflow expects ``x``, ``y``, and ``celltype`` columns.

.. code-block:: python

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

From Xenium output
------------------

.. code-block:: python

   from sfplot import load_xenium_data, compute_cophenetic_distances_from_adata

   adata = load_xenium_data("/path/to/xenium/run", normalize=False)
   row_coph, col_coph = compute_cophenetic_distances_from_adata(
       adata,
       cluster_col="Cluster",
       output_dir="output",
   )

Large datasets
--------------

For larger coordinate tables, use the memory-aware helper:

.. code-block:: python

   from sfplot import (
       pick_batch_size,
       compute_cophenetic_distances_from_df_memory_opt,
   )

   batch_size = pick_batch_size(n_cells=len(df))
   row_coph, col_coph = compute_cophenetic_distances_from_df_memory_opt(
       df=df,
       batch_size=batch_size,
   )
