======
sfplot
======

``sfplot`` is a Python package for spatial structure analysis in spatial omics data.
It implements the Search-and-Find Plot (SFplot) / Cell-GPS workflow used to compute
cophenetic distance-based structure maps, analyze cell-cell and transcript-cell
relationships, and visualize multiscale tissue organization.

Key features
------------

* Compute cophenetic distance matrices from ``AnnData`` objects or coordinate tables.
* Generate StructureMap heatmaps and circular dendrograms.
* Load and preprocess 10x Xenium outputs.
* Run transcript-by-cell analysis for large spatial datasets.
* Support memory-optimized workflows for larger coordinate tables.

Installation
------------

Install from GitHub:

.. code-block:: bash

   pip install git+https://github.com/hutaobo/sfplot.git

For local inspection:

.. code-block:: bash

   git clone https://github.com/hutaobo/sfplot.git
   cd sfplot
   pip install -e .

Reviewer note
-------------

The main implementation lives in ``src/sfplot/``. Manuscript-specific notebooks and
derived figure assets are stored in ``sfplot-manuscript/``. See ``REVIEWER_GUIDE.md``
for a short walkthrough of the repository.
