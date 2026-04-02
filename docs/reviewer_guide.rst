==============
Reviewer Guide
==============

This repository contains the code used to develop the ``sfplot`` / Cell-GPS
workflow described in the manuscript.

Suggested reading order
-----------------------

* ``README.md``
* ``src/sfplot/Searcher_Findee_Score.py``
* ``src/sfplot/data_processing.py``
* ``src/sfplot/tbc_analysis.py``

Repository map
--------------

* ``src/sfplot/``: core package implementation
* ``sfplot-manuscript/``: manuscript-facing notebooks and outputs
* ``benchmarking/``: benchmarking material
* ``docs/``: package documentation

Minimal install
---------------

.. code-block:: console

   $ git clone https://github.com/hutaobo/sfplot.git
   $ cd sfplot
   $ pip install -e .

Input contract
--------------

For the DataFrame workflow, the minimal required columns are ``x``, ``y``, and
``celltype``.

Reproducibility note
--------------------

Raw datasets are not bundled in this repository because of size and
data-distribution constraints. The repository is intended to expose the code,
analysis structure, and plotting workflows used in the study.
