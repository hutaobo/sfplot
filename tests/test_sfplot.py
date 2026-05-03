#!/usr/bin/env python

"""Basic smoke tests for the ``sfplot`` package."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import sfplot
import sfplot.preprocessing
import sfplot.analysis
import sfplot.plotting


class TestSfplot(unittest.TestCase):
    """Minimal tests that validate the public package surface."""

    def test_package_imports(self):
        """The top-level package should import successfully."""
        self.assertTrue(hasattr(sfplot, "__all__"))
        self.assertGreater(len(sfplot.__all__), 0)

    def test_expected_public_exports_exist(self):
        """Key reviewer-facing APIs should remain exposed at the top level."""
        expected = {
            "load_xenium_data",
            "compute_cophenetic_distances_from_df",
            "compute_cophenetic_distances_from_adata",
            "plot_cophenetic_heatmap",
            "transcript_by_cell_analysis",
            "pick_batch_size",
            "compute_groupwise_average_distance_between_two_dfs",
        }
        self.assertTrue(expected.issubset(set(sfplot.__all__)))

    def test_subpackage_preprocessing_exports(self):
        """sfplot.preprocessing should expose its public APIs."""
        expected = {"load_xenium_data", "merge_xenium_clusters_into_adata", "read_visium_bin"}
        self.assertTrue(expected.issubset(set(sfplot.preprocessing.__all__)))

    def test_subpackage_analysis_exports(self):
        """sfplot.analysis should expose its public APIs."""
        expected = {
            "compute_cophenetic_distances_from_df",
            "compute_cophenetic_distances_from_adata",
            "pick_batch_size",
            "split_B_by_distance_to_A",
            "transcript_by_cell_analysis",
        }
        self.assertTrue(expected.issubset(set(sfplot.analysis.__all__)))

    def test_subpackage_plotting_exports(self):
        """sfplot.plotting should expose its public APIs."""
        expected = {
            "circle_heatmap",
            "plot_cophenetic_heatmap",
        }
        self.assertTrue(expected.issubset(set(sfplot.plotting.__all__)))

    def test_backwards_compat_circular_dendrogram(self):
        """sfplot.circular_dendrogram backwards-compat alias should still work when pycirclize is installed."""
        try:
            import pycirclize  # noqa: F401
        except ImportError:
            self.skipTest("pycirclize not installed")
        self.assertTrue(hasattr(sfplot, "circular_dendrogram"))
        self.assertTrue(hasattr(sfplot.circular_dendrogram, "plot_circular_dendrogram_pycirclize"))


if __name__ == "__main__":
    unittest.main()
