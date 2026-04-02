#!/usr/bin/env python

"""Basic smoke tests for the ``sfplot`` package."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import sfplot


class TestSfplot(unittest.TestCase):
    """Minimal tests that validate the public package surface."""

    def test_package_imports(self):
        """The top-level package should import successfully."""
        self.assertTrue(hasattr(sfplot, "__all__"))
        self.assertGreater(len(sfplot.__all__), 0)

    def test_expected_public_exports_exist(self):
        """Key reviewer-facing APIs should remain exposed."""
        expected = {
            "load_xenium_data",
            "compute_cophenetic_distances_from_df",
            "compute_cophenetic_distances_from_adata",
            "plot_cophenetic_heatmap",
            "transcript_by_cell_analysis",
            "pick_batch_size",
        }
        self.assertTrue(expected.issubset(set(sfplot.__all__)))


if __name__ == "__main__":
    unittest.main()
