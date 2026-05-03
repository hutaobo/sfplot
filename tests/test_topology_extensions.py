#!/usr/bin/env python

"""Tests for weighted topology, ligand-receptor, and pathway extensions."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from sfplot import (  # noqa: E402
    compute_pathway_activity_matrix,
    compute_searcher_findee_distance_matrix_from_df,
    compute_weighted_searcher_findee_distance_matrix_from_df,
    ligand_receptor_target_consistency,
    ligand_receptor_topology_analysis,
    pathway_topology_analysis,
)


class TestTopologyExtensions(unittest.TestCase):
    def setUp(self):
        self.reference = pd.DataFrame(
            {
                "cell_id": [
                    "s1",
                    "s2",
                    "s3",
                    "s4",
                    "rare1",
                    "rare2",
                    "r1",
                    "r2",
                    "r3",
                    "r4",
                ],
                "x": [0.0, 0.1, 0.0, 0.1, 0.3, 0.3, 1.0, 1.1, 1.0, 1.1],
                "y": [0.0, 0.0, 0.2, 0.2, 0.05, 0.25, 0.0, 0.0, 0.2, 0.2],
                "celltype": [
                    "Sender",
                    "Sender",
                    "Sender",
                    "Sender",
                    "Rare",
                    "Rare",
                    "Receiver",
                    "Receiver",
                    "Receiver",
                    "Receiver",
                ],
            }
        )
        self.reference.index = self.reference["cell_id"]
        self.expression = pd.DataFrame(
            {
                "LIG": [18.0, 19.0, 20.0, 17.0, 16.0, 15.0, 0.0, 0.0, 0.0, 0.0],
                "REC": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 8.5, 9.5, 8.8],
                "CXCL12": [6.0, 6.5, 5.8, 6.2, 0.5, 0.2, 0.0, 0.1, 0.0, 0.0],
                "CXCR4": [0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 7.0, 6.8, 7.2, 6.9],
                "G1": [2.0, 2.5, 2.2, 2.3, 0.0, 0.0, 8.0, 8.5, 7.5, 8.1],
                "G2": [9.0, 8.8, 9.1, 8.9, 0.1, 0.1, 1.0, 1.2, 1.1, 1.0],
                "P1A": [5.0, 4.8, 5.2, 5.1, 0.1, 0.0, 0.2, 0.1, 0.0, 0.1],
                "P1B": [4.9, 5.0, 5.1, 4.8, 0.0, 0.0, 0.2, 0.1, 0.1, 0.0],
                "P2A": [0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 4.8, 5.0, 5.2, 4.9],
                "SPARSE": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 9.5, 0.0, 0.0],
            },
            index=self.reference["cell_id"],
        )
        self.t_and_c = pd.DataFrame(
            {
                "Sender": {
                    "LIG": 0.05,
                    "REC": 1.0,
                    "CXCL12": 0.08,
                    "CXCR4": 1.0,
                    "P1A": 0.10,
                    "P1B": 0.12,
                    "P2A": 1.0,
                    "SPARSE": 1.0,
                },
                "Rare": {
                    "LIG": 0.30,
                    "REC": 1.0,
                    "CXCL12": 0.35,
                    "CXCR4": 1.0,
                    "P1A": 0.60,
                    "P1B": 0.65,
                    "P2A": 1.0,
                    "SPARSE": 1.0,
                },
                "Receiver": {
                    "LIG": 1.0,
                    "REC": 0.04,
                    "CXCL12": 0.90,
                    "CXCR4": 0.06,
                    "P1A": 1.0,
                    "P1B": 1.0,
                    "P2A": 0.08,
                    "SPARSE": 0.10,
                },
            }
        )
        self.structure_map = pd.DataFrame(
            [
                [0.0, 0.25, 0.35],
                [0.25, 0.0, 0.45],
                [0.35, 0.45, 0.0],
            ],
            index=["Sender", "Rare", "Receiver"],
            columns=["Sender", "Rare", "Receiver"],
        )

    def test_weighted_distance_reduces_to_unweighted_when_weights_are_one(self):
        df = self.reference[["x", "y", "celltype"]].copy()
        df["weight"] = 1.0
        unweighted = compute_searcher_findee_distance_matrix_from_df(
            df[["x", "y", "celltype"]],
            x_col="x",
            y_col="y",
            celltype_col="celltype",
        )
        weighted = compute_weighted_searcher_findee_distance_matrix_from_df(
            df,
            x_col="x",
            y_col="y",
            group_col="celltype",
            weight_col="weight",
        )
        np.testing.assert_allclose(weighted.loc[unweighted.index, unweighted.columns], unweighted)

    def test_single_gene_weighted_sum_pathway_activity_matches_normalized_expression(self):
        activity = compute_pathway_activity_matrix(
            self.expression[["G1"]],
            {"SingleGenePathway": ["G1"]},
            method="weighted_sum",
            normalize=True,
        )
        g1 = self.expression["G1"]
        expected = (g1 - g1.min()) / (g1.max() - g1.min())
        np.testing.assert_allclose(activity["SingleGenePathway"], expected)

    def test_ligand_receptor_prefers_precomputed_anchors_and_emits_diagnostics(self):
        lr_pairs = pd.DataFrame([{"ligand": "LIG", "receptor": "REC", "evidence_weight": 1.0}])
        with tempfile.TemporaryDirectory() as tmpdir:
            result = ligand_receptor_topology_analysis(
                reference_df=self.reference,
                expression_df=self.expression,
                lr_pairs=lr_pairs,
                output_dir=tmpdir,
                t_and_c_df=self.t_and_c,
                structure_map=self.structure_map,
                anchor_mode="precomputed",
                k_neighbors=3,
                radius=1.5,
                min_cross_edges=1,
            )

            scores = result["scores"]
            best = scores.iloc[0]
            self.assertEqual(best["sender_celltype"], "Sender")
            self.assertEqual(best["receiver_celltype"], "Receiver")
            self.assertEqual(best["anchor_source_ligand"], "precomputed")
            self.assertEqual(best["anchor_source_receptor"], "precomputed")
            self.assertGreater(best["contact_strength_raw"], 0.0)
            self.assertGreater(best["contact_coverage"], 0.0)
            self.assertGreaterEqual(best["cross_edge_count"], 1)

            files = result["files"]
            for key in (
                "ligand_to_cell",
                "receptor_to_cell",
                "lr_sender_receiver_scores",
                "lr_component_diagnostics",
            ):
                self.assertTrue(pathlib.Path(files[key]).exists(), msg=key)
            self.assertTrue(pathlib.Path(files["lr_hotspot_cells_csv"]).exists())

    def test_precomputed_mode_gracefully_falls_back_to_recompute(self):
        lr_pairs = pd.DataFrame([{"ligand": "CXCL12", "receptor": "CXCR4", "evidence_weight": 1.0}])
        result = ligand_receptor_topology_analysis(
            reference_df=self.reference,
            expression_df=self.expression,
            lr_pairs=lr_pairs,
            anchor_mode="precomputed",
            k_neighbors=3,
            min_cross_edges=1,
        )
        best = result["scores"].iloc[0]
        self.assertEqual(best["anchor_source_ligand"], "recompute")
        self.assertEqual(best["anchor_source_receptor"], "recompute")
        self.assertEqual(best["sender_celltype"], "Sender")
        self.assertEqual(best["receiver_celltype"], "Receiver")

    def test_hybrid_mode_marks_partial_anchor_sources(self):
        partial_t_and_c = self.t_and_c.loc[["LIG"]]
        lr_pairs = pd.DataFrame([{"ligand": "LIG", "receptor": "REC", "evidence_weight": 1.0}])
        result = ligand_receptor_topology_analysis(
            reference_df=self.reference,
            expression_df=self.expression,
            lr_pairs=lr_pairs,
            t_and_c_df=partial_t_and_c,
            structure_map=self.structure_map,
            anchor_mode="hybrid",
            k_neighbors=3,
            min_cross_edges=1,
        )
        top = result["scores"].iloc[0]
        self.assertEqual(top["anchor_source_ligand"], "precomputed")
        self.assertEqual(top["anchor_source_receptor"], "recompute")

    def test_local_contact_zero_when_cross_edges_below_threshold(self):
        lr_pairs = pd.DataFrame([{"ligand": "LIG", "receptor": "REC", "evidence_weight": 1.0}])
        result = ligand_receptor_topology_analysis(
            reference_df=self.reference,
            expression_df=self.expression,
            lr_pairs=lr_pairs,
            t_and_c_df=self.t_and_c,
            structure_map=self.structure_map,
            anchor_mode="precomputed",
            k_neighbors=2,
            min_cross_edges=999,
        )
        top = result["scores"].iloc[0]
        self.assertEqual(float(top["local_contact"]), 0.0)
        self.assertFalse(np.isnan(float(top["LR_score"])))

    def test_pathway_topology_analysis_outputs_dual_modes_and_primary_aggregate(self):
        pathways = {
            "SenderProgram": ["P1A", "P1B"],
            "ReceiverProgram": ["P2A"],
            "SparseProgram": ["SPARSE"],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            result = pathway_topology_analysis(
                pathway_definitions=pathways,
                reference_df=self.reference,
                expression_df=self.expression[["P1A", "P1B", "P2A", "SPARSE"]],
                output_dir=tmpdir,
                t_and_c_df=self.t_and_c.loc[["P1A", "P1B", "P2A", "SPARSE"]],
                structure_map=self.structure_map,
                anchor_mode="precomputed",
                scoring_method="weighted_sum",
                primary_pathway_mode="gene_topology_aggregate",
                activity_threshold_schedule=(0.95, 0.80, 0.50),
                min_activity_cells=3,
            )

            aggregate = result["gene_topology_aggregate"]
            primary = result["pathway_to_cell"]
            activity = result["pathway_activity_to_cell"]
            comparison = result["pathway_mode_comparison"].set_index("pathway")

            pd.testing.assert_frame_equal(primary, aggregate)
            self.assertEqual(primary.loc["SenderProgram"].idxmin(), "Sender")
            self.assertEqual(primary.loc["ReceiverProgram"].idxmin(), "Receiver")
            self.assertEqual(primary.loc["SparseProgram"].idxmin(), "Receiver")
            self.assertIn("SparseProgram", activity.index)
            self.assertGreaterEqual(int(comparison.loc["SparseProgram", "retained_cell_count"]), 3)
            self.assertIn(
                float(comparison.loc["SparseProgram", "retained_quantile"]),
                {0.95, 0.8, 0.5},
            )

            files = result["files"]
            for key in (
                "pathway_to_cell",
                "pathway_structuremap",
                "pathway_activity_to_cell",
                "pathway_activity_structuremap",
                "pathway_mode_comparison",
            ):
                self.assertTrue(pathlib.Path(files[key]).exists(), msg=key)
            self.assertTrue(pathlib.Path(files["pathway_hotspot_cells_csv"]).exists())

    def test_target_consistency_adds_support_columns(self):
        lr_scores = pd.DataFrame(
            {
                "ligand": ["LIG"],
                "receiver_celltype": ["Receiver"],
                "LR_score": [0.8],
            }
        )
        receiver_signatures = {"Receiver": {"TG1": 2.0, "TG2": 1.0}}
        ligand_target_prior = pd.DataFrame(
            {
                "ligand": ["LIG", "LIG", "OTHER"],
                "target": ["TG1", "TG3", "TG2"],
                "weight": [0.8, 0.2, 1.0],
            }
        )
        out = ligand_receptor_target_consistency(lr_scores, receiver_signatures, ligand_target_prior)
        self.assertTrue(
            {
                "target_support",
                "topology_supported",
                "target_supported",
                "topology_and_target_supported",
            }.issubset(out.columns)
        )
        self.assertGreater(out.loc[0, "target_support"], 0.0)


if __name__ == "__main__":
    unittest.main()
