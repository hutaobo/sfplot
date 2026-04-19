#!/usr/bin/env python

"""Tests for streamed transcript loading helpers."""

import pathlib
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from sfplot.transcript_batching import (  # noqa: E402
    iter_gene_batches,
    load_transcript_batch,
    open_transcript_batch_source,
)


class TestTranscriptBatching(unittest.TestCase):
    """Validate the parquet-backed transcript streaming helpers."""

    def test_iter_gene_batches_preserves_order(self):
        genes = ["A", "B", "C", "D", "E"]
        batches = list(iter_gene_batches(genes, 2))
        self.assertEqual(batches, [["A", "B"], ["C", "D"], ["E"]])

    def test_load_transcript_batch_reads_only_requested_genes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = pathlib.Path(tmpdir)
            df = pd.DataFrame(
                {
                    "x_location": [1.0, 2.0, 3.0, 4.0],
                    "y_location": [5.0, 6.0, 7.0, 8.0],
                    "feature_name": ["EPCAM", "VIM", "NegControlProbe", "MS4A1"],
                    "cell_id": ["c1", "c2", "c3", "c4"],
                }
            )
            df.to_parquet(folder / "transcripts.parquet", index=False)

            source = open_transcript_batch_source(folder)
            self.assertIsNotNone(source)

            batch = load_transcript_batch(source, ["EPCAM", "MS4A1"])
            self.assertEqual(list(batch.columns), ["x", "y", "feature_name"])
            self.assertEqual(set(batch["feature_name"]), {"EPCAM", "MS4A1"})
            self.assertEqual(len(batch), 2)


if __name__ == "__main__":
    unittest.main()
