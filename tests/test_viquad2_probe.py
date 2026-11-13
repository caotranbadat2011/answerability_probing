import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probe" / "viquad2"))

from common import binary_metrics, choose_threshold, stratified_partition
from prepare_viquad2 import normalise_row


class ViQuadPreparationTests(unittest.TestCase):
    def test_impossible_maps_to_unanswerable(self):
        row = normalise_row(
            {"id": "x", "context": "c", "question": "q", "is_impossible": True},
            "train",
            0,
        )
        self.assertEqual(row["answerable"], 0)

    def test_answerable_maps_to_one(self):
        row = normalise_row(
            {"id": "x", "context": "c", "question": "q", "is_impossible": False},
            "train",
            0,
        )
        self.assertEqual(row["answerable"], 1)


class CalibrationTests(unittest.TestCase):
    def test_partition_preserves_both_classes(self):
        labels = np.asarray([0] * 10 + [1] * 10)
        selection, calibration = stratified_partition(labels, 0.5, 42)
        self.assertEqual(set(labels[selection]), {0, 1})
        self.assertEqual(set(labels[calibration]), {0, 1})
        self.assertFalse(set(selection) & set(calibration))

    def test_threshold_respects_refusal_recall_target(self):
        probabilities = np.asarray([0.05, 0.10, 0.20, 0.80, 0.90, 0.95])
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        threshold, metrics = choose_threshold(probabilities, labels, 0.90)
        self.assertGreaterEqual(metrics["refusal_recall"], 0.90)
        self.assertEqual(binary_metrics(probabilities, labels, threshold), metrics)


if __name__ == "__main__":
    unittest.main()
