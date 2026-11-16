import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probe" / "paper_data"))

from train_paper_probe import fit_bias_f1, item_folds, metrics


class PaperDataProtocolTests(unittest.TestCase):
    def test_item_folds_keep_each_item_together(self):
        meta = [
            {"item_id": "a"}, {"item_id": "a"},
            {"item_id": "b"}, {"item_id": "b"},
            {"item_id": "c"}, {"item_id": "c"},
        ]
        folds = item_folds(meta, folds=3, seed=0)
        self.assertEqual(folds[0], folds[1])
        self.assertEqual(folds[2], folds[3])
        self.assertEqual(folds[4], folds[5])
        self.assertEqual(set(folds.tolist()), {0, 1, 2})

    def test_f1_bias_and_metrics_detect_both_classes(self):
        scores = np.asarray([-3.0, -2.0, 1.0, 2.0])
        labels = np.asarray([0, 0, 1, 1])
        bias = fit_bias_f1(scores, labels)
        result = metrics(scores, labels, bias)
        self.assertEqual(result["refusal_correct"], 2)
        self.assertEqual(result["answerable_correct"], 2)
        self.assertEqual(result["refusal_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
