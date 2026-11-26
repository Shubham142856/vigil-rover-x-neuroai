import unittest
from pathlib import Path

from scientific_validation import paired_bootstrap_ci, validate


class ScientificValidationTests(unittest.TestCase):
    def test_paired_bootstrap_ci_brackets_true_difference(self):
        ci = paired_bootstrap_ci([2.0, 3.0, 4.0, 5.0], [1.0, 1.5, 2.5, 3.0], n_boot=500)
        self.assertAlmostEqual(ci["mean_diff"], 1.5)
        self.assertLessEqual(ci["ci95_low"], ci["mean_diff"])
        self.assertGreaterEqual(ci["ci95_high"], ci["mean_diff"])

    def test_validation_report_honestly_flags_missing_banc_without_fabricating(self):
        req = [
            Path("runs/camera-escape-zero-fp/episodes.csv"),
            Path("runs/closed-loop-eval/episodes.csv"),
            Path("runs/compass-benchmark/summary.json"),
        ]
        if not all(p.exists() for p in req):
            self.skipTest("Benchmark outputs missing")
        rep = validate(req[0], req[1], req[2], Path("data/banc-v888-absent"), Path("runs/scientific-validation"))
        self.assertFalse(rep["banc_v888"]["dataset_present"])
        self.assertIn("NOT_DOWNLOADED_LOCAL", rep["banc_v888"]["measurement_status"])


if __name__ == "__main__":
    unittest.main()
