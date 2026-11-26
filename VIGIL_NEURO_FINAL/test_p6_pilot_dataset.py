"""P6.5 & P6.6 Verification Tests: Multi-Horizon Survival Labels, Censoring & Pilot Dataset Invariants."""
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np

from pilot_dataset_v3 import HORIZONS_S, generate_pilot_dataset


class TestP6PilotDataset(unittest.TestCase):
    def test_pilot_dataset_v3_labels_monotonicity_censoring_and_short_vs_long_separation(self):
        tmp = Path.cwd() / ("pilot-v3-" + uuid.uuid4().hex[:6])
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        specs = {
            "train": [("post_sweep", "none", 1100), ("shelf_clear", "none", 1120), ("corridor_pedestrian", "none", 1130)],
            "val": [("shelf_clear", "freeze", 1220)],
            "test": [("ramp_descent", "none", 1350)],
        }
        man = generate_pilot_dataset(tmp, splits=specs)
        self.assertEqual(man["oracle_keys_in_inputs"], 0)
        self.assertEqual(man["world_split_leakage"], 0)
        self.assertEqual(man["feature_dim"], 24)
        self.assertEqual(man["horizons_s"], list(HORIZONS_S))

        # Verify short-safe (0.3s+brake) vs long-dangerous (2.4s continue) separation exists!
        self.assertGreater(man["splits"]["train"]["short_safe_long_danger_count"], 0)

        for sp in ("train", "val", "test"):
            with np.load(tmp / f"{sp}.npz") as d:
                X = d["X"]
                rh = d["y_risk_horizons"]
                hb = d["y_hazard_bin"]
                cens = d["y_censored"]
                du = d["y_signed_delta_u"]
                self.assertEqual(X.shape[1:], (24, 24))
                self.assertTrue(np.isfinite(X).all())
                # Cumulative collision probability over non-decreasing horizons MUST be monotonically non-decreasing
                self.assertTrue(np.all(np.diff(rh, axis=1) >= 0.0), f"Non-monotonic horizons in {sp}")
                # Censoring strictly consistent with final horizon risk
                self.assertTrue(np.array_equal(cens > 0.5, rh[:, -1] < 0.5))
                self.assertTrue(np.array_equal(hb == len(HORIZONS_S), cens > 0.5))
        # Negative signed delta_u preserved cleanly in pilot shards
        with np.load(tmp / "train.npz") as d:
            self.assertGreater(int(np.sum(d["y_signed_delta_u"] < -1e-6)), 0)


if __name__ == "__main__":
    unittest.main()
