"""Tests for P6.9.4 targeted counterfactual generator & untouched world families."""

import json
import tempfile
import unittest
from pathlib import Path

from p69_dataset_audit import audit_dataset_and_models
from p69_targeted_dataset import UNTOUCHED_WORLD_FAMILIES, default_p69_splits, generate_p69_dataset


class TestP69TargetedDataset(unittest.TestCase):
    def test_default_splits_scale_and_family_isolation(self):
        splits = default_p69_splits()
        self.assertGreaterEqual(len(splits["train"]), 70)
        self.assertGreaterEqual(len(splits["val"]), 15)
        self.assertGreaterEqual(len(splits["test"]), 15)
        self.assertGreaterEqual(sum(len(v) for v in splits.values()), 100)
        train_fams = {f for f, _, _ in splits["train"]}
        val_fams = {f for f, _, _ in splits["val"]}
        for uf in UNTOUCHED_WORLD_FAMILIES:
            self.assertNotIn(uf, train_fams)
            self.assertNotIn(uf, val_fams)

    def test_mini_p69_generation_positive_events_and_audit(self):
        mini = {
            "train": [
                ("close_static", "none", 2202),
                ("pedestrian_close_cross", "none", 2100),
                ("open_rover", "none", 2501),
            ],
            "val": [
                ("close_static", "none", 5040),
                ("pedestrian_close_cross", "none", 5020),
            ],
            "test": [
                ("chicane_slalom", "none", 6000),
                ("pedestrian_close_cross", "none", 6060),
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            manifest = generate_p69_dataset(out, splits=mini, miner_weights=None, hard_mine_frac=0.0)
            self.assertEqual(manifest["world_split_leakage"], 0)
            self.assertTrue(manifest["sensor_only_belief_provenance"])
            audit = audit_dataset_and_models(out)
            self.assertGreater(audit["splits"]["train"]["protected_0_30s"]["pos"], 0)
            self.assertGreater(audit["splits"]["val"]["protected_0_30s"]["pos"], 0)
            self.assertGreater(audit["splits"]["test"]["protected_0_30s"]["pos"], 0)
            self.assertGreater(audit["splits"]["train"]["continue_horizons"]["2.40s"]["pos"], 0)
            self.assertTrue(audit["splits"]["test"]["action_duration_0_30s_verified"])
            self.assertTrue(audit["splits"]["test"]["censoring_consistent"])


if __name__ == "__main__":
    unittest.main()
