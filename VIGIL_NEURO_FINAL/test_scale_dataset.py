import json
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np

from scale_dataset import generate_shards


class ScaleDatasetTests(unittest.TestCase):
    def test_scaled_shards_isolate_splits_and_exclude_oracle(self):
        tmp = Path.cwd() / ("scale-" + uuid.uuid4().hex)
        tmp.mkdir()
        self.addCleanup(shutil.rmtree, tmp)
        specs = {
            "train": [("post_sweep", "none", 1), ("corridor", "freeze", 2)],
            "val": [("shelf_clear", "delay", 3)],
            "test": [("shelf_hazard", "freeze", 4)],
        }
        manifest = generate_shards(tmp, episodes_per_split=specs)
        self.assertEqual(manifest["oracle_keys_in_inputs"], 0)
        self.assertEqual(manifest["world_split_leakage"], 0)
        for split in ("train", "val", "test"):
            with np.load(tmp / f"{split}.npz") as data:
                self.assertEqual(data["X"].shape[1:], (24, 16))
                self.assertTrue(np.isfinite(data["X"]).all())
                self.assertGreater(len(data["X"]), 0)
        self.assertGreater(manifest["splits"]["train"]["positive_contact_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
