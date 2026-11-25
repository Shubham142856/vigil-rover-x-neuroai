import json
import shutil
import unittest
import uuid
from pathlib import Path

from blindspot_dataset import ORACLE_KEYS, SectorBelief, generate_dataset, run_paired_check
from rover_sim import RoverSim


class BlindspotTests(unittest.TestCase):
    def test_indistinguishable_hidden_worlds_match_pixels_before_branch(self):
        max_diff, branches = run_paired_check(seed=11, steps_before_branch=6, hidden_side=1)
        self.assertEqual(max_diff, 0)
        self.assertTrue(ORACLE_KEYS.isdisjoint(branches["occluded_clear"]["obs"]))
        self.assertIn("forward", branches["occluded_hazard"]["actions"])

    def test_observation_aging_marks_stale_sectors_unknown(self):
        belief = SectorBelief(max_age_s=0.2)
        with RoverSim(scene="open", fault="freeze", fault_start_s=0.1, fault_end_s=0.8) as sim:
            snap0 = belief.update(sim.observe())
            self.assertEqual(snap0["free_fraction"], 1.0)
            for _ in range(10):
                obs = sim.step("forward")
            snap_frozen = belief.update(obs)
            self.assertFalse(snap_frozen["camera_healthy"])
            # After 0.5s freeze (> max_age_s=0.2), previously free sectors become stale/unknown
            self.assertEqual(snap_frozen["unknown_or_stale_fraction"], 1.0)

    def test_dataset_manifest_preserves_splits_and_excludes_oracle(self):
        tmp = Path.cwd() / ("blindspot-" + uuid.uuid4().hex)
        tmp.mkdir()
        self.addCleanup(shutil.rmtree, tmp)
        manifest = generate_dataset(tmp, seeds_per_split={"train": [101], "val": [201], "test": [301]})
        self.assertTrue(manifest["indistinguishable_histories_verified"])
        self.assertEqual(manifest["oracle_keys_in_inputs"], 0)
        self.assertEqual(manifest["pair_split_leakage"], 0)
        self.assertEqual(manifest["total_records"], 3 * 2 * 5)


if __name__ == "__main__":
    unittest.main()
