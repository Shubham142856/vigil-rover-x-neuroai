"""Real-time hardware transfer check: verify Layer-3 neural stack runs well under 100 ms (10 Hz) on CPU."""
import csv
import time
import unittest
from pathlib import Path

import numpy as np

from blindspot_dataset import SectorBelief
from camera_escape import Network, visual_looming
from closed_loop_benchmark import _load_neuro_cover_l
from compass_benchmark import ConnectomeCompass, load_compass_connectome
from real_world_env import RealWorldEnv
from scale_dataset import extract_features


class TestLatencyBudget(unittest.TestCase):
    def test_neural_stack_cpu_latency_under_100ms(self):
        bio_npz = Path("runs/graph-controls/escape/biological.npz")
        ann_csv = Path("runs/connectome/escape/annotations.csv")
        compass_dir = Path("runs/connectome/compass")
        weights_npz = Path("runs/kaggle-training-r06/portable_weights.npz")
        if not (bio_npz.exists() and ann_csv.exists() and compass_dir.exists() and weights_npz.exists()):
            self.skipTest("Connectome assets or portable weights missing")

        with ann_csv.open() as f:
            types = {r["root_id"]: r["primary_type"] for r in csv.DictReader(f)}
        ids = Path("runs/graph-controls/escape/nodes.txt").read_text().splitlines()
        net = Network(bio_npz, np.array([types[i] for i in ids]))
        compass = ConnectomeCompass(load_compass_connectome(compass_dir))
        ncl = _load_neuro_cover_l(weights_npz)
        self.assertIsNotNone(ncl)
        belief = SectorBelief(max_age_s=0.6)

        with RealWorldEnv(scene="corridor_pedestrian", seed=0, sensor_noise=True) as env:
            obs = env.observe()
            vstate = None
            ncl_vstate = None
            latencies_ms = []
            for _ in range(40):
                t0 = time.perf_counter()
                snap = belief.update(obs)
                lplc2_d, lc4_d, vstate = visual_looming(obs["rgb"], vstate, 0.1)
                net.update(lplc2_d, lc4_d)
                compass.step(0.0, 0.0)
                feats, ncl_vstate = extract_features(obs, snap, ncl_vstate, 0.1)
                ncl["m4"].step_core(feats)
                ncl["m8"].step_core(feats)
                for act in ("forward", "left", "right", "slow"):
                    ncl["m4"].predict_action(act)
                    ncl["m8"].predict_action(act)
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                obs = env.step("forward")

        p95_ms = float(np.percentile(latencies_ms, 95))
        self.assertLess(p95_ms, 100.0, f"p95 latency {p95_ms:.2f} ms exceeds 100 ms 10Hz budget")


if __name__ == "__main__":
    unittest.main()
