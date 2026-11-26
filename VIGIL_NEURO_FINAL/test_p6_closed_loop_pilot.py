"""Tests for P6.8 closed-loop pilot loader parity and smoke execution."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compass_benchmark import load_compass_connectome
from p6_closed_loop_pilot import (
    _PortableV3Policy,
    build_p6_arbitration_inputs,
    run_p6_episode,
)
from pilot_train_v3 import _BioRecurrentModel, _GRURecurrentModel


class TestP6ClosedLoopPilot(unittest.TestCase):
    def test_portable_v3_loader_matches_training_models(self):
        weights_path = Path("runs/pilot-v3-models/pilot_weights_v3.npz")
        data_path = Path("runs/pilot-v3-p6/test.npz")
        if not weights_path.exists() or not data_path.exists():
            self.skipTest("Pilot weights or test dataset missing")

        arr = np.load(weights_path, allow_pickle=False)
        test_npz = np.load(data_path, allow_pickle=False)
        x_seq = test_npz["X"][0]  # [24, 24]
        x_norm = ((x_seq - arr["feat_mean"][None, :]) / np.maximum(arr["feat_std"][None, :], 1e-4))[None, :, :]
        dummy_targets = {
            "y_hazard_bin": np.zeros(5, dtype=np.int64),
            "y_risk_horizons": np.zeros((5, 5), dtype=np.float64),
            "y_risk_protected": np.zeros(5, dtype=np.float64),
            "y_progress_protected": np.zeros(5, dtype=np.float64),
            "y_signed_delta_u": np.zeros(5, dtype=np.float64),
        }
        acts = np.arange(5, dtype=np.int64)
        x_batch = np.repeat(x_norm, 5, axis=0)

        for run_name, core_cls, variant in (
            ("FullV3_NeuroCOVER_Bio", _BioRecurrentModel, "full_v3_neurocover"),
            ("FullV3_NeuroCOVER_GRU", _GRURecurrentModel, "full_v3_neurocover"),
        ):
            portable = _PortableV3Policy(weights_path, run_name, variant)
            preds = portable.predict_actions(x_seq)

            prefix = f"{run_name}__"
            if core_cls is _BioRecurrentModel:
                adj = arr[f"{prefix}adj"]
                sign = arr[f"{prefix}sign"]
                model = _BioRecurrentModel(adj, sign, in_dim=24, n_actions=5, seed=0)
                for k in ("theta", "enc_w", "enc_b", "act_emb", "h0_w", "h0_b", "h2_w", "h2_b"):
                    model.params[k] = arr[f"{prefix}{k}"].astype(np.float64)
            else:
                model = _GRURecurrentModel(in_dim=24, hidden_dim=64, n_actions=5, seed=0)
                for k in ("x_w", "h_w", "b", "act_emb", "h0_w", "h0_b", "h2_w", "h2_b"):
                    model.params[k] = arr[f"{prefix}{k}"].astype(np.float64)

            _, out_ref, _ = model.forward_and_grads(x_batch, acts, dummy_targets, variant=variant)
            for idx, act_name in enumerate(("forward", "slow", "brake", "left", "right")):
                got = np.asarray(preds[act_name]["raw_out"], dtype=np.float64)
                np.testing.assert_allclose(got, out_ref[idx], atol=1e-6, rtol=1e-6)

    def test_single24s_vs_multihorizon_arbitration_separation(self):
        # Synthetic action prediction: safe right now (0.3s=0.02) but risky in 2.4s (0.88)
        raw = {
            "forward": {
                "q_hazards": [0.01, 0.01, 0.05, 0.25, 0.88],
                "r_cum": [0.01, 0.02, 0.07, 0.30, 0.91],
                "r_prot": 0.02,
                "progress": 0.28,
                "delta_u": 0.01,
            }
        }
        m4_s, _ = build_p6_arbitration_inputs(raw, "single_24s", use_recovery=False)
        m4_m, m8_m = build_p6_arbitration_inputs(raw, "full_v3_neurocover", use_recovery=True)
        self.assertGreater(m4_s["forward"]["contact_prob"], 0.80)
        self.assertLess(m4_m["forward"]["contact_prob"], 0.05)
        self.assertLess(m8_m["forward"]["contact_prob"], 0.05)

    def test_closed_loop_smoke_episode(self):
        import csv

        weights_path = Path("runs/pilot-v3-models/pilot_weights_v3.npz")
        escape_dir = Path("runs/graph-controls/escape")
        ann_path = Path("runs/connectome/escape/annotations.csv")
        compass_dir = Path("runs/connectome/compass")
        if not weights_path.exists() or not escape_dir.exists():
            self.skipTest("Pilot weights or connectome assets missing")

        with ann_path.open() as f:
            types = {r["root_id"]: r["primary_type"] for r in csv.DictReader(f)}
        ids = (escape_dir / "nodes.txt").read_text().splitlines()
        escape_assets = {
            "npz": escape_dir / "biological.npz",
            "labels": np.array([types[i] for i in ids]),
            "threshold": 0.0238,
        }
        compass_model = load_compass_connectome(compass_dir)
        rec = run_p6_episode(
            scenario="ramp_descent",
            config_name="P6_Full_V3_Bio",
            seed=0,
            escape_assets=escape_assets,
            compass_model=compass_model,
            env_mode="real_world",
            v3_weights_path=weights_path,
        )
        self.assertIn("body_collisions", rec)
        self.assertIn("collision_free_goal_completion", rec)
        self.assertEqual(rec["config"], "P6_Full_V3_Bio")


if __name__ == "__main__":
    unittest.main()
