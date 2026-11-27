"""Tests for P6.9.6 & P6.9.7 modular ensemble loader parity & smoke evaluation."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from p69_bio_diagnostics import _ModularAdaptiveBioModel, _ModularGRUModel
from p69_matched_eval import _EnsembleP69Policy, _PortableP69Policy


class TestP69MatchedEval(unittest.TestCase):
    def test_portable_modular_bio_and_gru_match_training_forward(self):
        rng = np.random.default_rng(11)
        x_seq = rng.normal(0.0, 1.0, (24, 24)).astype(np.float32)
        mu = rng.normal(0.0, 0.2, 24).astype(np.float32)
        sd = rng.uniform(0.5, 1.5, 24).astype(np.float32)
        x_norm = ((x_seq - mu[None, :]) / sd[None, :])[None, :, :]
        x_batch = np.repeat(x_norm, 5, axis=0)
        acts = np.arange(5, dtype=np.int64)
        dummy_t = {
            "y_hazard_bin": np.zeros(5, dtype=np.int64),
            "y_risk_horizons": np.zeros((5, 5), dtype=np.float64),
            "y_risk_protected": np.zeros(5, dtype=np.float64),
            "y_progress_protected": np.zeros(5, dtype=np.float64),
            "y_signed_delta_u": np.zeros(5, dtype=np.float64),
        }

        adj = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.float64)
        sgn = np.array([[1, -1, 1, -1]] * 4, dtype=np.float64)
        mod_bio = _ModularAdaptiveBioModel(adj, sgn, in_dim=24, seed=7)
        mod_gru = _ModularGRUModel(in_dim=24, hidden_dim=8, seed=9)

        arr = {"feat_mean": mu, "feat_std": sd}
        for name, mod in (("B", mod_bio), ("G", mod_gru)):
            arr[f"{name}__core_type"] = np.array([mod.core_type])
            arr[f"{name}__belief_skip_dim"] = np.array([mod.belief_skip_dim], dtype=np.int64)
            if hasattr(mod, "adj"):
                arr[f"{name}__adj"] = mod.adj.astype(np.float32)
                arr[f"{name}__sign"] = mod.sign.astype(np.float32)
            for k, v in mod.params.items():
                arr[f"{name}__{k}"] = v.astype(np.float32)

        for name, mod in (("B", mod_bio), ("G", mod_gru)):
            pol = _PortableP69Policy(arr, name, "full_v3_neurocover")
            preds = pol.predict_actions(x_seq)
            _, ref, _ = mod.forward_and_grads(x_batch, acts, dummy_t, variant="full_v3_neurocover")
            for idx, a_name in enumerate(("forward", "slow", "brake", "left", "right")):
                np.testing.assert_allclose(preds[a_name]["raw_out"], ref[idx], atol=1e-5, rtol=1e-5)

        # Ensemble spread check across 2 perturbed copies
        arr2 = dict(arr)
        arr2["B2__core_type"] = arr["B__core_type"]
        arr2["B2__belief_skip_dim"] = arr["B__belief_skip_dim"]
        arr2["B2__adj"] = arr["B__adj"]
        arr2["B2__sign"] = arr["B__sign"]
        for k, v in mod_bio.params.items():
            arr2[f"B2__{k}"] = (v + (0.2 if k == "hu2_b" else 0.0)).astype(np.float32)
        ens = _EnsembleP69Policy([_PortableP69Policy(arr2, "B", "full_v3_neurocover"), _PortableP69Policy(arr2, "B2", "full_v3_neurocover")])
        ep = ens.predict_actions(x_seq)
        self.assertGreater(ep["forward"]["delta_u_std"], 0.05)


if __name__ == "__main__":
    unittest.main()
