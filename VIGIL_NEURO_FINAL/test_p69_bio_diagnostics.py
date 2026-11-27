"""Unit tests for P6.9.5 Modular Adaptive Bio & Modular GRU analytical BPTT gradients."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from p69_bio_diagnostics import _ModularAdaptiveBioModel, _ModularGRUModel


class TestP69BioDiagnostics(unittest.TestCase):
    def _synthetic_batch(self):
        rng = np.random.default_rng(19)
        x = rng.normal(0.0, 1.0, (6, 8, 24))
        act = rng.integers(0, 5, size=6)
        targets = {
            "y_hazard_bin": rng.integers(0, 6, size=6),
            "y_risk_protected": rng.integers(0, 2, size=6).astype(np.float64),
            "y_progress_protected": rng.normal(0.15, 0.1, size=6),
            "y_signed_delta_u": rng.normal(0.05, 0.25, size=6),
        }
        return x, act, targets

    def test_modular_adaptive_bio_bptt_matches_finite_difference(self):
        x, act, targets = self._synthetic_batch()
        adj = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.float64)
        sgn = np.array([[1, -1, 1, -1]] * 4, dtype=np.float64)
        mod = _ModularAdaptiveBioModel(adj, sgn, in_dim=24, n_actions=5, seed=3)
        _, _, grads = mod.forward_and_grads(x, act, targets, variant="full_v3_neurocover")

        eps = 1e-6
        for pname in ("theta", "tau_logit", "enc_w", "hr0_w", "hu2_w", "act_emb"):
            param = mod.params[pname]
            it = np.ndindex(param.shape)
            checked = 0
            for idx in it:
                orig = param[idx]
                param[idx] = orig + eps
                lp, _, _ = mod.forward_and_grads(x, act, targets, variant="full_v3_neurocover")
                param[idx] = orig - eps
                lm, _, _ = mod.forward_and_grads(x, act, targets, variant="full_v3_neurocover")
                param[idx] = orig
                fd = (lp - lm) / (2.0 * eps)
                self.assertAlmostEqual(grads[pname][idx], fd, places=5, msg=f"Grad mismatch {pname}{idx}")
                checked += 1
                if checked >= 6:
                    break

    def test_modular_gru_bptt_matches_finite_difference(self):
        x, act, targets = self._synthetic_batch()
        mod = _ModularGRUModel(in_dim=24, hidden_dim=8, n_actions=5, seed=5)
        _, _, grads = mod.forward_and_grads(x, act, targets, variant="full_v3_neurocover")

        eps = 1e-6
        for pname in ("x_w", "h_w", "b", "hr0_w", "hu2_w", "act_emb"):
            param = mod.params[pname]
            checked = 0
            for idx in np.ndindex(param.shape):
                orig = param[idx]
                param[idx] = orig + eps
                lp, _, _ = mod.forward_and_grads(x, act, targets, variant="full_v3_neurocover")
                param[idx] = orig - eps
                lm, _, _ = mod.forward_and_grads(x, act, targets, variant="full_v3_neurocover")
                param[idx] = orig
                fd = (lp - lm) / (2.0 * eps)
                self.assertAlmostEqual(grads[pname][idx], fd, places=5, msg=f"GRU grad mismatch {pname}{idx}")
                checked += 1
                if checked >= 6:
                    break


if __name__ == "__main__":
    unittest.main()
