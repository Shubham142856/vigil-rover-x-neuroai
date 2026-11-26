"""Analytical vs finite-difference BPTT gradient parity & survival monotonicity tests for pilot_train_v3."""
import unittest

import numpy as np

from pilot_train_v3 import _BioRecurrentModel, _GRURecurrentModel, _compute_cumulative_risks


class TestPilotTrainV3Gradients(unittest.TestCase):
    def _dummy_batch(self, seq_len=6, in_dim=8, bsz=4, seed=11):
        rng = np.random.default_rng(seed)
        X = rng.normal(0.0, 0.5, (bsz, seq_len, in_dim))
        act = rng.integers(0, 5, size=bsz)
        targets = {
            "y_hazard_bin": np.array([1, 3, 5, 0], dtype=np.int64),
            "y_risk_horizons": np.array([
                [0, 1, 1, 1, 1],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
            ], dtype=np.float64),
            "y_risk_protected": np.array([1.0, 0.0, 0.0, 1.0]),
            "y_progress_protected": np.array([0.1, 0.3, 0.4, 0.05]),
            "y_signed_delta_u": np.array([0.2, -0.1, 0.0, 0.35]),
        }
        return X, act, targets

    def test_bio_and_gru_analytical_gradients_match_central_finite_differences(self):
        rng = np.random.default_rng(3)
        adj = (rng.random((6, 6)) > 0.4).astype(np.float64)
        sgn = np.where(rng.random((6, 6)) > 0.5, 1.0, -1.0)
        X, act, targets = self._dummy_batch(seq_len=5, in_dim=8, bsz=4)
        eps = 1e-5

        for model in (
            _BioRecurrentModel(adj, sgn, in_dim=8, seed=5),
            _GRURecurrentModel(in_dim=8, hidden_dim=6, seed=5),
        ):
            loss0, out0, grads = model.forward_and_grads(X, act, targets, "full_v3_neurocover")
            self.assertTrue(np.isfinite(loss0))
            r_cum = _compute_cumulative_risks(out0)
            self.assertTrue(np.all(np.diff(r_cum, axis=1) >= -1e-12))

            # Verify finite-difference parity on representative parameters across core & head
            check_keys = ("theta", "enc_w", "h0_w", "h2_w", "act_emb") if model.core_type == "bio" else ("x_w", "h_w", "b", "h0_w", "h2_w", "act_emb")
            for key in check_keys:
                arr = model.params[key]
                g_ana = grads[key]
                for idx_flat in (0, min(5, arr.size - 1), arr.size - 1):
                    idx = np.unravel_index(idx_flat, arr.shape)
                    orig = float(arr[idx])
                    arr[idx] = orig + eps
                    lp, _, _ = model.forward_and_grads(X, act, targets, "full_v3_neurocover")
                    arr[idx] = orig - eps
                    lm, _, _ = model.forward_and_grads(X, act, targets, "full_v3_neurocover")
                    arr[idx] = orig
                    num_grad = (lp - lm) / (2.0 * eps)
                    ana_grad = float(g_ana[idx])
                    self.assertAlmostEqual(
                        ana_grad,
                        num_grad,
                        delta=max(1e-4, 2e-3 * abs(num_grad)),
                        msg=f"Gradient mismatch {model.core_type}.{key}[{idx}]: ana={ana_grad} vs num={num_grad}",
                    )


if __name__ == "__main__":
    unittest.main()
