import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np

from compass_benchmark import GRUCompass, load_compass_connectome, run


class CompassBenchmarkTests(unittest.TestCase):
    def test_gru_bptt_gradient_matches_finite_difference(self):
        gru = GRUCompass(hidden=4, seed=2)
        rng = np.random.default_rng(1)
        X = rng.normal(size=(2, 5, 4))
        Y = rng.normal(size=(2, 5, 2))
        _, grad = gru.loss_and_grad(gru.params, X, Y)
        eps = 1e-6
        for idx in (0, 15, len(gru.params) - 1):
            p_plus = gru.params.copy(); p_plus[idx] += eps
            p_minus = gru.params.copy(); p_minus[idx] -= eps
            fd = (gru.loss_and_grad(p_plus, X, Y)[0] - gru.loss_and_grad(p_minus, X, Y)[0]) / (2 * eps)
            self.assertAlmostEqual(grad[idx], fd, places=6)

    def test_fafb_compass_connectome_and_benchmark(self):
        compass_dir = Path("runs/connectome/compass")
        if not compass_dir.exists():
            self.skipTest("Extracted FAFB compass graph not present")
        model = load_compass_connectome(compass_dir)
        # Opposite PEN_a hemispheres induce opposite-sign phase shifts
        self.assertLess(model["pen_a_left_shift_deg"] * model["pen_a_right_shift_deg"], 0.0)
        tmp = Path.cwd() / ("compass-bench-" + uuid.uuid4().hex)
        tmp.mkdir()
        self.addCleanup(shutil.rmtree, tmp)
        summary = run(compass_dir, tmp, seeds=3)
        self.assertGreater(summary["bump_stability"]["mean"], 0.7)
        self.assertEqual(summary["bump_stability"]["missing_estimates"], 0)
        # Connectome ring resists 1-frame conflicting distractor better than naive Bayes
        ring_conf = summary["mean_abs_error_deg"]["connectome_ring"]["conflict"]
        bayes_conf = summary["mean_abs_error_deg"]["bayesian"]["conflict"]
        self.assertLess(ring_conf, bayes_conf)


if __name__ == "__main__":
    unittest.main()
