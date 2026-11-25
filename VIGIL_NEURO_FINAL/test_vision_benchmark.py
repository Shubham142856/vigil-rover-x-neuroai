import json
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np

from vision_benchmark import emd_lplc2_traces, run


class VisionBenchmarkTests(unittest.TestCase):
    def test_synthetic_looming_exceeds_recession_and_translation(self):
        y, x = np.mgrid[-48:48, -64:64]
        def make_clip(mode):
            frames = []
            for t in range(18):
                r = 8 + 1.5 * t if mode == "approach" else (32 - 1.5 * t if mode == "recede" else 14)
                cx = -25 + 3 * t if mode == "translate" else 0
                d = np.hypot(x - cx, y)
                img = np.where(d <= r, 80.0, 240.0)
                frames.append(img)
            return np.stack(frames)

        app = emd_lplc2_traces(make_clip("approach"))[:, 1].max()
        rec = emd_lplc2_traces(make_clip("recede"))[:, 1].max()
        trn = emd_lplc2_traces(make_clip("translate"))[:, 1].max()
        self.assertGreater(app, 0)
        self.assertEqual(rec, 0.0)
        self.assertEqual(trn, 0.0)

    def test_roversim_g04_benchmark_suppresses_controls(self):
        tmp = Path.cwd() / ("vision-bench-" + uuid.uuid4().hex)
        tmp.mkdir()
        self.addCleanup(shutil.rmtree, tmp)
        summary = run(tmp, seed=7)
        self.assertTrue(summary["acceptance_checks"]["approach_is_peak_multiplicative"])
        self.assertTrue(summary["acceptance_checks"]["zero_false_positive_multiplicative_controls"])
