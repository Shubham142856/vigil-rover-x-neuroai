import shutil
import unittest
import uuid
from pathlib import Path

from closed_loop_benchmark import run


class ClosedLoopTests(unittest.TestCase):
    def test_closed_loop_ablation_zero_body_collisions_and_abstention(self):
        if not Path("runs/graph-controls/escape/biological.npz").exists():
            self.skipTest("Extracted escape biological graph missing")
        tmp = Path.cwd() / ("closed-loop-" + uuid.uuid4().hex)
        tmp.mkdir()
        self.addCleanup(shutil.rmtree, tmp)
        summary = run(tmp, seeds=2)
        for policy in ("recovery_off", "ordinary_recovery", "circuit_recovery", "neuro_cover_l"):
            self.assertEqual(summary["policies"][policy]["total_body_collisions"], 0)
            self.assertGreater(summary["policies"][policy]["abstentions"], 0)
        # Circuit-conditioned policy avoids false stops on open & near-miss scenes
        self.assertEqual(summary["policies"]["circuit_recovery"]["false_emergency_stops"], 0)
        self.assertEqual(summary["policies"]["neuro_cover_l"]["false_emergency_stops"], 0)


if __name__ == "__main__":
    unittest.main()
