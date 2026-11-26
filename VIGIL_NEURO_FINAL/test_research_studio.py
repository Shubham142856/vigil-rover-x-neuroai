import shutil
import unittest
import uuid
from pathlib import Path

from research_studio import build_studio, verify_replay_bundle


class ResearchStudioTests(unittest.TestCase):
    def test_recorded_runs_replay_identically_and_build_studio(self):
        runs = Path("runs")
        if not (runs / "my-first-drive/replay.jsonl").exists():
            self.skipTest("Recorded runs missing")
        bundle = verify_replay_bundle(runs / "my-first-drive")
        self.assertTrue(bundle["verified"])
        self.assertEqual(bundle["steps"], 81)

        tmp = Path.cwd() / ("studio-" + uuid.uuid4().hex)
        tmp.mkdir()
        self.addCleanup(shutil.rmtree, tmp)
        manifest = build_studio(runs, Path("VIGIL_NEURO_FINAL/Implementation_Acceptance_Gates.csv"), tmp)
        self.assertTrue((tmp / "index.html").exists())
        self.assertIn("my-first-drive", manifest["replays_verified"])


if __name__ == "__main__":
    unittest.main()
