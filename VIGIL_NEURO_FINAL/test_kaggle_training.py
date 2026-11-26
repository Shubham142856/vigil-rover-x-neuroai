import sys
import unittest
from pathlib import Path

# Add kaggle-training to sys.path
sys.path.insert(0, str(Path(__file__).parent / "kaggle-training"))
from train_neuro_cover import resolve_inputs, resolve_output_dir, main


class TestKaggleTrainingNotebookCompatibility(unittest.TestCase):
    def test_jupyter_kernel_argv_does_not_crash(self):
        # Simulate what Jupyter/Colab kernel injects into sys.argv
        colab_argv = [
            "-f",
            "/root/.local/share/jupyter/runtime/kernel-53b2dbc6-1871-49ef-a86f-051ba89b91f6.json",
        ]
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--data", type=Path, default=None)
        p.add_argument("--bio", type=Path, default=None)
        p.add_argument("--out", type=Path, default=None)
        p.add_argument("--epochs", type=int, default=50)

        # Ensure parse_known_args ignores -f without SystemExit
        a, extra = p.parse_known_args(colab_argv)
        self.assertEqual(a.epochs, 50)
        self.assertIn("-f", extra)

    def test_resolve_output_dir(self):
        tmp_out = Path("runs/test-out-dir")
        out = resolve_output_dir(tmp_out)
        self.assertTrue(out.exists())
        # Clean up
        if tmp_out.exists():
            tmp_out.rmdir()

    def test_resolve_inputs_local(self):
        data_d, bio_p, rew_p, ann_p, src = resolve_inputs(None, None, None, None)
        self.assertTrue(data_d.exists(), f"Dataset dir {data_d} not found")
        self.assertTrue(bio_p.exists(), f"Bio npz {bio_p} not found")
        self.assertTrue(rew_p.exists(), f"Rewired npz {rew_p} not found")
        self.assertTrue(ann_p.exists(), f"Annotations csv {ann_p} not found")


if __name__ == "__main__":
    unittest.main()
