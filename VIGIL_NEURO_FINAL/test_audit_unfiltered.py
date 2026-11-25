import csv
import shutil
import unittest
import uuid
from pathlib import Path

from audit_unfiltered_connectome import audit_unfiltered


class UnfilteredAuditTests(unittest.TestCase):
    def test_single_pass_matches_threshold_one_and_five(self):
        tmp = Path.cwd() / ("audit-" + uuid.uuid4().hex)
        tmp.mkdir()
        self.addCleanup(shutil.rmtree, tmp)
        for grp, ids, nts in [
            ("escape", ["1", "2", "3"], ["ACH", "ACH", ""]),
            ("compass", ["10", "11"], ["ACH", "GLUT"]),
        ]:
            d = tmp / "conn" / grp
            d.mkdir(parents=True)
            (d / "ids.txt").write_text("\n".join(ids) + "\n")
            with (d / "annotations.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["root_id", "primary_type", "side", "predicted_nt", "prediction_score"])
                for i, nt in zip(ids, nts):
                    w.writerow([i, "T", "left", nt, "0.9" if nt else ""])
        csv_path = tmp / "conns.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pre_root_id", "post_root_id", "syn_count"])
            w.writerows([["1", "2", "3"], ["1", "2", "3"], ["2", "3", "2"], ["10", "11", "4"]])
        rep = audit_unfiltered(csv_path, tmp / "conn", tmp / "out")
        self.assertEqual(rep["circuits"]["escape"]["min_1"]["retained_edges"], 2)
        self.assertEqual(rep["circuits"]["escape"]["min_5"]["retained_edges"], 1)
        self.assertEqual(rep["circuits"]["escape"]["min_5"]["isolated_nodes"], 1)
        self.assertEqual(rep["circuits"]["compass"]["min_1"]["retained_edges"], 1)
        self.assertEqual(rep["circuits"]["compass"]["min_5"]["retained_edges"], 0)
        self.assertEqual(rep["circuits"]["escape"]["sign_coverage"]["unknown_nt_nodes"], 1)


if __name__ == "__main__":
    unittest.main()
