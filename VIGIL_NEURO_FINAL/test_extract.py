import csv
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

from extract_graph import extract


class TestExtract(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.gettempdir()) / f"extract-test-{uuid.uuid4().hex}"
        self.directory.mkdir()
        self.addCleanup(shutil.rmtree, self.directory)
        self.ids = self.directory / "ids.txt"
        self.ids.write_text("1\n2\n3\n", encoding="utf-8")
        self.connections = self.directory / "connections.csv"
        self.out = self.directory / "out"

    def extract_rows(self, text, minimum=1, **columns):
        self.connections.write_text(text, encoding="utf-8")
        options = {"pre": "pre", "post": "post", "count": "n", **columns}
        return extract(self.ids, self.connections, self.out, minimum=minimum,
                       version="synthetic", **options)

    def edges(self):
        with (self.out / "edges.csv").open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_pool_before_threshold_and_preserve_direction(self):
        manifest = self.extract_rows(
            "pre,post,n,neuropil\n1,2,2,A\n1,2,3,B\n2,1,4,A\n2,9,99,A\n",
            minimum=5,
        )
        self.assertEqual(self.edges(), [{"pre_id": "1", "post_id": "2", "synapse_count": "5"}])
        self.assertEqual(manifest["retained_edges"], 1)
        self.assertEqual(manifest["isolated_nodes"], 1)
        self.assertEqual(manifest["source_rows"], 3)
        self.assertEqual((self.out / "nodes.txt").read_text().splitlines(), ["1", "2", "3"])

    def test_nonpositive_minimum_rejected(self):
        for minimum in (0, -1):
            with self.subTest(minimum=minimum), self.assertRaisesRegex(ValueError, "minimum"):
                self.extract_rows("pre,post,n\n1,2,0\n", minimum=minimum)

    def test_zero_counts_do_not_form_edges_or_remove_isolates(self):
        manifest = self.extract_rows("pre,post,n\n1,2,1\n3,1,0\n")
        self.assertEqual(manifest["source_rows"], 2)
        self.assertEqual(manifest["isolated_nodes"], 1)
        self.assertEqual(self.edges(), [{"pre_id": "1", "post_id": "2", "synapse_count": "1"}])

    def test_ambiguous_headers_rejected(self):
        with self.assertRaisesRegex(ValueError, "(?i)duplicate"):
            self.extract_rows("pre,post,n,n\n1,2,1,4\n")

    def test_columns_must_have_distinct_roles(self):
        with self.assertRaisesRegex(ValueError, "(?i)distinct"):
            self.extract_rows("pre,post,n\n1,2,4\n", post="pre")

    def test_bad_counts_and_missing_columns_rejected(self):
        for text in ("pre,post,n\n1,2,-1\n", "pre,post,n\n1,2,4.5\n", "pre,post\n1,2\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.extract_rows(text)

    def test_empty_induced_graph_rejected(self):
        with self.assertRaisesRegex(ValueError, "No retained edges"):
            self.extract_rows("pre,post,n\n1,9,7\n")

    def test_cli_gzip_exact_ids_and_hashes(self):
        a, b = "9007199254740992", "9007199254740993"
        self.ids.write_text(f"{a}\n{b}\n", encoding="utf-8-sig")
        compressed = self.connections.with_suffix(".csv.gz")
        with gzip.open(compressed, "wt", encoding="utf-8-sig", newline="") as stream:
            stream.write(f"pre,post,n\n{a},{b},4\n{b},{a},8\n")
        result = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).with_name("extract_graph.py")),
             "--ids", str(self.ids), "--connections", str(compressed),
             "--pre", "pre", "--post", "post", "--count", "n",
             "--out", str(self.out), "--version", "synthetic"],
            capture_output=True, text=True, check=True,
        )
        manifest = json.loads(result.stdout)
        self.assertEqual(manifest, json.loads((self.out / "manifest.json").read_text()))
        self.assertEqual(manifest["connection_sha256"], hashlib.sha256(compressed.read_bytes()).hexdigest())
        self.assertEqual(manifest["ids_sha256"], hashlib.sha256(self.ids.read_bytes()).hexdigest())
        self.assertEqual(self.edges(), [
            {"pre_id": a, "post_id": b, "synapse_count": "4"},
            {"pre_id": b, "post_id": a, "synapse_count": "8"},
        ])


if __name__ == "__main__":
    unittest.main()
