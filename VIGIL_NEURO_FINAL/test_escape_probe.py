import unittest

from escape_probe import probe


class ProbeTest(unittest.TestCase):
    def test_only_incoming_visual_edges_contribute(self):
        cells = {"10": ("LC4","left"), "11": ("LPLC2","left"), "20": ("DNp01","left")}
        rows = [("10","20",5), ("11","20",15), ("20","10",100), ("11","10",50)]
        result = probe(cells, rows)[0]
        self.assertEqual(result["LC4_contacts"], 5)
        self.assertEqual(result["LPLC2_contacts"], 15)
        self.assertEqual(result["LC4_unit_drive"], 0.25)
        self.assertEqual(result["LPLC2_unit_drive"], 0.75)

    def test_disconnected_giant_fiber_is_reported(self):
        result = probe({"20": ("DNp01","right")}, [])[0]
        self.assertEqual(result["LC4_sources"], 0)
        self.assertIsNone(result["LC4_unit_drive"])
