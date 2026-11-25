import unittest
from collections import Counter
from graph_controls import rewire


class GraphControlTests(unittest.TestCase):
    def test_directed_degrees_and_magnitudes(self):
        edges=[(i,(i+1)%12,i+1) for i in range(12)]+[(i,(i+3)%12,2) for i in range(12)]
        changed=rewire(edges,17)
        self.assertEqual(changed,rewire(edges,17))
        self.assertEqual(Counter((i,w) for i,j,w in edges),Counter((i,w) for i,j,w in changed))
        self.assertEqual(Counter(j for i,j,w in edges),Counter(j for i,j,w in changed))
        self.assertEqual(len({(i,j) for i,j,w in changed}),len(edges))
        self.assertTrue(all(i!=j for i,j,w in changed))
        self.assertNotEqual(set(edges),set(changed))
