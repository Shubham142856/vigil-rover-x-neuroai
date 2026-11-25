import unittest
import numpy as np
from scipy.sparse import csr_matrix
from circuit_response import response


class CircuitTests(unittest.TestCase):
    def test_direction_delay_and_relaxation(self):
        w=csr_matrix([[0,0],[3,0]])
        pre=np.array([True,False])
        result=response(w,pre,~pre)
        self.assertEqual(result[0],0)
        self.assertGreater(result[1],0)
        self.assertLess(result[-1],result.max()/100)
        np.testing.assert_array_equal(response(w,~pre,pre),np.zeros(60))
