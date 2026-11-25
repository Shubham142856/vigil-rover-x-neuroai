import unittest
import numpy as np
from heading_memory import advance, decode, wrap


class HeadingTests(unittest.TestCase):
    def test_single_bin_shift_crosses_angle_boundary(self):
        angles=np.linspace(-np.pi,np.pi,72,endpoint=False)
        mass=np.zeros(72);mass[-1]=1
        result=advance(mass,2*np.pi/72)
        self.assertAlmostEqual(wrap(decode(result,angles)-angles[0]),0)
        np.testing.assert_allclose(advance(result,-2*np.pi/72),mass,atol=1e-14)

    def test_ambiguous_heading_and_fractional_motion(self):
        self.assertIsNone(decode(np.ones(72)/72,np.linspace(-np.pi,np.pi,72,endpoint=False)))
        mass=np.array([1.,0.,0.,0.])
        np.testing.assert_allclose(advance(mass,-np.pi/4),[.5,0,0,.5])
