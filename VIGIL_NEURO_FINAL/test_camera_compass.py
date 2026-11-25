import unittest
import numpy as np
from camera_compass import bearing, fresh_heading


class CameraCompassTests(unittest.TestCase):
    def test_centered_and_displaced_marker(self):
        rgb=np.zeros((96,128,3),dtype=np.uint8)
        self.assertIsNone(bearing(rgb))
        rgb[30:60,63:65,0]=200
        self.assertAlmostEqual(bearing(rgb),0)
        self.assertGreater(bearing(np.roll(rgb,10,axis=1)),0)

    def test_repeated_and_delayed_packets_do_not_update(self):
        packet=dict(frame_valid=True,capture_s=0.5,time_s=0.5,rgb=np.zeros((96,128,3),dtype=np.uint8))
        packet["rgb"][30:60,63:65,0]=200
        self.assertIsNone(fresh_heading(packet,0.5))
        self.assertIsNone(fresh_heading(dict(packet,time_s=1),0.4))
        self.assertAlmostEqual(fresh_heading(packet,0.4),0)
