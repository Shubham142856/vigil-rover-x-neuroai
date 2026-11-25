import unittest
import numpy as np
from camera_escape import visual_drive, visual_looming, choose_threshold


class EscapeCameraTests(unittest.TestCase):
    def test_approach_produces_dynamic_drive_only_when_growing(self):
        a = np.zeros((96, 128, 3), dtype=np.uint8)
        a[45:51, 60:68, 0] = 200
        b = a.copy()
        b[35:61, 52:76, 0] = 200
        size, growth = visual_drive(b, 0.0, 0.1)
        self.assertGreater(size, 0)
        self.assertGreater(growth, 0)
        self.assertEqual(visual_drive(b, size, 0.1)[1], 0)
        self.assertEqual(visual_drive(a, size, 0.1)[1], 0)

    def test_looming_rejects_lateral_near_miss_drift(self):
        centered0 = np.zeros((96, 128, 3), dtype=np.uint8)
        centered0[42:54, 60:68, 0] = 200
        centered1 = np.zeros((96, 128, 3), dtype=np.uint8)
        centered1[36:60, 54:74, 0] = 200
        _, _, st = visual_looming(centered0, None, 0.1)
        c_size, c_growth, _ = visual_looming(centered1, st, 0.1)

        lat0 = np.zeros((96, 128, 3), dtype=np.uint8)
        lat0[42:54, 92:98, 0] = 200
        lat1 = np.zeros((96, 128, 3), dtype=np.uint8)
        lat1[40:56, 108:116, 0] = 200
        _, _, lst = visual_looming(lat0, None, 0.1)
        l_size, l_growth, _ = visual_looming(lat1, lst, 0.1)

        self.assertGreater(c_size, 5 * l_size)
        self.assertGreater(c_growth, 5 * l_growth)

    def test_threshold_uses_both_event_classes_with_zero_fp(self):
        t = choose_threshold([(0.1, False), (0.25, False), (0.3, True), (0.7, True)])
        self.assertGreater(t, 0.25)
        self.assertLessEqual(t, 0.3)
        # Earliest threshold above negatives should be preferred over highest (0.7)
        self.assertLess(t, 0.5)
