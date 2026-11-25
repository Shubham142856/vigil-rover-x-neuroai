import unittest
from baseline_policy import decide


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.clear = dict(frame_valid=True, frame_age_s=0.05, brightness=0.5,
                          contrast=0.2, obstacle_center=0.0, looming_center=0.0,
                          whisker_left=False, whisker_right=False)

    def test_empty_healthy_frame_permits_progress(self):
        self.assertEqual(decide(self.clear)[0], "forward")

    def test_contact_preempts_camera_failure(self):
        for side in ("whisker_left", "whisker_right"):
            self.assertEqual(decide({side: True, "frame_valid": False}), ("brake", "tactile contact"))

    def test_invalid_camera_or_frontal_warning_stops(self):
        for change in ({"frame_valid": False}, {"frame_age_s": 0.5}, {"brightness": 1, "contrast": 0},
                       {"obstacle_center": 0.4}, {"looming_center": 0.2}):
            with self.subTest(change=change):
                self.assertEqual(decide(self.clear | change)[0], "brake")

    def test_missing_stale_or_nonfinite_telemetry_stops(self):
        for key in self.clear:
            data = self.clear.copy()
            del data[key]
            with self.subTest(missing=key):
                self.assertEqual(decide(data)[0], "brake")
        for change in ({"frame_age_s": float("nan")}, {"frame_age_s": -1}, {"looming_center": float("nan")}):
            self.assertEqual(decide(self.clear | change)[0], "brake")
