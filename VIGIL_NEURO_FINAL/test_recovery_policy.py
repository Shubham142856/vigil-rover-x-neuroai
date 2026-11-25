import shutil
import unittest
import uuid
from pathlib import Path

from recovery_policy import choose_recovery_action, evaluate_tactile_scenarios


class RecoveryPolicyTests(unittest.TestCase):
    def test_safe_abstention_when_opposite_sector_unverified_or_blocked(self):
        obs = {"whisker_left": True, "whisker_right": False}
        blocked = {
            "camera_healthy": False,
            "sector_states": ["occupied", "occupied", "stale_or_unknown", "occupied", "observed_free"],
        }
        action, reason = choose_recovery_action(obs, blocked)
        self.assertEqual(action, "brake")
        self.assertIn("abstain", reason)

    def test_verified_clear_escape_during_camera_outage(self):
        obs = {"whisker_left": True, "whisker_right": False}
        clear_right = {
            "camera_healthy": False,
            "sector_states": ["occupied", "occupied", "observed_free", "observed_free", "observed_free"],
        }
        action, _ = choose_recovery_action(obs, clear_right)
        self.assertEqual(action, "right")

    def test_tactile_scenarios_zero_body_collisions(self):
        tmp = Path.cwd() / ("recovery-" + uuid.uuid4().hex)
        tmp.mkdir()
        self.addCleanup(shutil.rmtree, tmp)
        res = evaluate_tactile_scenarios(tmp)
        self.assertEqual(res["bilateral_wall"]["final"]["body_contact_events"], 0)
        self.assertTrue(res["onesided_blocked_outage"]["abstained_safely"])
        self.assertEqual(res["onesided_blocked_outage"]["final"]["body_contact_events"], 0)
        self.assertTrue(res["onesided_verified_clear_outage"]["escaped_clear"])
        self.assertEqual(res["onesided_verified_clear_outage"]["final"]["body_contact_events"], 0)


if __name__ == "__main__":
    unittest.main()
