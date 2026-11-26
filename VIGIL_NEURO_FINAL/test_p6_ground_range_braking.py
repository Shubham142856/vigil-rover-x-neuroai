"""P6.3 & P6.4 Analytical and PyBullet Verification Tests:
1. Calibrated Ground-Contact IPM Range Accuracy for tall (1.2 m) vs short (0.24 m) obstacles.
2. Stopping-Distance Conservation against actual PyBullet RoverSim & RealWorldEnv braking trajectories.
"""
import unittest

import numpy as np

from ground_range_estimator import AdaptiveBrakingModel, estimate_ground_contact_clearance
from real_world_env import RealWorldEnv
from rover_sim import RoverSim


class TestP6GroundRangeAndBraking(unittest.TestCase):
    def test_ipm_distinguishes_tall_distant_pedestrian_from_close_obstacle(self):
        # 1. Tall pedestrian at x = 1.05 m (corridor_pedestrian) must NOT register within 0.25 m
        with RealWorldEnv(scene="corridor_pedestrian", seed=0, sensor_noise=False) as env:
            obs = env.observe()
            est_ped = estimate_ground_contact_clearance(obs["rgb"], pitch_rad=0.0)
        self.assertTrue(est_ped["obstacle_detected"])
        self.assertGreater(est_ped["estimated_clearance_m"], 0.45, f"Tall distant pedestrian misestimated close: {est_ped}")

        # 2. Short post at close range x = 0.35 m vs far range x = 0.85 m must separate cleanly
        ranges = {}
        for x_true in (0.35, 0.55, 0.85):
            with RoverSim(scene="open", seed=7) as sim:
                sim.solids.append(sim.box([x_true, 0.0, 0.12], [0.03, 0.03, 0.12], 0, [0.86, 0.17, 0.09, 1]))
                sim.packet = None
                sim.queue.clear()
                sim.update_camera()
                est = estimate_ground_contact_clearance(sim.observe()["rgb"], pitch_rad=0.0)
                ranges[x_true] = est["estimated_range_m"]
        self.assertLess(ranges[0.35], ranges[0.55])
        self.assertLess(ranges[0.55], ranges[0.85])
        self.assertLess(abs(ranges[0.55] - 0.55), 0.15, f"IPM error too high at 0.55m: {ranges}")

    def test_empty_open_scene_reports_clear_range(self):
        with RealWorldEnv(scene="open", seed=1, sensor_noise=True) as env:
            est = estimate_ground_contact_clearance(env.observe()["rgb"])
        self.assertGreater(est["estimated_clearance_m"], 1.5)

    def test_adaptive_braking_envelope_bounds_pybullet_stopping_distance(self):
        bm = AdaptiveBrakingModel()
        # Measure actual braking roll distances in RoverSim and RealWorldEnv (low friction mu=0.45)
        for cls, sc, mu in [
            (RoverSim, "open", 0.60),
            (RealWorldEnv, "open", 0.60),
            (RealWorldEnv, "outdoor_pothole", 0.45),
        ]:
            with cls(scene=sc, seed=2) as sim:
                for _ in range(15):
                    sim.step("forward")
                ev0 = sim.evaluator()
                v0 = ev0["speed_m_s"]
                x0 = ev0["x_m"]
                for _ in range(30):
                    sim.step("brake")
                dx_actual = sim.evaluator()["x_m"] - x0
                d_pred = bm.stopping_distance_m(v0, friction_mu=mu)
                self.assertGreaterEqual(
                    d_pred,
                    dx_actual,
                    f"{cls.__name__}({sc}, mu={mu}): actual stop {dx_actual:.3f}m exceeded envelope {d_pred:.3f}m at v0={v0:.3f}",
                )


if __name__ == "__main__":
    unittest.main()
