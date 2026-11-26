"""Analytical, color-agnostic, and ego-motion verification tests for T4/T5 dual-pathway vision."""
import math
import unittest

import numpy as np
import pybullet as pb

from real_sensor_model import CameraParams, RealSensorModel
from real_world_env import RealWorldEnv
from rover_sim import RoverSim
from t4_t5_vision import (
    DEFAULT_CX,
    DEFAULT_CY,
    DEFAULT_FX,
    DEFAULT_FY,
    DualPathwayT4T5Vision,
    to_normalized_luminance,
    warp_rotation_homography,
)


class TestT4T5VisionAndSensorCalibration(unittest.TestCase):
    def test_camera_params_match_pybullet_projection(self):
        fy_true = 96.0 / (2.0 * math.tan(math.radians(66.0) / 2.0))
        p = CameraParams()
        self.assertAlmostEqual(p.fx, fy_true, delta=1e-6)
        self.assertAlmostEqual(p.fy, fy_true, delta=1e-6)
        self.assertAlmostEqual(p.cx, 63.5, delta=1e-6)
        self.assertAlmostEqual(p.cy, 47.5, delta=1e-6)

    def test_synthetic_rotation_homography_reduces_yaw_motion_energy(self):
        # Generate vertical checkerboard pattern; rotate camera right/left cleanly via known d_yaw
        h, w = 96, 128
        xs = np.arange(w)[None, :]
        ys = np.arange(h)[:, None]
        tex0 = 0.5 + 0.45 * np.sin(0.35 * xs) * np.cos(0.35 * ys)
        d_yaw = 0.05  # 1 rad/s * 0.05s
        tex1 = warp_rotation_homography(tex0, delta_yaw_rad=d_yaw, delta_pitch_rad=0.0)
        tex2 = warp_rotation_homography(tex1, delta_yaw_rad=d_yaw, delta_pitch_rad=0.0)
        uncomp_err = float(np.mean(np.abs(tex1[10:-10, 15:-15] - tex0[10:-10, 15:-15])))

        rgb0 = np.stack([tex0 * 255] * 3, axis=-1).astype(np.uint8)
        rgb1 = np.stack([tex1 * 255] * 3, axis=-1).astype(np.uint8)
        rgb2 = np.stack([tex2 * 255] * 3, axis=-1).astype(np.uint8)
        vis = DualPathwayT4T5Vision()
        vis.step(rgb0, dt=0.05, gyro_z_rad_s=0.0)
        vis.step(rgb1, dt=0.05, gyro_z_rad_s=1.0)
        out = vis.step(rgb2, dt=0.05, gyro_z_rad_s=1.0)

        self.assertGreater(uncomp_err, 0.05)
        self.assertGreater(out["raw_t4"] + out["raw_t5"], 0.05)
        # Residual ego-motion path should cancel pure rotation by > 90%
        self.assertLess(
            out["residual_t4"] + out["residual_t5"],
            0.10 * (out["raw_t4"] + out["raw_t5"]),
            f"Rotation homography failed to suppress spurious yaw EMD: {out}",
        )

    def test_color_agnostic_gray_approach_activates_lplc2_while_yaw_and_empty_stay_quiet(self):
        # Verify non-red gray obstacle on collision course triggers LPLC2/LC4 (unlike red-only detectors)
        def run_regime(regime):
            vis = DualPathwayT4T5Vision()
            peaks = {"raw_lplc2": 0.0, "res_lplc2": 0.0, "res_t5": 0.0, "raw_t5": 0.0}
            with RoverSim(scene="open", seed=7) as sim:
                if regime == "gray_approach":
                    # Add non-red neutral gray obstacle directly ahead (red-threshold detectors miss it)
                    sim.solids.append(sim.box([0.62, 0.0, 0.12], [0.03, 0.04, 0.12], 0, [0.35, 0.35, 0.35, 1]))
                elif regime == "gray_yaw":
                    sim.solids.append(sim.box([0.62, 0.0, 0.12], [0.03, 0.04, 0.12], 0, [0.35, 0.35, 0.35, 1]))
                elif regime == "gray_lateral":
                    post = sim.box([0.55, -0.30, 0.12], [0.03, 0.04, 0.12], 0, [0.35, 0.35, 0.35, 1])
                    sim.solids.append(post)
                sim.packet = None
                sim.queue.clear()
                sim.update_camera()
                last_cap = None
                for step in range(30):
                    obs = sim.observe()
                    if obs["frame_valid"] and obs["capture_s"] != last_cap:
                        gz = 1.0 if regime == "gray_yaw" and last_cap is not None else 0.0
                        out = vis.step(obs["rgb"], dt=0.1, gyro_z_rad_s=gz)
                        peaks["raw_lplc2"] = max(peaks["raw_lplc2"], out["raw_lplc2"])
                        peaks["res_lplc2"] = max(peaks["res_lplc2"], out["residual_lplc2"])
                        peaks["raw_t5"] = max(peaks["raw_t5"], out["raw_t5"])
                        peaks["res_t5"] = max(peaks["res_t5"], out["residual_t5"])
                        last_cap = obs["capture_s"]
                    if regime == "gray_yaw":
                        sim.step("left")
                    elif regime == "gray_lateral":
                        pb.resetBasePositionAndOrientation(post, [0.55, -0.30 + 0.02 * (step + 1), 0.12], [0, 0, 0, 1], physicsClientId=sim.client)
                        sim.step("brake")
                    else:
                        sim.step("forward")
            return peaks

        p_app = run_regime("gray_approach")
        p_yaw = run_regime("gray_yaw")
        p_lat = run_regime("gray_lateral")
        p_empty = run_regime("empty")

        self.assertGreater(p_app["raw_lplc2"], 0.0, f"Gray obstacle approach must activate LPLC2: {p_app}")
        self.assertGreater(p_app["res_lplc2"], p_yaw["res_lplc2"] * 3.0)
        self.assertGreater(p_app["res_lplc2"], p_lat["res_lplc2"] * 2.0)
        self.assertEqual(p_empty["res_lplc2"], 0.0)

    def test_real_world_env_emits_imu_and_encoder_and_preserves_rollback(self):
        from branch_rollout import compare_actions

        with RealWorldEnv(scene="ramp_descent", seed=2, sensor_noise=True) as env:
            obs = env.observe()
            for key in ("imu_gyro_z", "imu_gyro_y", "imu_accel_x", "encoder_speed_m_s"):
                self.assertIn(key, obs)
            for _ in range(5):
                obs = env.step("left")
            self.assertGreater(abs(obs["imu_gyro_z"]), 0.2)

            b1 = compare_actions(env, ["forward", "left", "brake"], 8)
            b2 = compare_actions(env, ["forward", "left", "brake"], 8)
            for act in ("forward", "left", "brake"):
                self.assertEqual(b1[act]["body_contact"], b2[act]["body_contact"])
                self.assertAlmostEqual(b1[act]["forward_progress_m"], b2[act]["forward_progress_m"], delta=1e-3)


if __name__ == "__main__":
    unittest.main()
