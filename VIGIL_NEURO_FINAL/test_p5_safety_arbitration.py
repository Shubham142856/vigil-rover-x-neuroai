"""18 Mandatory P5 Safety, Arbitration, Vision, IMU, Recoverability & Evaluation Tests (Section 6)."""
import math
import unittest

import numpy as np
import pybullet as pb

from blindspot_dataset import make_paired_world
from branch_rollout import evaluate_action_branches_v2
from evaluation_metrics_v2 import clopper_pearson_ci, clopper_pearson_one_sided_upper, classify_episode
from real_world_env import RealWorldEnv
from rover_sim import RoverSim
from safety_arbitrator_v2 import (
    ActionFeasibilityChecker,
    ArbitrationConfig,
    IntegrityReport,
    ObservationIntegrityMonitor,
    compute_arbitrated_risk,
    fuse_biological_visual_threat,
    select_protected_action_p5,
)
from spatial_belief_v2 import FactorizedSpatialBelief
from t4_t5_vision import DualPathwayT4T5Vision


def _clear_snap(n=9):
    return {
        "camera_healthy": True,
        "sector_states": ["observed_free"] * n,
        "sector_ages_s": [0.02] * n,
        "occupancy_probs": [0.05] * n,
        "world_occupancy_probs": [0.5] * 30,
        "yaw_est_rad": 0.0,
        "heading_uncertainty_rad": 0.02,
        "task_uncertainty": 0.05,
    }


def _healthy_integrity():
    return IntegrityReport(
        camera_status="healthy", camera_healthy=True, delta_yaw_rad=0.0,
        imu_available=True, imu_duplicate=False, imu_gyro_z=0.0, imu_gyro_y=0.0,
        heading_uncertainty_delta_rad=0.0, probing_exhausted=False,
    )


def _ok_obs():
    return {
        "time_s": 0.5,
        "rgb": np.full((96, 128, 3), 180, dtype=np.uint8),
        "whisker_left": False,
        "whisker_right": False,
        "obstacle_center": 0.0,
        "frame_valid": True,
        "frame_age_s": 0.0,
    }


class TestP5Mandatory18Suite(unittest.TestCase):
    def setUp(self):
        self.checker = ActionFeasibilityChecker()
        self.cfg = ArbitrationConfig()
        self.clear_vis = {"raw_lplc2": 0.0, "residual_lplc2": 0.0, "residual_lc4": 0.0, "lateral_imbalance": 0.0}

    def _preds(self, fwd_r4, fwd_r8):
        base = {a: {"contact_prob": 0.05, "progress": 0.10, "recover_prob": 0.40} for a in ("forward", "slow", "left", "right", "brake")}
        base["forward"]["progress"] = 0.35
        base4 = {k: dict(v) for k, v in base.items()}
        base8 = {k: dict(v) for k, v in base.items()}
        base4["forward"]["contact_prob"] = fwd_r4
        base8["forward"]["contact_prob"] = fwd_r8
        return base4, base8

    # 1. M4 high risk, M8 low risk -> forward rejected
    def test_01_m4_high_m8_low_rejects_forward(self):
        m4, m8 = self._preds(0.85, 0.05)
        r_arb = compute_arbitrated_risk(0.85, 0.05, self.cfg.kappa_disagree)
        self.assertLessEqual(r_arb, 1.0)
        act, _ = select_protected_action_p5(_ok_obs(), _clear_snap(), _healthy_integrity(), self.clear_vis, 0.2, m4, m8, "forward", self.checker, self.cfg)
        self.assertNotEqual(act, "forward")

    # 2. M8 high risk, M4 low risk -> forward rejected
    def test_02_m8_high_m4_low_rejects_forward(self):
        m4, m8 = self._preds(0.05, 0.85)
        act, _ = select_protected_action_p5(_ok_obs(), _clear_snap(), _healthy_integrity(), self.clear_vis, 0.2, m4, m8, "forward", self.checker, self.cfg)
        self.assertNotEqual(act, "forward")

    # 3. Both models confidently wrong (predict 0.01 contact!) -> independent tactile & looming still protect
    def test_03_both_models_confidently_wrong_still_protected(self):
        m4, m8 = self._preds(0.01, 0.01)
        obs_whisk = dict(_ok_obs(), whisker_left=True, whisker_right=True)
        act1, _ = select_protected_action_p5(obs_whisk, _clear_snap(), _healthy_integrity(), self.clear_vis, 0.2, m4, m8, "forward", self.checker, self.cfg)
        self.assertEqual(act1, "brake")

        vis_threat = {"raw_lplc2": 1.2, "residual_lplc2": 1.1, "residual_lc4": 0.9, "lateral_imbalance": 0.05}
        act2, _ = select_protected_action_p5(_ok_obs(), _clear_snap(), _healthy_integrity(), vis_threat, 0.3, m4, m8, "forward", self.checker, self.cfg)
        self.assertNotEqual(act2, "forward")

    # 4. Static wall with rover approaching -> collision-relevant expansion preserved (no raw&res AND trap)
    def test_04_static_wall_approach_preserves_collision_expansion(self):
        # Simulate static forward expansion where residual stays moderate while raw_lplc2 is strong during linear approach
        vis = {"raw_lplc2": 0.85, "residual_lplc2": 0.15, "residual_lc4": 0.10, "lateral_imbalance": 0.02}
        threat, _, _ = fuse_biological_visual_threat(vis, _ok_obs(), _healthy_integrity())
        self.assertTrue(threat)

    # 5. Empty ramp descent -> no persistent false looming response
    def test_05_empty_ramp_descent_no_false_looming(self):
        vis = DualPathwayT4T5Vision()
        mon = ObservationIntegrityMonitor()
        threats = 0
        with RealWorldEnv(scene="ramp", seed=2, sensor_noise=True) as env:
            for _ in range(30):
                obs = env.step("forward")
                rep = mon.step(obs)
                vo = vis.step(obs["rgb"], 0.05, rep.imu_gyro_z, rep.imu_gyro_y)
                thr, _, _ = fuse_biological_visual_threat(vo, obs, rep)
                if thr:
                    threats += 1
        self.assertLessEqual(threats, 2, f"Empty ramp descent triggered {threats} false threats")

    # 6. Lateral near miss -> no unnecessary emergency stop
    def test_06_lateral_near_miss_no_unnecessary_stop(self):
        vis_lat = {"raw_lplc2": 0.10, "residual_lplc2": 0.08, "residual_lc4": 0.15, "lateral_imbalance": 0.95}
        threat, _, _ = fuse_biological_visual_threat(vis_lat, _ok_obs(), _healthy_integrity())
        self.assertFalse(threat)

    # 7. Genuine crossing collision -> protective response
    def test_07_genuine_crossing_collision_protected(self):
        vis = DualPathwayT4T5Vision()
        mon = ObservationIntegrityMonitor()
        protected = False
        with RoverSim(scene="open", seed=5) as sim:
            post = sim.box([0.45, 0.0, 0.12], [0.05, 0.05, 0.12], 0, [0.86, 0.17, 0.09, 1])
            sim.solids.append(post)
            sim.packet = None; sim.queue.clear(); sim.update_camera()
            for step in range(25):
                obs = sim.step("forward")
                rep = mon.step(obs, imu_enabled=False)
                vo = vis.step(obs["rgb"], 0.05, 0.0, 0.0)
                thr, _, _ = fuse_biological_visual_threat(vo, obs, rep)
                if thr or obs["whisker_left"] or obs["whisker_right"]:
                    protected = True
                    break
        self.assertTrue(protected)

    # 8. Previously free but stale sector -> not automatically accepted
    def test_08_stale_previously_free_sector_rejected(self):
        snap = _clear_snap()
        snap["sector_states"] = ["stale_or_unknown"] * 9
        snap["sector_ages_s"] = [1.5] * 9
        feas = self.checker.feasible_actions(("forward", "left", "brake"), snap, _ok_obs(), _healthy_integrity(), 0.25)
        self.assertFalse(feas["forward"][0])
        self.assertFalse(feas["left"][0])
        self.assertTrue(feas["brake"][0])

    # 9. Obstacle behind reversing rover -> unsafe reverse rejected
    def test_09_unsafe_reverse_rejected(self):
        snap = _clear_snap()
        # World map has default prior 0.50 (unverified) or rear occupied -> must reject blind reverse
        feas1 = self.checker.feasible_actions(("reverse",), snap, _ok_obs(), _healthy_integrity(), 0.0)
        self.assertFalse(feas1["reverse"][0])
        snap["world_occupancy_probs"] = [0.85] * 30
        feas2 = self.checker.feasible_actions(("reverse",), snap, _ok_obs(), _healthy_integrity(), 0.0)
        self.assertFalse(feas2["reverse"][0])

    # 10. Gyroscope timestamp duplication -> no duplicate heading integration
    def test_10_gyro_timestamp_duplication_not_double_integrated(self):
        mon = ObservationIntegrityMonitor()
        obs1 = dict(_ok_obs(), time_s=0.10, imu_gyro_z=1.0, imu_gyro_y=0.0)
        obs2 = dict(_ok_obs(), time_s=0.20, imu_gyro_z=1.0, imu_gyro_y=0.0)
        obs_dup = dict(_ok_obs(), time_s=0.20, imu_gyro_z=1.0, imu_gyro_y=0.0)
        r1 = mon.step(obs1)
        r2 = mon.step(obs2)
        r3 = mon.step(obs_dup)
        self.assertAlmostEqual(r2.delta_yaw_rad, 0.10, places=3)
        self.assertTrue(r3.imu_duplicate)
        self.assertEqual(r3.delta_yaw_rad, 0.0)

    # 11. Missing IMU measurement -> directional uncertainty increases
    def test_11_missing_imu_increases_directional_uncertainty(self):
        b = FactorizedSpatialBelief(max_age_s=0.6)
        u0 = b.heading_uncertainty_rad
        obs_no_imu = dict(_ok_obs(), time_s=0.10)  # no imu_gyro_z key
        snap = b.update(obs_no_imu)
        self.assertGreater(snap["heading_uncertainty_rad"], u0)

    # 12. Temporary camera freeze -> passive brake recovery not miscredited
    def test_12_temporary_camera_freeze_does_not_credit_brake(self):
        with make_paired_world("occluded_clear", seed=7, fault="freeze") as sim:
            b = FactorizedSpatialBelief(max_age_s=0.6)
            for _ in range(12):
                b.update(sim.observe(), world_yaw_rad=sim.evaluator()["yaw_rad"])
                sim.step("forward")
            res = evaluate_action_branches_v2(sim, ["brake", "left"], 28, b)
        self.assertEqual(res["brake"]["delta_u"], 0.0)

    # 13. Persistent blackout -> no repeated ineffective probing (stops after max_freeze_probes)
    def test_13_persistent_blackout_halts_repeated_probing(self):
        mon = ObservationIntegrityMonitor(max_freeze_probes=2)
        obs_dead = {"time_s": 1.0, "frame_valid": False, "frame_age_s": None, "whisker_left": False, "whisker_right": False, "brightness": 0.0, "contrast": 0.0}
        snap = _clear_snap()
        snap["camera_healthy"] = False
        r1 = mon.step(obs_dead)
        self.assertFalse(r1.probing_exhausted)
        mon.record_action("left", outage_active=True)
        obs_dead["time_s"] = 1.05
        mon.step(obs_dead)
        mon.record_action("right", outage_active=True)
        obs_dead["time_s"] = 1.10
        r3 = mon.step(obs_dead)
        self.assertTrue(r3.probing_exhausted)
        feas = self.checker.feasible_actions(("left", "right", "brake"), snap, obs_dead, r3, 0.0)
        self.assertFalse(feas["left"][0])
        self.assertFalse(feas["right"][0])

    # 14. Recoverable geometric occlusion -> safe information-recovery action considered
    def test_14_recoverable_occlusion_considers_recovery_action(self):
        snap = _clear_snap()
        snap["task_uncertainty"] = 0.85  # high corridor uncertainty
        m4, m8 = self._preds(0.65, 0.65)
        m8["left"]["progress"] = 0.12
        m8["left"]["delta_u"] = 0.55
        m8["brake"]["delta_u"] = 0.0
        act, _ = select_protected_action_p5(_ok_obs(), snap, _healthy_integrity(), self.clear_vis, 0.1, m4, m8, "forward", self.checker, self.cfg)
        self.assertEqual(act, "left")

    # 15. Neural action oscillation -> switching penalty bounds unnecessary flipping
    def test_15_switching_penalty_bounds_action_oscillation(self):
        snap = _clear_snap()
        snap["task_uncertainty"] = 0.60
        m4, m8 = self._preds(0.75, 0.75)
        # Left & right equal on risk/progress/info; if prev_action=="left", switching penalty keeps "left"
        m8["left"]["delta_u"] = 0.30
        m8["right"]["delta_u"] = 0.32
        act, _ = select_protected_action_p5(_ok_obs(), snap, _healthy_integrity(), self.clear_vis, 0.1, m4, m8, "left", self.checker, self.cfg)
        self.assertEqual(act, "left")

    # 16. Counterfactual negative information gain -> negative label preserved
    def test_16_negative_delta_u_preserved(self):
        with RoverSim(scene="open", seed=7) as sim:
            b = FactorizedSpatialBelief(max_age_s=0.6)
            for _ in range(4):
                b.update(sim.observe(), world_yaw_rad=sim.evaluator()["yaw_rad"])
                sim.step("forward")
            res = evaluate_action_branches_v2(sim, ["forward", "brake", "left"], 28, b)
        self.assertLess(res["left"]["delta_u"], 0.0)

    # 17. All actions infeasible -> appropriate protected abstention
    def test_17_all_infeasible_triggers_protected_abstention(self):
        snap = _clear_snap()
        snap["camera_healthy"] = False
        snap["sector_states"] = ["stale_or_unknown"] * 9
        obs_dead = dict(_ok_obs(), frame_valid=False, frame_age_s=None)
        dead_int = IntegrityReport("blackout", False, 0.0, True, False, 0.0, 0.0, 0.0, True)
        m4, m8 = self._preds(0.1, 0.1)
        act, rsn = select_protected_action_p5(obs_dead, snap, dead_int, self.clear_vis, 0.0, m4, m8, "brake", self.checker, self.cfg)
        self.assertEqual(act, "brake")
        self.assertIn("abstain", rsn)

    # 18. Disaggregated metrics & consistent two-sided Clopper-Pearson bounds check
    def test_18_disaggregated_metrics_and_general_clopper_pearson(self):
        lo0, hi0 = clopper_pearson_ci(0, 24)
        self.assertEqual(lo0, 0.0)
        self.assertAlmostEqual(hi0, 1.0 - 0.025 ** (1.0 / 24), places=6)
        self.assertAlmostEqual(clopper_pearson_one_sided_upper(0, 24), 1.0 - 0.05 ** (1.0 / 24), places=6)
        lo2, hi2 = clopper_pearson_ci(2, 50)
        self.assertGreater(lo2, 0.0)
        self.assertLess(lo2, 2 / 50)
        self.assertGreater(hi2, 2 / 50)

        ep_stall = classify_episode("ramp_descent", 0.05, 0.0, 0, 0, False, 0, ["brake"] * 80, 0.4, 0.4)
        self.assertTrue(ep_stall["permanent_stall"])
        self.assertFalse(ep_stall["collision_free_goal_completion"])
        ep_goal = classify_episode("ramp_descent", 0.70, 0.0, 0, 0, False, 0, ["forward"] * 80, 0.4, 0.05)
        self.assertTrue(ep_goal["collision_free_goal_completion"])
        self.assertFalse(ep_goal["permanent_stall"])


if __name__ == "__main__":
    unittest.main()
