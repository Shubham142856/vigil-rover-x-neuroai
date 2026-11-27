"""P6.9.2 & P6.9.3 Unit & Integration Tests:
- Multi-modal circular von Mises heading belief (IMU, landmark, optic-flow, blackout, cue conflict, reacquisition)
- Global heading uncertainty vs local egocentric feasibility decoupling
- Stale IPM invalidation on freeze/blackout, contiguous IPM range filter, full chassis-width 'slow' guard
- Bounded active observation streak/cooldown + conservative uncertainty-penalized info gain
- Restored G08 open_healthy navigation without artificial uncertainty resets
"""

import csv
import math
import unittest
from pathlib import Path

import numpy as np

from blindspot_dataset import SectorBelief
from compass_benchmark import load_compass_connectome
from ground_range_estimator import estimate_ground_contact_clearance
from p6_closed_loop_pilot import run_p6_episode
from safety_arbitrator_v2 import (
    ActionFeasibilityChecker,
    ArbitrationConfig,
    BoundedExplorationMonitor,
    IntegrityReport,
    ObservationIntegrityMonitor,
    fuse_biological_visual_threat,
    select_protected_action_p5,
)
from spatial_belief_v2 import FactorizedSpatialBelief, _vm_inv_resultant, _vm_mean_resultant


def _obs(t=0.5, contrast=0.2, brightness=0.5, healthy=True):
    rgb = np.full((96, 128, 3), 180, dtype=np.uint8)
    if contrast < 0.01:
        rgb[:] = 120
    return {
        "time_s": t,
        "rgb": rgb,
        "frame_valid": healthy,
        "frame_age_s": 0.0 if healthy else 1.2,
        "whisker_left": False,
        "whisker_right": False,
        "brightness": brightness,
        "contrast": contrast,
        "obstacle_center": 0.0,
    }


class TestP69HeadingAndProtection(unittest.TestCase):
    def test_von_mises_bessel_roundtrip_accuracy(self):
        for k in (0.05, 0.5, 2.0, 10.0, 45.0, 200.0):
            r = _vm_mean_resultant(k)
            k_back = _vm_inv_resultant(r)
            self.assertAlmostEqual(k_back, k, delta=max(1e-4, 1e-3 * k))

    def test_blackout_increases_uncertainty_and_reacquisition_recovers(self):
        b = FactorizedSpatialBelief(max_age_s=0.6)
        u0 = b.heading_uncertainty_rad
        # Simulate camera blackout without IMU over multiple steps
        for step in range(1, 15):
            snap = b.update(_obs(t=0.05 * step, healthy=False), action_cmd="left")
        u_blackout = snap["heading_uncertainty_rad"]
        self.assertGreater(u_blackout, u0 + 0.25)

        # Reacquire consistent visual landmark -> justified reduction in circular uncertainty
        snap_rec = b.update(
            _obs(t=0.80, healthy=True, contrast=0.22),
            action_cmd="brake",
            visual_yaw_rad=snap["yaw_est_rad"],
            visual_yaw_kappa=18.0,
        )
        self.assertLess(snap_rec["heading_uncertainty_rad"], u_blackout)

    def test_cue_conflict_expands_circular_uncertainty(self):
        b = FactorizedSpatialBelief(max_age_s=0.6)
        b.heading_kappa = 40.0
        b.heading_uncertainty_rad = 0.16
        u_before = b.heading_uncertainty_rad
        # Strong 90-degree conflict between predicted yaw (0.0) and visual landmark cue (pi/2)
        snap = b.update(
            _obs(t=0.05, healthy=True, contrast=0.22),
            action_cmd="brake",
            visual_yaw_rad=math.pi / 2.0,
            visual_yaw_kappa=25.0,
        )
        self.assertTrue(snap["heading_conflict"])
        self.assertGreater(snap["heading_uncertainty_rad"], u_before)

    def test_blank_healthy_camera_does_not_stabilize_turning_uncertainty(self):
        b = FactorizedSpatialBelief(max_age_s=0.6)
        for step in range(1, 10):
            snap = b.update(
                _obs(t=0.05 * step, healthy=True, contrast=0.005),
                action_cmd="left",
                visual_motion_stable=False,
            )
        self.assertGreater(snap["heading_uncertainty_rad"], 0.20)

    def test_global_heading_uncertainty_blocks_reverse_not_local_healthy_corridor(self):
        checker = ActionFeasibilityChecker()
        snap = {
            "camera_healthy": True,
            "sector_states": ["observed_free"] * 9,
            "sector_ages_s": [0.02] * 9,
            "occupancy_probs": [0.05] * 9,
            "world_occupancy_probs": [0.05] * 30,  # rear clear in world map, MUST still fail under global uncertainty
            "yaw_est_rad": 0.0,
            "heading_uncertainty_rad": 1.10,  # > max_head_unc (0.55) and > 0.80
        }
        rep = IntegrityReport("healthy", True, None, False, False, 0.0, 0.0, 0.0, False)
        feas = checker.feasible_actions(("reverse", "slow", "left", "brake"), snap, _obs(), rep, 0.10)
        self.assertFalse(feas["reverse"][0])
        self.assertTrue(feas["slow"][0])
        self.assertTrue(feas["left"][0])

    def test_stale_frozen_ipm_invalidated_and_slow_checks_front_footprint(self):
        # 1. Create RGB with huge red obstacle right in front, mark camera unhealthy (frozen/stale)
        obs = _obs(healthy=False)
        obs["rgb"][45:92, 42:86, 0] = 230
        obs["rgb"][45:92, 42:86, 1:] = 10
        obs["obstacle_center"] = 0.05
        dead_rep = IntegrityReport("frozen", False, 0.0, True, False, 0.0, 0.0, 0.0, False)
        vis = {"raw_lplc2": 0.0, "residual_lplc2": 0.0, "residual_lc4": 0.0, "lateral_imbalance": 0.0}
        threat, reason, _ = fuse_biological_visual_threat(vis, obs, dead_rep)
        self.assertFalse(threat, f"Stale frozen IPM falsely triggered visual threat: {reason}")

        # 2. Adjacent front corner (mid-1) occupied -> 'slow' must be rejected across chassis footprint
        checker = ActionFeasibilityChecker()
        snap = {
            "camera_healthy": True,
            "sector_states": ["observed_free", "observed_free", "observed_free", "occupied", "observed_free", "observed_free", "observed_free", "observed_free", "observed_free"],
            "near_blocked_sectors": [False, False, False, True, False, False, False, False, False],
            "sector_ages_s": [0.02] * 9,
            "world_occupancy_probs": [0.5] * 30,
            "yaw_est_rad": 0.0,
            "heading_uncertainty_rad": 0.05,
        }
        ok_rep = IntegrityReport("healthy", True, 0.0, True, False, 0.0, 0.0, 0.0, False)
        feas = checker.feasible_actions(("slow",), snap, _obs(healthy=True), ok_rep, 0.12)
        self.assertFalse(feas["slow"][0])

    def test_bounded_exploration_forces_cooldown_and_conservative_info_gain(self):
        cfg = ArbitrationConfig(max_explore_turn_streak=2, explore_cooldown_steps=1, gamma_info_unc=0.80)
        checker = ActionFeasibilityChecker()
        mon = BoundedExplorationMonitor()
        snap = {
            "camera_healthy": True,
            "sector_states": ["observed_free"] * 9,
            "sector_ages_s": [0.02] * 9,
            "world_occupancy_probs": [0.5] * 30,
            "yaw_est_rad": 0.0,
            "heading_uncertainty_rad": 0.05,
            "task_uncertainty": 0.90,
        }
        rep = IntegrityReport("healthy", True, 0.0, True, False, 0.0, 0.0, 0.0, False)
        vis = {"raw_lplc2": 0.0, "residual_lplc2": 0.0, "residual_lc4": 0.0, "lateral_imbalance": 0.0}
        m4 = {a: {"contact_prob": 0.05 if a != "forward" else 0.70} for a in ("forward", "slow", "brake", "left", "right")}
        # High uncertainty on 'right' vs low uncertainty on 'left' -> conservative info gain prefers 'left'
        m8 = {
            "forward": {"contact_prob": 0.70, "progress": 0.25, "delta_u": 0.0, "delta_u_std": 0.0},
            "slow": {"contact_prob": 0.05, "progress": 0.08, "delta_u": 0.05, "delta_u_std": 0.0},
            "brake": {"contact_prob": 0.0, "progress": 0.0, "delta_u": 0.0, "delta_u_std": 0.0},
            "left": {"contact_prob": 0.05, "progress": 0.02, "delta_u": 0.50, "delta_u_std": 0.05},
            "right": {"contact_prob": 0.05, "progress": 0.02, "delta_u": 0.58, "delta_u_std": 0.30},
        }
        a1, _ = select_protected_action_p5(_obs(), snap, rep, vis, 0.1, m4, m8, "brake", checker, cfg, exploration_monitor=mon)
        a2, _ = select_protected_action_p5(_obs(), snap, rep, vis, 0.1, m4, m8, a1, checker, cfg, exploration_monitor=mon)
        a3, _ = select_protected_action_p5(_obs(), snap, rep, vis, 0.1, m4, m8, a2, checker, cfg, exploration_monitor=mon)
        self.assertEqual(a1, "left")
        self.assertEqual(a2, "left")
        # After max_explore_turn_streak=2, step 3 MUST pause/re-observe instead of spinning continuously
        self.assertIn(a3, ("slow", "brake"))

    def test_contiguous_ipm_ignores_isolated_bottom_floor_crack(self):
        # Create scene with distant red target near horizon [24:36] and isolated dark strip on floor [88:91]
        rgb = np.full((96, 128, 3), 185, dtype=np.uint8)
        rgb[24:36, 56:72, 0] = 225
        rgb[24:36, 56:72, 1:] = 20
        rgb[88:91, 45:80, :] = 30  # disconnected dark floor shadow/seam
        est = estimate_ground_contact_clearance(rgb)
        self.assertGreater(est["estimated_clearance_m"], 0.80)

    def test_open_healthy_g08_navigates_to_goal_under_modal_belief(self):
        escape_dir = Path("runs/graph-controls/escape")
        ann_path = Path("runs/connectome/escape/annotations.csv")
        compass_dir = Path("runs/connectome/compass")
        weights_path = Path("runs/pilot-v3-models/pilot_weights_v3.npz")
        if not weights_path.exists() or not escape_dir.exists():
            self.skipTest("Pilot weights or connectome assets missing")
        with ann_path.open() as f:
            types = {r["root_id"]: r["primary_type"] for r in csv.DictReader(f)}
        ids = (escape_dir / "nodes.txt").read_text().splitlines()
        escape_assets = {
            "npz": escape_dir / "biological.npz",
            "labels": np.array([types[i] for i in ids]),
            "threshold": 0.0238,
        }
        compass_model = load_compass_connectome(compass_dir)
        rec = run_p6_episode(
            scenario="open_healthy",
            config_name="P6_C_Plus_IPMFeas",
            seed=0,
            escape_assets=escape_assets,
            compass_model=compass_model,
            env_mode="rover_sim",
            v3_weights_path=weights_path,
        )
        self.assertEqual(rec["body_collisions"], 0)
        self.assertTrue(rec["collision_free_goal_completion"], f"open_healthy failed goal under modal belief: {rec}")


if __name__ == "__main__":
    unittest.main()
