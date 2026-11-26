"""P4 Property & Paired-World Tests: FactorizedSpatialBelief and Information-Recoverability G_t(a)."""
import unittest

import numpy as np

from blindspot_dataset import make_paired_world
from branch_rollout import evaluate_action_branches_v2
from real_world_env import RealWorldEnv
from rover_sim import RoverSim
from spatial_belief_v2 import FactorizedSpatialBelief


class TestRecoverabilityV2(unittest.TestCase):
    def test_healthy_empty_scene_clears_uncertainty_across_lighting(self):
        # Empty scenes must register low occupancy and healthy camera under varied lighting/env
        for sc in ("open",):
            for cls in (RoverSim, RealWorldEnv):
                with cls(scene=sc, seed=5) as sim:
                    belief = FactorizedSpatialBelief(max_age_s=0.6)
                    u_start = belief.task_uncertainty(0.0)
                    for _ in range(8):
                        obs = sim.observe()
                        snap = belief.update(obs)
                        sim.step("forward")
                    u_end = snap["task_uncertainty"]
                    self.assertTrue(snap["camera_healthy"])
                    self.assertLess(max(snap["occupancy_probs"][2:7]), 0.15)
                    self.assertLess(u_end, u_start * 0.5)

    def test_color_agnostic_gray_obstacle_updates_occupancy(self):
        with RoverSim(scene="open", seed=3) as sim:
            sim.solids.append(sim.box([0.55, 0.0, 0.12], [0.03, 0.12, 0.12], 0, [0.35, 0.35, 0.35, 1]))
            sim.packet = None
            sim.queue.clear()
            sim.update_camera()
            belief = FactorizedSpatialBelief(max_age_s=0.6)
            for _ in range(5):
                obs = sim.observe()
                snap = belief.update(obs, world_yaw_rad=sim.evaluator()["yaw_rad"])
                sim.step("brake")
            center_occ = max(snap["occupancy_probs"][3:6])
            self.assertGreater(center_occ, 0.70, f"Gray obstacle missed in occupancy probs: {snap['occupancy_probs']}")

    def test_paired_recoverable_occlusion_vs_persistent_blackout(self):
        with make_paired_world("occluded_clear", seed=7, fault="none") as sim_ok:
            b_ok = FactorizedSpatialBelief(max_age_s=0.6)
            for _ in range(6):
                b_ok.update(sim_ok.observe(), world_yaw_rad=sim_ok.evaluator()["yaw_rad"])
                sim_ok.step("forward")
            res_ok = evaluate_action_branches_v2(sim_ok, ["forward", "left", "right", "brake"], 18, b_ok)

        with make_paired_world("occluded_clear", seed=7, fault="freeze") as sim_dead:
            sim_dead.window = (0.25, 5.0)
            b_dead = FactorizedSpatialBelief(max_age_s=0.6)
            for _ in range(6):
                b_dead.update(sim_dead.observe(), world_yaw_rad=sim_dead.evaluator()["yaw_rad"])
                sim_dead.step("forward")
            res_dead = evaluate_action_branches_v2(sim_dead, ["forward", "left", "right", "brake"], 18, b_dead)

        max_turn_gain_ok = max(res_ok["left"]["info_gain"], res_ok["right"]["info_gain"])
        max_turn_gain_dead = max(res_dead["left"]["info_gain"], res_dead["right"]["info_gain"])
        self.assertGreater(max_turn_gain_ok, 0.05, f"Expected positive info gain on occluded healthy scene: {res_ok}")
        self.assertEqual(max_turn_gain_dead, 0.0, f"Dead-camera blind turning must yield ZERO info gain: {res_dead}")

    def test_clear_open_scene_gives_negligible_turn_info_gain(self):
        with RoverSim(scene="open", seed=7) as sim:
            belief = FactorizedSpatialBelief(max_age_s=0.6)
            for _ in range(12):
                belief.update(sim.observe(), world_yaw_rad=sim.evaluator()["yaw_rad"])
                sim.step("forward")
            res = evaluate_action_branches_v2(sim, ["forward", "brake", "left", "right"], 18, belief)
        self.assertLess(res["left"]["info_gain"], 0.02, f"Open terrain spurious turn gain: {res['left']}")
        self.assertLess(res["right"]["info_gain"], 0.02, f"Open terrain spurious turn gain: {res['right']}")

    def test_world_referenced_turning_away_gives_non_positive_delta_u_and_preserves_negative_values(self):
        # Verify Refinement A & C: when forward world sectors become unobserved/stale and rover turns away,
        # delta_u can be <= 0 (including strictly negative if turn misses re-observing goal sectors vs forward/brake).
        with RoverSim(scene="open", seed=7) as sim:
            b = FactorizedSpatialBelief(max_age_s=0.6)
            # Observe initially cleanly
            for _ in range(4):
                b.update(sim.observe(), world_yaw_rad=sim.evaluator()["yaw_rad"])
                sim.step("forward")
            res = evaluate_action_branches_v2(sim, ["forward", "brake", "left"], 28, b)
        # Turning 1.4 rad left looks out at flank angles while center goal region becomes stale relative to brake!
        self.assertLess(res["left"]["delta_u"], 0.0, f"Turning away should have negative delta_u, got: {res['left']}")


if __name__ == "__main__":
    unittest.main()
