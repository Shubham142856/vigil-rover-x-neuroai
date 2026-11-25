import importlib
import json
import shutil
import uuid
import unittest
from pathlib import Path

import numpy as np

from baseline_policy import decide


class RoverTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("rover_sim"), "Rover simulator is missing")
        self.module = importlib.import_module("rover_sim")

    def rover(self, **kwargs):
        return self.enterContext(self.module.RoverSim(**kwargs))

    def test_motion_brakes_and_turns_at_fixed_control_intervals(self):
        sim = self.rover(scene="open")
        start = sim.evaluator()["x_m"]
        sim.step("forward")
        first_speed = sim.evaluator()["speed_m_s"]
        for _ in range(25):
            sim.step("forward")
        moving = sim.evaluator()
        self.assertGreater(moving["speed_m_s"], first_speed + 0.12)
        self.assertGreater(moving["x_m"] - start, 0.22)
        for _ in range(15):
            sim.step("brake")
        stopped = sim.evaluator()
        self.assertLess(stopped["speed_m_s"], 0.015)
        self.assertGreater(stopped["x_m"], moving["x_m"] + 0.008)
        self.assertLess(stopped["x_m"] - moving["x_m"], 0.12)
        for _ in range(15):
            obs = sim.step("left")
        self.assertGreater(sim.evaluator()["yaw_rad"] - stopped["yaw_rad"], 0.45)
        self.assertAlmostEqual(obs["time_s"], 56 / 20)

    def test_spatial_rgb_stream_is_reproducible_and_policy_sees_only_sensors(self):
        a = self.rover(scene="wall", seed=13)
        b = self.rover(scene="wall", seed=13)
        initial = a.observe()["rgb"].copy()
        for _ in range(22):
            obs = a.step("forward")
            duplicate = b.step("forward")
        self.assertEqual(obs["rgb"].dtype, np.uint8)
        self.assertEqual(obs["rgb"].shape, (96, 128, 3))
        np.testing.assert_array_equal(obs["rgb"], duplicate["rgb"])
        self.assertFalse(np.array_equal(obs["rgb"], initial))
        self.assertEqual(a.evaluator(), b.evaluator())
        self.assertTrue(set(obs).isdisjoint({"x_m", "yaw_rad", "body_contact", "scene", "fault"}))
        self.assertIn(decide(obs)[0], ("forward", "brake"))

    def test_independent_feelers_trigger_before_chassis_and_brake(self):
        for scene, expected in [("left", (True, False)), ("right", (False, True)),
                                ("wall", (True, True))]:
            with self.subTest(scene=scene):
                sim = self.rover(scene=scene)
                for _ in range(100):
                    obs = sim.step("forward")
                    if obs["whisker_left"] or obs["whisker_right"]:
                        break
                self.assertEqual((obs["whisker_left"], obs["whisker_right"]), expected)
                self.assertEqual(decide(obs), ("brake", "tactile contact"))
                for _ in range(20):
                    sim.step("forward")
                self.assertLess(sim.evaluator()["speed_m_s"], 0.015)
                self.assertEqual(sim.evaluator()["body_contact_events"], 0)

    def test_center_post_causes_body_contact_without_feeler_contact(self):
        sim = self.rover(scene="post")
        hits = []
        for _ in range(100):
            obs = sim.step("forward")
            state = sim.evaluator()
            if state["body_contact"]:
                hits.append(state)
                self.assertFalse(obs["whisker_left"] or obs["whisker_right"])
        self.assertTrue(hits)
        self.assertEqual(hits[-1]["body_contact_events"], 1)

    def test_faults_preserve_source_time_and_recover(self):
        for fault, delivery in [("drop", 0.1), ("freeze", 0.4), ("delay", 0.1)]:
            with self.subTest(fault=fault):
                sim = self.rover(fault=fault, fault_start_s=0.2, fault_end_s=0.5, delay_s=0.3)
                for _ in range(8):
                    obs = sim.step("forward")
                self.assertAlmostEqual(obs["capture_s"], 0.1)
                self.assertAlmostEqual(obs["delivery_s"], delivery)
                self.assertAlmostEqual(obs["frame_age_s"], 0.3)
                self.assertEqual(decide(obs)[0], "brake")
                for _ in range(5):
                    obs = sim.step("brake")
                self.assertAlmostEqual(obs["capture_s"], 0.6)
                self.assertLess(obs["frame_age_s"], 0.06)

    def test_delayed_packets_arrive_on_schedule(self):
        sim = self.rover(fault="delay", fault_start_s=0, fault_end_s=1, delay_s=0.15)
        self.assertFalse(sim.observe()["frame_valid"])
        for _ in range(3):
            obs = sim.step("brake")
        self.assertEqual((obs["capture_s"], obs["delivery_s"]), (0, 0.15))
        self.assertTrue(obs["frame_valid"])

    def test_export_references_observation_at_decision_time(self):
        tmp = Path.cwd() / ("rover-test-" + uuid.uuid4().hex)
        tmp.mkdir()
        self.addCleanup(shutil.rmtree, tmp)
        dest = Path(tmp) / "episode"
        self.module.run_episode(dest, steps=9, scene="open", seed=12)
        records = [json.loads(line) for line in (dest / "replay.jsonl").read_text().splitlines()]
        self.assertEqual(len(records), 10)
        self.assertIsNone(records[-1]["action"])
        summary = json.loads((dest / "summary.json").read_text())
        with np.load(dest / "frames.npz") as frames:
            self.assertEqual(frames["rgb"].shape, (5, 96, 128, 3))
            self.assertEqual(summary["delivered_frames"], 5)
            for record in records:
                sample = record["sensors"]
                index = sample["frame_index"]
                self.assertEqual(sample["capture_s"], frames["capture_s"][index])
                self.assertEqual(sample["delivery_s"], frames["delivery_s"][index])
                if record["action"]:
                    self.assertEqual(record["action"], decide(sample)[0])
        with self.assertRaises(FileExistsError):
            self.module.run_episode(dest, steps=1)


if __name__ == "__main__":
    unittest.main()
