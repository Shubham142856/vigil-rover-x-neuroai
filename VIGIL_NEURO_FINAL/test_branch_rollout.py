import unittest
import numpy as np
from rover_sim import RoverSim
from branch_rollout import compare_actions

class BranchTests(unittest.TestCase):
    def test_branches_restore_delayed_camera_and_world(self):
        with RoverSim(scene="post",fault="delay",fault_start_s=0,fault_end_s=2) as sim:
            for _ in range(12): sim.step("forward")
            before=sim.evaluator(),sim.observe()
            a=compare_actions(sim,["forward","brake"],30)
            b=compare_actions(sim,["brake","forward"],30)
            self.assertEqual(a,b)
            self.assertEqual(sim.evaluator(),before[0])
            self.assertEqual(sim.observe()["capture_s"],before[1]["capture_s"])
            np.testing.assert_array_equal(sim.observe()["rgb"],before[1]["rgb"])
            self.assertTrue(a["forward"]["body_contact"])
            self.assertFalse(a["brake"]["body_contact"])
