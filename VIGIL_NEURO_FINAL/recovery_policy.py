"""Conservative tactile & blind-spot recovery controller with safe abstention."""
import argparse
import json
from pathlib import Path

from blindspot_dataset import SectorBelief
from rover_sim import RoverSim


def choose_recovery_action(obs, belief_snapshot, looming_alert=False):
    """Select safe rover action from delivered sensors and aged sector belief.

    Never turns blindly into an occupied, stale, or unknown sector.
    """
    left = obs.get("whisker_left")
    right = obs.get("whisker_right")
    if left is None or right is None:
        return "brake", "missing tactile observation"
    if left and right:
        return "brake", "bilateral tactile contact"

    states = belief_snapshot["sector_states"]
    n = len(states)
    mid = n // 2
    right_clear = all(st == "observed_free" for st in states[mid + 1 :])
    left_clear = all(st == "observed_free" for st in states[:mid])

    if left and not right:
        if right_clear:
            return "right", "verified-clear right escape on left tactile contact"
        return "brake", "abstain: right escape unverified or blocked"
    if right and not left:
        if left_clear:
            return "left", "verified-clear left escape on right tactile contact"
        return "brake", "abstain: left escape unverified or blocked"

    if not belief_snapshot["camera_healthy"]:
        return "brake", "abstain: camera outage without verified-clear path"
    if looming_alert:
        return "brake", "circuit looming collision alert"
    return "forward", "clear path below looming threshold"


def evaluate_tactile_scenarios(out):
    """Run matched G03/G08 tactile + camera-outage PyBullet scenarios."""
    results = {}

    # 1. Bilateral stop against front wall
    with RoverSim(scene="wall", seed=7) as sim:
        reasons = []
        for _ in range(50):
            obs = sim.observe()
            snap = SectorBelief(max_age_s=0.6).update(obs)
            action, reason = choose_recovery_action(obs, snap)
            reasons.append(reason)
            sim.step(action)
        results["bilateral_wall"] = {
            "final": sim.evaluator(),
            "last_reason": reasons[-1],
        }

    # 2. One-sided left contact during camera outage with blocked opposite side -> abstain/brake
    with RoverSim(scene="left", seed=7, fault="freeze", fault_start_s=0.95, fault_end_s=5.0) as sim:
        # Block right sector (outside right whisker at y=-0.07) so right escape is unsafe.
        sim.solids.append(sim.box([0.55, -0.18, 0.12], [0.03, 0.05, 0.12], 0, [0.86, 0.17, 0.09, 1]))
        sim.packet = None
        sim.queue.clear()
        sim.update_camera()
        belief = SectorBelief(max_age_s=0.4)
        abstained = False
        for _ in range(50):
            obs = sim.observe()
            snap = belief.update(obs)
            action, reason = choose_recovery_action(obs, snap)
            if obs["whisker_left"] and action == "brake" and "abstain" in reason:
                abstained = True
            sim.step(action)
        results["onesided_blocked_outage"] = {
            "final": sim.evaluator(),
            "abstained_safely": abstained,
        }

    # 3. One-sided left contact during fresh-memory camera outage with verified-clear right side -> escape
    with RoverSim(scene="left", seed=7, fault="drop", fault_start_s=0.95, fault_end_s=1.4) as sim:
        belief = SectorBelief(max_age_s=0.6)
        escaped = False
        for _ in range(60):
            obs = sim.observe()
            snap = belief.update(obs)
            action, reason = choose_recovery_action(obs, snap)
            if obs["whisker_left"] and action == "right":
                escaped = True
            sim.step(action)
        results["onesided_verified_clear_outage"] = {
            "final": sim.evaluator(),
            "escaped_clear": escaped,
        }

    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    print(json.dumps(evaluate_tactile_scenarios(p.parse_args().out), indent=2))
