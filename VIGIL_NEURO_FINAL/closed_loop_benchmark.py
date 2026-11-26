"""G08 closed-loop benchmark: recovery-off vs ordinary recovery vs circuit-conditioned recovery."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from blindspot_dataset import ORACLE_KEYS, SectorBelief, make_paired_world
from camera_compass import bearing, fresh_heading
from camera_escape import Network, visual_looming
from compass_benchmark import ConnectomeCompass, load_compass_connectome
from recovery_policy import choose_recovery_action
from rover_sim import RoverSim


def build_scenario(name, seed):
    rng = np.random.default_rng(seed)
    if name == "open_healthy":
        sim = RoverSim(scene="open", seed=seed)
    elif name == "looming_post":
        x = float(rng.uniform(0.60, 0.78))
        sim = RoverSim(scene="open", seed=seed)
        sim.solids.append(sim.box([x, 0.0, 0.12], [0.03, 0.02, 0.12], 0, [0.86, 0.17, 0.09, 1]))
    elif name == "near_miss_post":
        x = float(rng.uniform(0.60, 0.78))
        side = 1.0 if seed % 2 == 0 else -1.0
        y = side * float(rng.uniform(0.15, 0.22))
        sim = RoverSim(scene="open", seed=seed)
        sim.solids.append(sim.box([x, y, 0.12], [0.03, 0.02, 0.12], 0, [0.86, 0.17, 0.09, 1]))
    elif name == "freeze_outage_clear":
        sim = RoverSim(scene="left", seed=seed, fault="drop", fault_start_s=0.95, fault_end_s=1.45)
    elif name == "blocked_outage":
        sim = RoverSim(scene="left", seed=seed, fault="freeze", fault_start_s=0.95, fault_end_s=5.0)
        sim.solids.append(sim.box([0.55, -0.18, 0.12], [0.03, 0.05, 0.12], 0, [0.86, 0.17, 0.09, 1]))
    elif name == "occluded_hazard":
        return make_paired_world("occluded_hazard", seed=seed, fault="none", hidden_side=1)
    else:
        raise ValueError(f"Unknown scenario: {name}")
    sim.packet = None
    sim.queue.clear()
    sim.update_camera()
    return sim


def run_episode(scenario, policy_mode, seed, escape_assets, compass_model):
    belief = SectorBelief(max_age_s=0.6)
    vstate = None
    last_cap = None
    net = Network(escape_assets["npz"], escape_assets["labels"]) if escape_assets else None
    thresh = escape_assets["threshold"] if escape_assets else 0.024
    compass = ConnectomeCompass(compass_model) if compass_model else None
    latched_escape = False
    abstained = False
    whisker_events = 0
    prev_whisker = False
    anchor = None

    with build_scenario(scenario, seed) as sim:
        obs = sim.observe()
        for step in range(80):
            if not ORACLE_KEYS.isdisjoint(obs):
                raise AssertionError("Oracle field leaked into closed-loop observation")
            snap = belief.update(obs)
            w_any = bool(obs["whisker_left"] or obs["whisker_right"])
            if w_any and not prev_whisker:
                whisker_events += 1
            prev_whisker = w_any

            if policy_mode == "recovery_off":
                # Conservative passive stop: brakes on tactile, outage, or any visual warning; never turns.
                if w_any or not snap["camera_healthy"] or obs["obstacle_center"] > 0.12:
                    action, reason = "brake", "passive stop (recovery off)"
                    if not snap["camera_healthy"]:
                        abstained = True
                else:
                    action, reason = "forward", "open forward"
            elif policy_mode == "ordinary_recovery":
                # Generic pixel-fraction threshold (prone to false alarm on off-center obstacles) + tactile recovery
                generic_alert = obs["obstacle_center"] > 0.08 or obs["looming_center"] > 0.04
                action, reason = choose_recovery_action(obs, snap, looming_alert=generic_alert)
                if "abstain" in reason:
                    abstained = True
            elif policy_mode == "circuit_recovery":
                # Connectome-informed: FAFB escape looming circuit + compass stability + safe recovery
                if obs["capture_s"] != last_cap:
                    dt = 0.1 if last_cap is None else max(0.05, obs["capture_s"] - last_cap)
                    lplc2_d, lc4_d, vstate = visual_looming(obs["rgb"], vstate, dt)
                    if net is not None:
                        score = net.update(lplc2_d, lc4_d)
                        if score >= thresh:
                            latched_escape = True
                    if compass is not None:
                        b = bearing(obs["rgb"])
                        if anchor is None and b is not None:
                            anchor = b
                        cue = None if (b is None or anchor is None) else (b - anchor)
                        _, bump = compass.step(0.0, cue)
                        if bump < 0.25:
                            latched_escape = True
                    last_cap = obs["capture_s"]
                action, reason = choose_recovery_action(obs, snap, looming_alert=latched_escape)
                if "abstain" in reason:
                    abstained = True
            else:
                raise ValueError(policy_mode)

            obs = sim.step(action)
        ev = sim.evaluator()

    navigable = scenario in ("open_healthy", "near_miss_post")
    false_stop = navigable and ev["x_m"] < 0.90 and ev["body_contact_events"] == 0 and not abstained
    if scenario in ("open_healthy", "near_miss_post"):
        safe_success = ev["body_contact_events"] == 0 and ev["x_m"] >= 0.90
    elif scenario in ("looming_post", "occluded_hazard"):
        safe_success = ev["body_contact_events"] == 0
    else:
        safe_success = ev["body_contact_events"] == 0 and (abstained or ev["x_m"] > 0.30)
    return {
        "scenario": scenario,
        "policy": policy_mode,
        "seed": seed,
        "whisker_contact_events": whisker_events,
        "body_contact_events": int(ev["body_contact_events"]),
        "progress_m": float(ev["x_m"]),
        "abstained": bool(abstained),
        "false_emergency_stop": bool(false_stop),
        "safe_success": bool(safe_success),
    }


def run(out: Path, seeds=4):
    escape_dir = Path("runs/graph-controls/escape")
    ann_path = Path("runs/connectome/escape/annotations.csv")
    compass_dir = Path("runs/connectome/compass")

    with ann_path.open() as f:
        types = {r["root_id"]: r["primary_type"] for r in csv.DictReader(f)}
    ids = (escape_dir / "nodes.txt").read_text().splitlines()
    escape_assets = {
        "npz": escape_dir / "biological.npz",
        "labels": np.array([types[i] for i in ids]),
        "threshold": 0.0238,
    }
    compass_model = load_compass_connectome(compass_dir)

    scenarios = ("open_healthy", "looming_post", "near_miss_post", "freeze_outage_clear", "blocked_outage", "occluded_hazard")
    policies = ("recovery_off", "ordinary_recovery", "circuit_recovery")

    rows = []
    for policy in policies:
        for sc in scenarios:
            for s in range(seeds):
                rows.append(run_episode(sc, policy, s, escape_assets, compass_model))

    out.mkdir(parents=True, exist_ok=True)
    with (out / "episodes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    agg = {}
    for p in policies:
        sub = [r for r in rows if r["policy"] == p]
        agg[p] = {
            "episodes": len(sub),
            "total_body_collisions": sum(r["body_contact_events"] for r in sub),
            "total_whisker_events": sum(r["whisker_contact_events"] for r in sub),
            "safe_success_rate": float(np.mean([r["safe_success"] for r in sub])),
            "false_emergency_stops": sum(r["false_emergency_stop"] for r in sub),
            "abstentions": sum(r["abstained"] for r in sub),
            "mean_progress_m": float(np.mean([r["progress_m"] for r in sub])),
        }
    summary = {"seeds": seeds, "scenarios": list(scenarios), "policies": agg}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=int, default=4)
    args = p.parse_args()
    print(json.dumps(run(args.out, args.seeds), indent=2))
