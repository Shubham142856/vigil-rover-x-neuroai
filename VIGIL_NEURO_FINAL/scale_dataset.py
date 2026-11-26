"""Scaled counterfactual navigation dataset across world geometries and held-out faults."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from blindspot_dataset import ORACLE_KEYS, SectorBelief, make_paired_world
from branch_rollout import compare_actions
from camera_escape import visual_looming
from real_world_env import RealWorldEnv
from rover_sim import RoverSim

ACTIONS = ("forward", "slow", "brake", "left", "right")
R05_FAMILIES = (
    "corridor_pedestrian",
    "warehouse_forklift_cross",
    "outdoor_pothole",
    "lab_glass_wall",
    "ramp_descent",
    "night_mode",
)


def build_world(family, seed, fault):
    """Instantiate diverse PyBullet geometries keeping oracle state separate."""
    rng = np.random.default_rng(seed)
    if family in R05_FAMILIES:
        return RealWorldEnv(scene=family, seed=seed, fault=fault, fault_start_s=0.35, fault_end_s=0.95, sensor_noise=True)
    elif family == "corridor":
        sim = RoverSim(scene="open", seed=seed, fault=fault, fault_start_s=0.35, fault_end_s=0.95)
        sim.solids.append(sim.box([0.9, 0.24, 0.12], [0.8, 0.02, 0.12], 0, [0.45, 0.48, 0.52, 1]))
        sim.solids.append(sim.box([0.9, -0.24, 0.12], [0.8, 0.02, 0.12], 0, [0.45, 0.48, 0.52, 1]))
        if seed % 2 == 0:
            y = float(rng.uniform(-0.04, 0.04))
            sim.solids.append(sim.box([float(rng.uniform(0.58, 0.85)), y, 0.12], [0.03, 0.025, 0.12], 0, [0.86, 0.17, 0.09, 1]))
    elif family == "shelf_clear":
        return make_paired_world("occluded_clear", seed=seed, fault=fault, hidden_side=1 if seed % 2 else -1)
    elif family == "shelf_hazard":
        return make_paired_world("occluded_hazard", seed=seed, fault=fault, hidden_side=1 if seed % 2 else -1)
    elif family == "post_sweep":
        sim = RoverSim(scene="open", seed=seed, fault=fault, fault_start_s=0.35, fault_end_s=0.95)
        x = float(rng.uniform(0.52, 0.85))
        side = 1.0 if seed % 2 == 0 else -1.0
        y = side * (float(rng.uniform(0.0, 0.02)) if (seed // 2) % 2 == 0 else float(rng.uniform(0.14, 0.22)))
        sim.solids.append(sim.box([x, y, 0.12], [0.03, 0.02, 0.12], 0, [0.86, 0.17, 0.09, 1]))
    else:
        raise ValueError(f"Unknown world family: {family}")
    sim.packet = None
    sim.queue.clear()
    sim.update_camera()
    return sim


def extract_features(obs, belief_snap, looming_state, dt):
    """Build strictly sensor-only 16-dim feature vector (zero oracle fields)."""
    if not ORACLE_KEYS.isdisjoint(obs):
        raise AssertionError("Oracle key leaked into observation")
    lplc2_d, lc4_d, next_state = visual_looming(obs["rgb"], looming_state, dt)
    feats = np.array(
        [
            float(obs["frame_valid"]),
            float(obs["frame_age_s"] if obs["frame_age_s"] is not None else 1.0),
            float(obs["brightness"]),
            float(obs["contrast"]),
            float(obs["obstacle_center"]),
            float(obs["looming_center"]),
            float(obs["whisker_left"]),
            float(obs["whisker_right"]),
            float(lplc2_d),
            float(lc4_d),
            float(belief_snap["camera_healthy"]),
            float(belief_snap["free_fraction"]),
            float(belief_snap["unknown_or_stale_fraction"]),
            float(sum(s == "occupied" for s in belief_snap["sector_states"]) / len(belief_snap["sector_states"])),
            float(obs["time_s"]),
            float(dt),
        ],
        dtype=np.float32,
    )
    return feats, next_state


def generate_shards(out: Path, episodes_per_split=None, suite="g07"):
    """Generate world-split counterfactual training shards (.npz + manifest.json)."""
    if episodes_per_split is None:
        g07_splits = {
            "train": [("post_sweep", "none", s) for s in range(100, 112)]
            + [("shelf_clear", "none", s) for s in range(120, 126)]
            + [("shelf_hazard", "none", s) for s in range(120, 126)]
            + [("corridor", "freeze", s) for s in range(130, 138)],
            "val": [("post_sweep", "freeze", s) for s in range(200, 206)]
            + [("shelf_clear", "delay", s) for s in range(210, 213)]
            + [("shelf_hazard", "delay", s) for s in range(210, 213)],
            "test": [("corridor", "drop", s) for s in range(300, 306)]
            + [("shelf_clear", "freeze", s) for s in range(310, 313)]
            + [("shelf_hazard", "freeze", s) for s in range(310, 313)],
        }
        r05_splits = {
            "train": [(f, fault, 400 + s) for f in R05_FAMILIES for fault in ("none", "freeze") for s in range(3)],
            "val": [(f, "delay", 500 + s) for f in R05_FAMILIES for s in range(2)],
            "test": [(f, "drop", 600 + s) for f in R05_FAMILIES for s in range(2)],
        }
        if suite == "r05":
            episodes_per_split = r05_splits
        elif suite == "combined":
            episodes_per_split = {k: g07_splits[k] + r05_splits[k] for k in ("train", "val", "test")}
        else:
            episodes_per_split = g07_splits

    out.mkdir(parents=True, exist_ok=True)
    split_arrays = {}
    state_split_map = {}

    for split, specs in episodes_per_split.items():
        seqs, act_ids, y_contact, y_prog, y_recov = [], [], [], [], []
        for family, fault, seed in specs:
            state_id = f"{family}:{fault}:{seed}"
            state_split_map.setdefault(state_id, set()).add(split)
            with build_world(family, seed, fault) as sim:
                belief = SectorBelief(max_age_s=0.45)
                vstate = None
                hist = []
                prev_t = 0.0
                for step in range(22):
                    obs = sim.observe()
                    dt = max(0.05, obs["time_s"] - prev_t)
                    prev_t = obs["time_s"]
                    snap = belief.update(obs)
                    feat, vstate = extract_features(obs, snap, vstate, dt)
                    hist.append(feat)
                    if step in (12, 16, 20):
                        branches = compare_actions(sim, list(ACTIONS), 28)
                        seq = np.zeros((24, 16), dtype=np.float32)
                        seq[-len(hist) :] = np.stack(hist)
                        for a_idx, act in enumerate(ACTIONS):
                            res = branches[act]
                            seqs.append(seq)
                            act_ids.append(a_idx)
                            y_contact.append(1.0 if res["body_contact"] else 0.0)
                            y_prog.append(float(res["forward_progress_m"]))
                            y_recov.append(1.0 if (not res["body_contact"] and abs(res["heading_change_rad"]) > 0.2) else 0.0)
                    sim.step("forward")

        arr_path = out / f"{split}.npz"
        np.savez_compressed(
            arr_path,
            X=np.asarray(seqs, dtype=np.float32),
            action=np.asarray(act_ids, dtype=np.int64),
            y_contact=np.asarray(y_contact, dtype=np.float32),
            y_progress=np.asarray(y_prog, dtype=np.float32),
            y_recover=np.asarray(y_recov, dtype=np.float32),
        )
        split_arrays[split] = {
            "samples": len(seqs),
            "positive_contact_rate": float(np.mean(y_contact)) if y_contact else 0.0,
            "sha256": hashlib.sha256(arr_path.read_bytes()).hexdigest(),
        }

    leakage = sum(len(s) != 1 for s in state_split_map.values())
    if leakage:
        raise AssertionError("World-state split leakage detected")

    manifest = {
        "feature_dim": 16,
        "seq_len": 24,
        "actions": list(ACTIONS),
        "oracle_keys_in_inputs": 0,
        "world_split_leakage": leakage,
        "splits": split_arrays,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--suite", choices=["g07", "r05", "combined"], default="g07")
    args = p.parse_args()
    print(json.dumps(generate_shards(args.out, suite=args.suite), indent=2))
