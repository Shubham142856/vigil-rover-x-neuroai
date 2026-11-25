"""G06 blind-spot dataset: indistinguishable hidden worlds, observation aging, and split manifest."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from branch_rollout import compare_actions
from rover_sim import RoverSim

ORACLE_KEYS = frozenset(
    {
        "x_m",
        "y_m",
        "yaw_rad",
        "speed_m_s",
        "body_contact",
        "body_contact_events",
        "scene",
        "fault",
        "hidden_present",
        "hidden_side",
        "world_variant",
        "future_contact",
    }
)


class SectorBelief:
    """Egocentric sector visibility memory with conservative occlusion and observation aging."""

    def __init__(self, n_sectors=9, fov_rad=np.radians(78.0), max_age_s=0.5):
        self.n_sectors = n_sectors
        self.edges = np.linspace(-fov_rad / 2, fov_rad / 2, n_sectors + 1)
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.max_age_s = max_age_s
        self.last_free_s = np.full(n_sectors, -np.inf, dtype=float)
        self.occupied = np.zeros(n_sectors, dtype=bool)
        self.occluded = np.zeros(n_sectors, dtype=bool)

    def update(self, obs):
        now = float(obs["time_s"])
        fresh = (
            obs.get("frame_valid") is True
            and obs.get("frame_age_s") is not None
            and float(obs["frame_age_s"]) <= 0.051
        )
        if not fresh:
            return self.snapshot(now, camera_healthy=False)

        rgb = obs["rgb"]
        r, g, b = rgb.astype(float).transpose(2, 0, 1)
        mask = (r > 60) & (r > 1.7 * g) & (r > 1.7 * b)
        col_hit = mask.any(axis=0)
        focal = rgb.shape[0] / (2 * np.tan(np.radians(66) / 2))
        center = (rgb.shape[1] - 1) / 2
        col_angles = np.arctan((np.arange(rgb.shape[1]) - center) / focal)

        self.occupied.fill(False)
        self.occluded.fill(False)
        for s in range(self.n_sectors):
            in_sector = (col_angles >= self.edges[s]) & (col_angles <= self.edges[s + 1])
            if in_sector.any() and col_hit[in_sector].any():
                self.occupied[s] = True
                self.occluded[s] = True
                self.last_free_s[s] = -np.inf
            else:
                self.last_free_s[s] = now
        return self.snapshot(now, camera_healthy=True)

    def snapshot(self, now, camera_healthy):
        ages = now - self.last_free_s
        states = []
        for s in range(self.n_sectors):
            if self.occupied[s]:
                states.append("occupied")
            elif not np.isfinite(ages[s]) or ages[s] > self.max_age_s:
                states.append("stale_or_unknown")
            else:
                states.append("observed_free")
        return {
            "camera_healthy": bool(camera_healthy),
            "sector_states": states,
            "sector_ages_s": [None if not np.isfinite(a) else round(float(a), 4) for a in ages],
            "free_fraction": float(sum(st == "observed_free" for st in states) / self.n_sectors),
            "unknown_or_stale_fraction": float(sum(st != "observed_free" for st in states) / self.n_sectors),
        }


def make_paired_world(variant, seed=7, fault="none", hidden_side=1):
    """Build paired worlds whose initial view behind an occluder is identical."""
    sim = RoverSim(scene="open", seed=seed, fault=fault, fault_start_s=0.35, fault_end_s=0.85)
    # Foreground occluder centered at x=0.55 blocking view of region behind it.
    sim.solids.append(sim.box([0.55, 0.0, 0.12], [0.03, 0.14, 0.12], 0, [0.86, 0.17, 0.09, 1]))
    if variant == "occluded_hazard":
        # Hidden obstacle tucked behind the occluder on one flank (invisible from start).
        sim.solids.append(sim.box([0.72, 0.12 * hidden_side, 0.12], [0.03, 0.03, 0.12], 0, [0.86, 0.17, 0.09, 1]))
    elif variant != "occluded_clear":
        raise ValueError(f"Unknown world variant: {variant}")
    sim.packet = None
    sim.queue.clear()
    sim.update_camera()
    return sim


def run_paired_check(seed=7, steps_before_branch=6, hidden_side=1):
    """Verify exact pre-branch RGB indistinguishability and distinct branch outcomes."""
    histories = {}
    branches = {}
    for variant in ("occluded_clear", "occluded_hazard"):
        with make_paired_world(variant, seed=seed, hidden_side=hidden_side) as sim:
            belief = SectorBelief()
            frames = []
            obs = sim.observe()
            belief_snap = belief.update(obs)
            frames.append(obs["rgb"].copy())
            for _ in range(steps_before_branch):
                obs = sim.step("forward")
                belief_snap = belief.update(obs)
                frames.append(obs["rgb"].copy())
            histories[variant] = frames
            branches[variant] = {
                "belief": belief_snap,
                "obs": {k: v for k, v in obs.items() if k != "rgb"},
                "actions": compare_actions(sim, ["forward", "slow", "brake", "left", "right"], 20),
            }
    max_pixel_diff = max(
        int(np.max(np.abs(a.astype(int) - b.astype(int))))
        for a, b in zip(histories["occluded_clear"], histories["occluded_hazard"])
    )
    return max_pixel_diff, branches


def generate_dataset(out, seeds_per_split=None):
    if seeds_per_split is None:
        seeds_per_split = {"train": [101, 102, 103, 104], "val": [201, 202], "test": [301, 302]}

    out.mkdir(parents=True, exist_ok=True)
    records = []
    max_history_diff = 0
    for split, seeds in seeds_per_split.items():
        for seed in seeds:
            pair_id = f"pair-seed-{seed}"
            hidden_side = 1 if seed % 2 else -1
            max_diff, branches = run_paired_check(seed=seed, steps_before_branch=6, hidden_side=hidden_side)
            max_history_diff = max(max_history_diff, max_diff)
            for variant, info in branches.items():
                # Strictly sanitize inputs: prohibit all oracle keys.
                inputs = {k: v for k, v in info["obs"].items() if k not in ORACLE_KEYS}
                inputs["sector_belief"] = info["belief"]
                if not ORACLE_KEYS.isdisjoint(inputs):
                    raise AssertionError("Oracle key leaked into inputs")
                for action, outcome in info["actions"].items():
                    rec = {
                        "split": split,
                        "pair_id": pair_id,
                        "world_variant": variant,
                        "action": action,
                        "inputs": inputs,
                        "oracle_labels": {
                            "hidden_present": variant == "occluded_hazard",
                            "hidden_side": hidden_side if variant == "occluded_hazard" else 0,
                            **outcome,
                        },
                    }
                    records.append(rec)

    rec_path = out / "records.jsonl"
    with rec_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Verify pair-level split isolation: no pair_id appears in more than one split.
    pair_splits = {}
    for r in records:
        pair_splits.setdefault(r["pair_id"], set()).add(r["split"])
    if any(len(s) != 1 for s in pair_splits.values()):
        raise AssertionError("Split leakage across paired branches")

    manifest = {
        "total_records": len(records),
        "pairs": len(pair_splits),
        "splits": {k: len(v) for k, v in seeds_per_split.items()},
        "max_pre_branch_pixel_diff": int(max_history_diff),
        "indistinguishable_histories_verified": bool(max_history_diff == 0),
        "oracle_keys_in_inputs": 0,
        "pair_split_leakage": 0,
        "records_sha256": hashlib.sha256(rec_path.read_bytes()).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    print(json.dumps(generate_dataset(p.parse_args().out), indent=2))
