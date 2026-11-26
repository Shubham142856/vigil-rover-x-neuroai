"""G08 closed-loop benchmark: recovery-off vs ordinary recovery vs circuit-conditioned recovery.

Extended with:
  - neuro_cover_l policy: loads portable_weights.npz from Kaggle training (G07)
    and runs M4_BioMasked forward pass on each observation.
  - real_world policy: same as circuit_recovery but uses RealWorldEnv via
    SimInterface instead of RoverSim, exercising the hardware-interface layer.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from blindspot_dataset import ORACLE_KEYS, SectorBelief, make_paired_world
from camera_compass import bearing, fresh_heading
from camera_escape import Network, visual_looming
from compass_benchmark import ConnectomeCompass, load_compass_connectome
from real_world_env import RealWorldEnv
from recovery_policy import choose_recovery_action
from rover_sim import RoverSim
from scale_dataset import ACTIONS, extract_features

_WEIGHTS_CANDIDATES = (
    Path("runs/kaggle-training-r06/portable_weights.npz"),
    Path("runs/kaggle-training-v1/portable_weights.npz"),
)


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -30.0, 30.0))
    return 1.0 / (1.0 + float(np.exp(-x)))


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.where(x > 20.0, x, np.log1p(np.exp(np.clip(x, -50.0, 20.0))))


class _PortableMaskedPolicy:
    """Exact NumPy runtime for Kaggle-trained MaskedCircuitCore + ActionHeadPolicy."""

    def __init__(self, arrays, prefix: str):
        p = f"{prefix}__"
        adj = arrays[p + "core.adj_mask"]
        sign = arrays[p + "core.sign_mask"]
        theta = arrays[p + "core.theta"]
        self.w_eff = (adj * sign * _softplus(theta)).astype(np.float32)
        self.enc_w = arrays[p + "core.enc.weight"].astype(np.float32)
        self.enc_b = arrays[p + "core.enc.bias"].astype(np.float32)
        self.act_emb = arrays[p + "act_emb.weight"].astype(np.float32)
        self.h0_w = arrays[p + "head.0.weight"].astype(np.float32)
        self.h0_b = arrays[p + "head.0.bias"].astype(np.float32)
        self.h2_w = arrays[p + "head.2.weight"].astype(np.float32)
        self.h2_b = arrays[p + "head.2.bias"].astype(np.float32)
        self.threshold = float(arrays[p + "threshold"][0])
        self.state = np.zeros(adj.shape[0], dtype=np.float32)

    def step_core(self, feats: np.ndarray):
        drive = feats @ self.enc_w.T + self.enc_b
        self.state = self.state + 0.2 * (np.tanh(drive + self.state @ self.w_eff.T) - self.state)

    def predict_action(self, action_name: str) -> dict:
        idx = ACTIONS.index(action_name)
        inp = np.concatenate([self.state, self.act_emb[idx]])
        h0 = np.maximum(0.0, inp @ self.h0_w.T + self.h0_b)
        out = h0 @ self.h2_w.T + self.h2_b
        return {
            "contact_prob": _sigmoid(out[0]),
            "progress": float(out[1]),
            "recover_prob": _sigmoid(out[2]),
        }


def _load_neuro_cover_l(weights_path: Path = None):
    """Load frozen M4_BioMasked (risk) and M8_NeuroCOVER_L (recovery) from portable_weights.npz."""
    path = Path(weights_path) if weights_path is not None else next((p for p in _WEIGHTS_CANDIDATES if p.exists()), None)
    if path is None or not path.exists():
        return None
    arrays = np.load(path)
    if "M4_BioMasked__core.theta" not in arrays or "M8_NeuroCOVER_L__core.theta" not in arrays:
        return None
    return {
        "path": str(path),
        "m4": _PortableMaskedPolicy(arrays, "M4_BioMasked"),
        "m8": _PortableMaskedPolicy(arrays, "M8_NeuroCOVER_L"),
    }


def _make_sim(scene, seed, env_mode="rover_sim", **kwargs):
    if env_mode == "real_world":
        return RealWorldEnv(scene=scene, seed=seed, sensor_noise=True, **kwargs)
    return RoverSim(scene=scene, seed=seed, **kwargs)


R05_SCENARIOS = (
    "corridor_pedestrian",
    "warehouse_forklift_cross",
    "outdoor_pothole",
    "lab_glass_wall",
    "ramp_descent",
    "night_mode",
)


def build_scenario(name, seed, env_mode="rover_sim"):
    rng = np.random.default_rng(seed)
    if name in R05_SCENARIOS:
        return RealWorldEnv(scene=name, seed=seed, sensor_noise=True)
    elif name == "open_healthy":
        sim = _make_sim("open", seed=seed, env_mode=env_mode)
    elif name == "looming_post":
        x = float(rng.uniform(0.60, 0.78))
        sim = _make_sim("open", seed=seed, env_mode=env_mode)
        box_fn = sim._box if hasattr(sim, "_box") else sim.box
        sim.solids.append(box_fn([x, 0.0, 0.12], [0.03, 0.02, 0.12], 0, [0.86, 0.17, 0.09, 1]))
    elif name == "near_miss_post":
        x = float(rng.uniform(0.60, 0.78))
        side = 1.0 if seed % 2 == 0 else -1.0
        y = side * float(rng.uniform(0.15, 0.22))
        sim = _make_sim("open", seed=seed, env_mode=env_mode)
        box_fn = sim._box if hasattr(sim, "_box") else sim.box
        sim.solids.append(box_fn([x, y, 0.12], [0.03, 0.02, 0.12], 0, [0.86, 0.17, 0.09, 1]))
    elif name == "freeze_outage_clear":
        sim = _make_sim("left", seed=seed, env_mode=env_mode, fault="drop", fault_start_s=0.95, fault_end_s=1.45)
    elif name == "blocked_outage":
        sim = _make_sim("left", seed=seed, env_mode=env_mode, fault="freeze", fault_start_s=0.95, fault_end_s=5.0)
        box_fn = sim._box if hasattr(sim, "_box") else sim.box
        sim.solids.append(box_fn([0.55, -0.18, 0.12], [0.03, 0.05, 0.12], 0, [0.86, 0.17, 0.09, 1]))
    elif name == "occluded_hazard":
        return make_paired_world("occluded_hazard", seed=seed, fault="none", hidden_side=1)
    else:
        raise ValueError(f"Unknown scenario: {name}")
    if hasattr(sim, "_packet"):
        sim._packet = None
        sim._queue.clear()
        sim._update_camera()
    else:
        sim.packet = None
        sim.queue.clear()
        sim.update_camera()
    return sim


def run_episode(scenario, policy_mode, seed, escape_assets, compass_model, env_mode="rover_sim", weights_path=None):
    belief = SectorBelief(max_age_s=0.6)
    vstate = None
    last_cap = None
    net = Network(escape_assets["npz"], escape_assets["labels"]) if escape_assets else None
    thresh = escape_assets["threshold"] if escape_assets else 0.024
    compass = ConnectomeCompass(compass_model) if compass_model else None
    latched_escape = False
    escape_hold = 0
    abstained = False
    whisker_events = 0
    prev_whisker = False
    anchor = None
    ncl_models = _load_neuro_cover_l(weights_path) if policy_mode == "neuro_cover_l" else None
    ncl_vstate = None
    ncl_last_cap = None
    last_scan_step = -20

    with build_scenario(scenario, seed, env_mode=env_mode) as sim:
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
            elif policy_mode in ("circuit_recovery", "neuro_cover_l"):
                # Biological FAFB core: LPLC2/LC4 looming circuit + EPG/PEN ConnectomeCompass
                bio_score = 0.0
                if obs["capture_s"] != last_cap:
                    dt_bio = 0.1 if last_cap is None else max(0.05, obs["capture_s"] - last_cap)
                    lplc2_d, lc4_d, vstate = visual_looming(obs["rgb"], vstate, dt_bio)
                    if net is not None:
                        bio_score = net.update(lplc2_d, lc4_d)
                        if bio_score >= thresh:
                            if env_mode == "real_world":
                                escape_hold = 8
                            else:
                                latched_escape = True
                    if compass is not None:
                        b = bearing(obs["rgb"])
                        if anchor is None and b is not None:
                            anchor = b
                        cue = None if (b is None or anchor is None) else (b - anchor)
                        _, bump = compass.step(0.0, cue)
                        if bump < 0.25:
                            if env_mode == "real_world":
                                escape_hold = max(escape_hold, 4)
                            else:
                                latched_escape = True
                    last_cap = obs["capture_s"]

                # Leaky refractory hold in dynamic real-world scenes: clears once obstacle exits lane
                if env_mode == "real_world" and escape_hold > 0:
                    if bio_score < thresh * 0.75 and obs["obstacle_center"] < 0.08:
                        escape_hold -= 1
                looming_alert = latched_escape or (escape_hold > 0)

                m4_fwd = m8_fwd = None
                if policy_mode == "neuro_cover_l" and ncl_models is not None:
                    # Learned extension: M4_BioMasked risk + M8_NeuroCOVER_L recoverability on top of bio core
                    dt_ncl = 0.1 if ncl_last_cap is None else max(0.05, obs["time_s"] - ncl_last_cap)
                    feats, ncl_vstate = extract_features(obs, snap, ncl_vstate, dt_ncl)
                    ncl_last_cap = obs["time_s"]
                    ncl_models["m4"].step_core(feats)
                    ncl_models["m8"].step_core(feats)
                    if step >= 8:
                        m4_fwd = ncl_models["m4"].predict_action("forward")
                        m8_fwd = ncl_models["m8"].predict_action("forward")
                        # Dual-core consensus reduces false alarms on ramp pitch / lighting flicker
                        if (
                            m4_fwd["contact_prob"] >= ncl_models["m4"].threshold
                            and m8_fwd["contact_prob"] >= ncl_models["m8"].threshold
                        ):
                            looming_alert = True

                action, reason = choose_recovery_action(obs, snap, looming_alert=looming_alert)
                if "abstain" in reason:
                    abstained = True
                elif (
                    policy_mode == "neuro_cover_l"
                    and ncl_models is not None
                    and step >= 8
                    and snap["camera_healthy"]
                    and not w_any
                ):
                    states = snap["sector_states"]
                    mid = len(states) // 2
                    left_clear = all(st == "observed_free" for st in states[:mid])
                    right_clear = all(st == "observed_free" for st in states[mid + 1 :])

                    # Counterfactual evasion: if forward is blocked by looming_alert, steer into a verified-free sector
                    if action == "brake" and looming_alert and step - last_scan_step >= 4:
                        evade_cands = []
                        for s_act, is_free in (("left", left_clear), ("right", right_clear)):
                            if not is_free:
                                continue
                            p4 = ncl_models["m4"].predict_action(s_act)
                            p8 = ncl_models["m8"].predict_action(s_act)
                            if p4["contact_prob"] < ncl_models["m4"].threshold and p8["recover_prob"] >= 0.50:
                                evade_cands.append((p8["recover_prob"] + 0.5 * p8["progress"], s_act))
                        if evade_cands:
                            evade_cands.sort(reverse=True)
                            action = evade_cands[0][1]
                            last_scan_step = step
                    elif action == "forward" and snap["unknown_or_stale_fraction"] > 0.35 and step - last_scan_step >= 12:
                        scan_cands = []
                        for s_act, is_free in (("left", left_clear), ("right", right_clear)):
                            if not is_free:
                                continue
                            s_risk = ncl_models["m4"].predict_action(s_act)["contact_prob"]
                            s_recov = ncl_models["m8"].predict_action(s_act)["recover_prob"]
                            if s_risk < ncl_models["m4"].threshold and s_recov >= 0.55:
                                scan_cands.append((s_recov, s_act))
                        if scan_cands:
                            scan_cands.sort(reverse=True)
                            action = scan_cands[0][1]
                            last_scan_step = step
                    elif (
                        action == "forward"
                        and m4_fwd is not None
                        and m8_fwd is not None
                        and m4_fwd["contact_prob"] >= 0.5 * ncl_models["m4"].threshold
                        and m8_fwd["recover_prob"] < 0.65
                    ):
                        m4_slow = ncl_models["m4"].predict_action("slow")
                        if m4_slow["contact_prob"] < ncl_models["m4"].threshold:
                            action = "slow"
            else:
                raise ValueError(policy_mode)

            obs = sim.step(action)
        ev = sim.evaluator()

    navigable = scenario in ("open_healthy", "near_miss_post")
    false_stop = navigable and ev["x_m"] < 0.90 and ev["body_contact_events"] == 0 and not abstained
    if scenario in ("open_healthy", "near_miss_post"):
        safe_success = ev["body_contact_events"] == 0 and ev["x_m"] >= 0.90
    elif scenario in ("looming_post", "occluded_hazard") or scenario in R05_SCENARIOS:
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


def run(out: Path, seeds=4, env_mode="rover_sim", suite="g08", weights_path=None):
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

    if suite == "r05":
        scenarios = R05_SCENARIOS
        env_mode = "real_world"
    else:
        scenarios = ("open_healthy", "looming_post", "near_miss_post", "freeze_outage_clear", "blocked_outage", "occluded_hazard")
    policies = ("recovery_off", "ordinary_recovery", "circuit_recovery", "neuro_cover_l")

    rows = []
    for policy in policies:
        for sc in scenarios:
            for s in range(seeds):
                rows.append(run_episode(sc, policy, s, escape_assets, compass_model, env_mode=env_mode, weights_path=weights_path))

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
    resolved_weights = (
        str(weights_path)
        if weights_path is not None
        else next((str(p) for p in _WEIGHTS_CANDIDATES if p.exists()), None)
    )
    summary = {"seeds": seeds, "suite": suite, "env_mode": env_mode, "weights_path": resolved_weights, "scenarios": list(scenarios), "policies": agg}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--env", choices=["rover_sim", "real_world"], default="rover_sim")
    p.add_argument("--suite", choices=["g08", "r05"], default="g08")
    p.add_argument("--weights", type=Path, default=None)
    args = p.parse_args()
    print(json.dumps(run(args.out, args.seeds, env_mode=args.env, suite=args.suite, weights_path=args.weights), indent=2))
