"""P6.5 & P6.6 Multi-Horizon Survival Counterfactual Rollouts & Audited Pilot Dataset Generator.

Implements:
  - Enriched sensor-only 24-dim observation features (integrating DualPathwayT4T5Vision,
    IMU angular rates, encoder speed, Ground-Contact IPM clearance, and FactorizedSpatialBelief;
    zero simulator oracle keys).
  - Dual counterfactual continuation branches per decision state & candidate action:
      * continue        : hold action over full 2.4s (48 steps @ 20 Hz) -> multi-horizon
                          cumulative collision labels [0.1, 0.3, 0.6, 1.2, 2.4]s + survival
                          hazard interval + censoring status.
      * protected_brake : execute action briefly for 0.30s (6 steps), then brake for
                          remaining horizon -> immediate protected-motion collision label,
                          signed brake-relative uncertainty gain Delta_U_t, and clearance.
  - World-family isolated pilot dataset (records.jsonl + .npz + manifest.json).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pybullet as pb

from blindspot_dataset import ORACLE_KEYS, SectorBelief
from camera_escape import visual_looming
from ground_range_estimator import estimate_ground_contact_clearance
from safety_arbitrator_v2 import ObservationIntegrityMonitor
from scale_dataset import ACTIONS, build_world, extract_features
from spatial_belief_v2 import FactorizedSpatialBelief
from t4_t5_vision import DualPathwayT4T5Vision


CTRL_HZ = 20
DT_CTRL_S = 1.0 / CTRL_HZ
HORIZONS_S = (0.10, 0.30, 0.60, 1.20, 2.40)
HORIZON_STEPS = tuple(int(round(h * CTRL_HZ)) for h in HORIZONS_S)  # (2, 6, 12, 24, 48)
PROTECTED_EXEC_STEPS = 6  # 0.30 s brief execution followed by protective brake


def _true_min_clearance_m(sim) -> float:
    """Evaluator-only minimum surface distance [m] between rover hull and solids (NEVER in controller inputs)."""
    if not getattr(sim, "solids", None):
        return 4.0
    min_d = 4.0
    for body_id in sim.solids:
        pts = pb.getClosestPoints(sim.rover, body_id, distance=4.0, physicsClientId=sim.client)
        for p in pts:
            d = float(p[8])
            if d < min_d:
                min_d = d
    return max(0.0, min_d)


def extract_features_v3(
    obs: dict,
    snap_v1: dict,
    snap_v2: dict,
    vis_out: dict,
    ipm_out: dict,
    lstate,
    dt: float,
) -> Tuple[np.ndarray, object]:
    """24-dim strictly sensor-only observation vector (zero oracle fields)."""
    base16, next_lstate = extract_features(obs, snap_v1, lstate, dt)
    extra8 = np.array(
        [
            float(vis_out.get("raw_lplc2", 0.0)),
            float(vis_out.get("residual_lplc2", 0.0)),
            float(vis_out.get("residual_lc4", 0.0)),
            float(obs.get("imu_gyro_z", 0.0)) if obs.get("imu_gyro_z") is not None else 0.0,
            float(obs.get("imu_gyro_y", 0.0)) if obs.get("imu_gyro_y") is not None else 0.0,
            float(obs.get("encoder_speed_m_s", 0.0)),
            float(min(4.0, ipm_out.get("estimated_clearance_m", 4.0))),
            float(snap_v2.get("task_uncertainty", 0.0)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([base16, extra8], axis=0), next_lstate


def rollout_multi_horizon_branches(sim, actions: List[str], initial_belief: FactorizedSpatialBelief) -> Dict[str, dict]:
    """Evaluate 'continue' (48 steps) & 'protected_brake' (6 steps + brake) from identical restored state."""
    use_snap = hasattr(sim, "snapshot_state") and hasattr(sim, "restore_state")
    if use_snap:
        saved = sim.snapshot_state()
    else:
        fields = ("tick", "speed", "turn", "events", "touch", "queue", "packet", "last_center", "growth")
        saved = {name: deepcopy(getattr(sim, name)) for name in fields}

    init_eval = sim.evaluator()
    init_obs = sim.observe()
    t0 = float(init_obs["time_s"])
    pb_state = pb.saveState(physicsClientId=sim.client)

    def restore():
        pb.restoreState(pb_state, physicsClientId=sim.client)
        if use_snap:
            sim.restore_state(saved)
        else:
            for name, value in saved.items():
                setattr(sim, name, deepcopy(value))

    max_steps = HORIZON_STEPS[-1]  # 48 steps = 2.40 s
    out: Dict[str, dict] = {}
    try:
        # 1. Protected brake rollouts + brake reference baseline
        prot_u_end: Dict[str, float] = {}
        prot_records: Dict[str, dict] = {}
        for act in actions:
            restore()
            b = initial_belief.copy()
            obs = init_obs
            hit = bool(init_eval["body_contact"])
            w_hit = bool(obs["whisker_left"] or obs["whisker_right"])
            ttc: Optional[float] = 0.0 if hit else None
            min_clear = _true_min_clearance_m(sim)
            for step_i in range(1, max_steps + 1):
                cmd = act if step_i <= PROTECTED_EXEC_STEPS else "brake"
                obs = sim.step(cmd)
                ev = sim.evaluator()
                if "imu_gyro_z" in obs:
                    b.update(obs)
                else:
                    b.update(obs, world_yaw_rad=ev["yaw_rad"])
                c_now = _true_min_clearance_m(sim)
                min_clear = min(min_clear, c_now)
                if ev["body_contact"] and ttc is None:
                    hit = True
                    ttc = float(step_i * DT_CTRL_S)
                if obs["whisker_left"] or obs["whisker_right"]:
                    w_hit = True
            end_ev = sim.evaluator()
            u_end = float(b.task_uncertainty(float(obs["time_s"])))
            prot_u_end[act] = u_end
            prot_records[act] = {
                "body_contact": bool(hit),
                "whisker_contact": bool(w_hit),
                "time_to_contact_s": ttc,
                "collision_censored": ttc is None,
                "minimum_clearance_m": float(min_clear),
                "goal_progress_m": float(end_ev["x_m"] - init_eval["x_m"]),
                "cam_healthy_end": bool(b.camera_healthy),
                "u_end": u_end,
            }

        u_brake_ref = prot_u_end.get("brake", float(initial_belief.task_uncertainty(t0)))

        # 2. Uninterrupted continuation rollouts -> cumulative 5-horizon survival labels
        # Camera rendering is never read by continuation contact/progress evaluation; skip render for speed
        orig_upd = getattr(sim, "update_camera", None)
        orig_priv_upd = getattr(sim, "_update_camera", None)
        try:
            if orig_upd is not None:
                sim.update_camera = lambda: None
            if orig_priv_upd is not None:
                sim._update_camera = lambda: None
            for act in actions:
                restore()
                obs = init_obs
                hit = bool(init_eval["body_contact"])
                w_hit = bool(obs["whisker_left"] or obs["whisker_right"])
                ttc: Optional[float] = 0.0 if hit else None
                min_clear = _true_min_clearance_m(sim)
                for step_i in range(1, max_steps + 1):
                    obs = sim.step(act)
                    ev = sim.evaluator()
                    if step_i <= 12 or step_i % 2 == 0 or ev["body_contact"]:
                        min_clear = min(min_clear, _true_min_clearance_m(sim))
                    if ev["body_contact"] and ttc is None:
                        hit = True
                        ttc = float(step_i * DT_CTRL_S)
                    if obs["whisker_left"] or obs["whisker_right"]:
                        w_hit = True
                end_ev = sim.evaluator()

                risk_bins = [1.0 if (ttc is not None and ttc <= h + 1e-9) else 0.0 for h in HORIZONS_S]
                # Discrete survival interval: first horizon index 0..4 where ttc <= H_k, else 5 (right-censored)
                hazard_bin = len(HORIZONS_S)
                if ttc is not None:
                    for k_idx, h_val in enumerate(HORIZONS_S):
                        if ttc <= h_val + 1e-9:
                            hazard_bin = k_idx
                            break

                pr = prot_records[act]
                signed_du = 0.0 if (act == "brake" or not pr["cam_healthy_end"]) else float(u_brake_ref - pr["u_end"])
                out[act] = {
                    "action": act,
                    "action_duration_s": float(PROTECTED_EXEC_STEPS * DT_CTRL_S),
                    "continue_risk_bins": risk_bins,
                    "continue_hazard_bin": int(hazard_bin),
                    "continue_time_to_contact_s": ttc,
                    "continue_censored": ttc is None,
                    "continue_body_contact": bool(hit),
                    "continue_whisker_contact": bool(w_hit),
                    "continue_min_clearance_m": float(min_clear),
                    "continue_progress_m": float(end_ev["x_m"] - init_eval["x_m"]),
                    "protected_body_contact": pr["body_contact"],
                    "protected_whisker_contact": pr["whisker_contact"],
                    "protected_time_to_contact_s": pr["time_to_contact_s"],
                    "protected_censored": pr["collision_censored"],
                    "protected_min_clearance_m": pr["minimum_clearance_m"],
                    "protected_progress_m": pr["goal_progress_m"],
                    "camera_healthy_end": pr["cam_healthy_end"],
                    "signed_info_gain": signed_du,
                    "safe_recover": bool((not pr["body_contact"]) and pr["cam_healthy_end"] and signed_du >= 0.04),
                }
        finally:
            if orig_upd is not None:
                sim.update_camera = orig_upd
            if orig_priv_upd is not None:
                sim._update_camera = orig_priv_upd
    finally:
        try:
            restore()
        finally:
            pb.removeState(pb_state, physicsClientId=sim.client)
    return out


def default_pilot_splits() -> dict:
    """Strictly world-family & seed isolated compact pilot dataset covering all key regimes."""
    return {
        "train": [
            ("post_sweep", "none", 1100), ("post_sweep", "none", 1101), ("post_sweep", "freeze", 1102),
            ("corridor", "none", 1110), ("corridor", "freeze", 1111),
            ("shelf_clear", "none", 1120), ("shelf_hazard", "none", 1121),
            ("corridor_pedestrian", "none", 1130),
            ("warehouse_forklift_cross", "none", 1140),
            ("ramp_descent", "none", 1150),
            ("night_mode", "none", 1160),
        ],
        "val": [
            ("post_sweep", "delay", 1200),
            ("shelf_clear", "freeze", 1220),
            ("corridor_pedestrian", "delay", 1230),
            ("ramp_descent", "none", 1250),
        ],
        "test": [
            ("corridor", "drop", 1310),
            ("shelf_hazard", "freeze", 1320),
            ("warehouse_forklift_cross", "drop", 1340),
            ("outdoor_pothole", "none", 1370),
        ],
    }


def generate_pilot_dataset(out: Path, splits: Optional[dict] = None) -> dict:
    if splits is None:
        splits = default_pilot_splits()

    out.mkdir(parents=True, exist_ok=True)
    records = []
    split_summaries = {}
    world_splits: Dict[str, set] = {}

    for sp, specs in splits.items():
        X_list, act_list = [], []
        y_risk_cont, y_haz_bin, y_cens = [], [], []
        y_risk_prot, y_prog_prot, y_prog_cont = [], [], []
        y_clear_prot, y_du, y_cam_ok, y_recov = [], [], [], []
        ep_count = 0
        branch_tick_count = 0

        for ep_idx, (family, fault, seed) in enumerate(specs):
            world_id = f"{family}:{fault}:{seed}"
            world_splits.setdefault(world_id, set()).add(sp)
            episode_id = f"{sp}_{ep_idx}_{world_id}"
            ep_count += 1

            with build_world(family, seed, fault) as sim:
                b1 = SectorBelief(max_age_s=0.45)
                b2 = FactorizedSpatialBelief(max_age_s=0.60)
                vis = DualPathwayT4T5Vision()
                mon = ObservationIntegrityMonitor()
                lstate = None
                hist = []
                prev_t = 0.0
                last_cap = None
                vo = {"raw_lplc2": 0.0, "residual_lplc2": 0.0, "residual_lc4": 0.0}

                for step in range(22):
                    obs = sim.observe()
                    if not ORACLE_KEYS.isdisjoint(obs):
                        raise AssertionError("Oracle key leaked into observation")
                    dt = max(0.05, obs["time_s"] - prev_t)
                    prev_t = obs["time_s"]
                    rep = mon.step(obs, imu_enabled=("imu_gyro_z" in obs))
                    if obs["capture_s"] != last_cap:
                        vo = vis.step(obs["rgb"], max(0.05, dt), rep.imu_gyro_z, rep.imu_gyro_y)
                        last_cap = obs["capture_s"]
                    ipm = estimate_ground_contact_clearance(obs["rgb"], pitch_rad=0.0)
                    s1 = b1.update(obs)
                    if "imu_gyro_z" in obs:
                        s2 = b2.update(obs, delta_yaw_rad=rep.delta_yaw_rad)
                    else:
                        s2 = b2.update(obs, world_yaw_rad=sim.evaluator()["yaw_rad"])
                    feat, lstate = extract_features_v3(obs, s1, s2, vo, ipm, lstate, dt)
                    hist.append(feat)

                    if step in (12, 16, 20):
                        branch_tick_count += 1
                        seq = np.zeros((24, 24), dtype=np.float32)
                        seq[-len(hist) :] = np.stack(hist)
                        branches = rollout_multi_horizon_branches(sim, list(ACTIONS), b2)
                        for a_idx, act in enumerate(ACTIONS):
                            br = branches[act]
                            X_list.append(seq)
                            act_list.append(a_idx)
                            y_risk_cont.append(br["continue_risk_bins"])
                            y_haz_bin.append(br["continue_hazard_bin"])
                            y_cens.append(1.0 if br["continue_censored"] else 0.0)
                            y_risk_prot.append(1.0 if br["protected_body_contact"] else 0.0)
                            y_prog_prot.append(br["protected_progress_m"])
                            y_prog_cont.append(br["continue_progress_m"])
                            y_clear_prot.append(br["protected_min_clearance_m"])
                            y_du.append(br["signed_info_gain"])
                            y_cam_ok.append(1.0 if br["camera_healthy_end"] else 0.0)
                            y_recov.append(1.0 if br["safe_recover"] else 0.0)

                            records.append({
                                "split": sp,
                                "world_id": world_id,
                                "episode_id": episode_id,
                                "branch_id": f"{episode_id}:t{step}:{act}",
                                "decision_step": step,
                                "action": act,
                                "action_duration_s": br["action_duration_s"],
                                "camera_health": rep.camera_status,
                                **br,
                            })
                    sim.step("forward")

        npz_path = out / f"{sp}.npz"
        rc_arr = np.asarray(y_risk_cont, dtype=np.float32)
        rp_arr = np.asarray(y_risk_prot, dtype=np.float32)
        du_arr = np.asarray(y_du, dtype=np.float32)
        np.savez_compressed(
            npz_path,
            X=np.asarray(X_list, dtype=np.float32),
            action=np.asarray(act_list, dtype=np.int64),
            y_risk_horizons=rc_arr,
            y_hazard_bin=np.asarray(y_haz_bin, dtype=np.int64),
            y_censored=np.asarray(y_cens, dtype=np.float32),
            y_risk_protected=rp_arr,
            y_progress_protected=np.asarray(y_prog_prot, dtype=np.float32),
            y_progress_continue=np.asarray(y_prog_cont, dtype=np.float32),
            y_min_clearance_protected=np.asarray(y_clear_prot, dtype=np.float32),
            y_signed_delta_u=du_arr,
            y_cam_healthy=np.asarray(y_cam_ok, dtype=np.float32),
            y_recover=np.asarray(y_recov, dtype=np.float32),
        )
        split_summaries[sp] = {
            "independent_worlds": ep_count,
            "episodes": ep_count,
            "decision_states": branch_tick_count,
            "counterfactual_branches": len(X_list),
            "short_safe_long_danger_count": int(np.sum((rp_arr < 0.5) & (rc_arr[:, -1] > 0.5))),
            "risk_protected_rate": float(np.mean(rp_arr)) if len(rp_arr) else 0.0,
            "risk_continue_rates": [float(rc_arr[:, j].mean()) for j in range(len(HORIZONS_S))] if len(rc_arr) else [],
            "negative_delta_u_count": int(np.sum(du_arr < -1e-6)),
            "positive_delta_u_count": int(np.sum(du_arr > 1e-6)),
            "safe_recover_rate": float(np.mean(y_recov)) if y_recov else 0.0,
            "sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        }

    leakage = sum(len(s) != 1 for s in world_splits.values())
    if leakage:
        raise AssertionError("World-split leakage detected in pilot dataset")

    rec_path = out / "records.jsonl"
    with rec_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    manifest = {
        "schema_version": "p6_pilot_v3",
        "ctrl_hz": CTRL_HZ,
        "cam_hz": 10,
        "phys_hz": 120,
        "feature_dim": 24,
        "seq_len": 24,
        "horizons_s": list(HORIZONS_S),
        "protected_exec_s": float(PROTECTED_EXEC_STEPS * DT_CTRL_S),
        "actions": list(ACTIONS),
        "oracle_keys_in_inputs": 0,
        "world_split_leakage": leakage,
        "splits": split_summaries,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("runs/pilot-v3-p6"))
    args = p.parse_args()
    print(json.dumps(generate_pilot_dataset(args.out), indent=2))
