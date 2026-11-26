"""VIGIL-ROVER X P6.8 Closed-Loop Pilot & Targeted Interaction Ablation.

Evaluates P6 methodology fixes (ground-contact IPM + adaptive braking,
66-deg ego FOV window fix, multi-horizon survival + immediate protected risk,
signed counterfactual delta_u) in closed-loop PyBullet G08 and R05 suites
against frozen P5/V1 baselines and architecture controls (Bio vs Rewired-7 vs GRU).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from blindspot_dataset import ORACLE_KEYS, SectorBelief
from camera_compass import bearing
from camera_escape import Network
from closed_loop_benchmark import R05_SCENARIOS, build_scenario
from compass_benchmark import ConnectomeCompass, load_compass_connectome
from evaluation_metrics_v2 import aggregate_episodes, classify_episode
from ground_range_estimator import estimate_ground_contact_clearance
from p5_causal_ablation import run_causal_episode
from pilot_dataset_v3 import ACTIONS, extract_features_v3
from safety_arbitrator_v2 import (
    ActionFeasibilityChecker,
    ArbitrationConfig,
    ObservationIntegrityMonitor,
    select_protected_action_p5,
)
from spatial_belief_v2 import FactorizedSpatialBelief
from t4_t5_vision import DualPathwayT4T5Vision


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.where(x > 20.0, x, np.log1p(np.exp(np.clip(x, -30.0, 20.0))))


class _PortableV3Policy:
    """Vectorized NumPy evaluator matching pilot_train_v3 forward pass."""

    def __init__(self, npz_path: Path, run_name: str, variant: str):
        arr = np.load(npz_path, allow_pickle=False)
        self.run_name = run_name
        self.variant = variant
        self.feat_mean = arr["feat_mean"].astype(np.float64)
        self.feat_std = np.maximum(arr["feat_std"].astype(np.float64), 1e-4)

        prefix = f"{run_name}__"
        self.core_type = str(arr[f"{prefix}core_type"].item())
        self.act_emb = arr[f"{prefix}act_emb"].astype(np.float64)
        self.h0_w = arr[f"{prefix}h0_w"].astype(np.float64)
        self.h0_b = arr[f"{prefix}h0_b"].astype(np.float64)
        self.h2_w = arr[f"{prefix}h2_w"].astype(np.float64)
        self.h2_b = arr[f"{prefix}h2_b"].astype(np.float64)

        if self.core_type in ("bio", "rewired"):
            adj = arr[f"{prefix}adj"].astype(np.float64)
            sign = arr[f"{prefix}sign"].astype(np.float64)
            theta = arr[f"{prefix}theta"].astype(np.float64)
            self.w_eff = adj * sign * _softplus(theta)
            self.enc_w = arr[f"{prefix}enc_w"].astype(np.float64)
            self.enc_b = arr[f"{prefix}enc_b"].astype(np.float64)
            self.n_nodes = self.w_eff.shape[0]
        elif self.core_type == "gru":
            self.x_w = arr[f"{prefix}x_w"].astype(np.float64)
            self.h_w = arr[f"{prefix}h_w"].astype(np.float64)
            self.b = arr[f"{prefix}b"].astype(np.float64)
            self.hidden_dim = self.h_w.shape[1]
        else:
            raise ValueError(f"Unsupported core_type: {self.core_type}")

    def encode_sequence(self, seq_24x24: np.ndarray) -> np.ndarray:
        x_norm = (seq_24x24.astype(np.float64) - self.feat_mean[None, :]) / self.feat_std[None, :]
        seq_len = x_norm.shape[0]
        if self.core_type in ("bio", "rewired"):
            z = np.zeros(self.n_nodes, dtype=np.float64)
            drives = x_norm @ self.enc_w.T + self.enc_b
            w_eff_t = self.w_eff.T
            for t in range(seq_len):
                pre = drives[t] + z @ w_eff_t
                z = z + 0.2 * (np.tanh(pre) - z)
            return z

        d = self.hidden_dim
        wx_r, wx_z, wx_n = self.x_w[:d], self.x_w[d : 2 * d], self.x_w[2 * d :]
        wh_r, wh_z, wh_n = self.h_w[:d], self.h_w[d : 2 * d], self.h_w[2 * d :]
        br, bz, bn = self.b[:d], self.b[d : 2 * d], self.b[2 * d :]
        h = np.zeros(d, dtype=np.float64)
        for t in range(seq_len):
            xt = x_norm[t]
            r = _sigmoid(xt @ wx_r.T + h @ wh_r.T + br)
            z = _sigmoid(xt @ wx_z.T + h @ wh_z.T + bz)
            n_t = np.tanh(xt @ wx_n.T + r * (h @ wh_n.T) + bn)
            h = (1.0 - z) * h + z * n_t
        return h

    def predict_actions(self, seq_24x24: np.ndarray) -> Dict[str, dict]:
        z_end = self.encode_sequence(seq_24x24)
        z_rep = np.broadcast_to(z_end[None, :], (len(ACTIONS), z_end.shape[0]))
        h_in = np.concatenate([z_rep, self.act_emb], axis=1)
        h0 = np.maximum(0.0, h_in @ self.h0_w.T + self.h0_b)
        out = h0 @ self.h2_w.T + self.h2_b  # [5, 8]
        q = _sigmoid(out[:, :5])
        r_cum = 1.0 - np.cumprod(1.0 - q, axis=1)
        r_prot = _sigmoid(out[:, 5])
        preds: Dict[str, dict] = {}
        for idx, act in enumerate(ACTIONS):
            preds[act] = {
                "q_hazards": q[idx].tolist(),
                "r_cum": r_cum[idx].tolist(),
                "r_prot": float(r_prot[idx]),
                "progress": float(out[idx, 6]),
                "delta_u": float(out[idx, 7]),
                "raw_out": out[idx].tolist(),
            }
        return preds


def build_p6_arbitration_inputs(
    raw_preds: Dict[str, dict],
    variant: str,
    use_recovery: bool = True,
) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """Map V3 multi-horizon outputs into immediate safety + strategic utility inputs."""
    m4_preds: Dict[str, dict] = {}
    m8_preds: Dict[str, dict] = {}
    for act, raw in raw_preds.items():
        if variant == "single_24s":
            # Legacy 2.4s-only formulation tests horizon mismatch against 0.55 safety limit
            r24 = float(raw["q_hazards"][4])
            m4_preds[act] = {"contact_prob": r24}
            m8_preds[act] = {
                "contact_prob": r24,
                "progress": 0.0,
                "delta_u": 0.0,
                "recover_prob": 0.5,
            }
        else:
            # Multi-horizon protected separation:
            # - Hard safety limit arbitrates 0.30s protected contact vs 0.30s continue hazard
            # - Long-horizon 2.4s risk provides a soft strategic penalty on forward motion
            r_prot = float(raw["r_prot"])
            r_03 = float(raw["r_cum"][1])
            r_24 = float(raw["r_cum"][4])
            prog_eff = float(raw["progress"]) - (0.35 * r_24 if act != "brake" else 0.0)
            du_val = float(raw["delta_u"]) if use_recovery else 0.0
            rec_prob = float(0.5 + 0.5 * np.tanh(du_val)) if use_recovery else 0.5
            m4_preds[act] = {"contact_prob": r_prot}
            m8_preds[act] = {
                "contact_prob": r_03,
                "progress": prog_eff,
                "delta_u": du_val,
                "recover_prob": rec_prob,
            }
    return m4_preds, m8_preds


P6_CONFIGS = (
    "P5_A_V1_Frozen",
    "P5_C_Visual_Only",
    "P5_G_OldWeights_UpdatedFeas",
    "P6_C_Plus_IPMFeas",
    "P6_Bio_Single24s",
    "P6_Bio_MultiProt_NoRecov",
    "P6_Full_V3_Bio",
    "P6_Full_V3_Rewired7",
    "P6_Full_V3_GRU",
)


def _v3_policy_spec(config_name: str) -> Tuple[Optional[str], Optional[str], bool]:
    """Return (run_name, variant, use_recovery) for V3 configs."""
    mapping = {
        "P6_C_Plus_IPMFeas": (None, None, False),
        "P6_Bio_Single24s": ("Single24s_Bio", "single_24s", False),
        "P6_Bio_MultiProt_NoRecov": ("MultiHorizonProt_Bio", "multi_horizon_prot", False),
        "P6_Full_V3_Bio": ("FullV3_NeuroCOVER_Bio", "full_v3_neurocover", True),
        "P6_Full_V3_Rewired7": ("FullV3_NeuroCOVER_Rewired7", "full_v3_neurocover", True),
        "P6_Full_V3_GRU": ("FullV3_NeuroCOVER_GRU", "full_v3_neurocover", True),
    }
    return mapping[config_name]


def run_p6_episode(
    scenario: str,
    config_name: str,
    seed: int,
    escape_assets: dict,
    compass_model: dict,
    env_mode: str,
    v3_weights_path: Path,
    old_weights_path: Optional[Path] = None,
    cached_policy: Optional[_PortableV3Policy] = None,
) -> dict:
    """Execute one deterministic closed-loop P6 evaluation episode."""
    if config_name == "P5_A_V1_Frozen":
        rec = run_causal_episode(scenario, "A_V1_Frozen", seed, escape_assets, compass_model, env_mode, old_weights_path)
        rec["config"] = config_name
        return rec
    if config_name == "P5_C_Visual_Only":
        rec = run_causal_episode(scenario, "C_Visual_Fusion_Only", seed, escape_assets, compass_model, env_mode, None)
        rec["config"] = config_name
        return rec
    if config_name == "P5_G_OldWeights_UpdatedFeas":
        rec = run_causal_episode(scenario, "G_Full_P5_Architecture", seed, escape_assets, compass_model, env_mode, old_weights_path)
        rec["config"] = config_name
        return rec

    run_name, variant, use_recovery = _v3_policy_spec(config_name)
    policy = cached_policy
    if policy is None and run_name is not None:
        policy = _PortableV3Policy(v3_weights_path, run_name, variant)

    belief_v1 = SectorBelief(max_age_s=0.45)
    belief_v2 = FactorizedSpatialBelief(max_age_s=0.60)
    vis_v2 = DualPathwayT4T5Vision()
    integrity_mon = ObservationIntegrityMonitor(max_freeze_probes=2)
    checker = ActionFeasibilityChecker()
    cfg = ArbitrationConfig()
    compass = ConnectomeCompass(compass_model)
    _ = Network(escape_assets["npz"], escape_assets["labels"])

    lstate = None
    hist = []
    last_cap = None
    prev_t = 0.0
    anchor = None
    prev_action = "brake"
    actions_seq = []
    abstained = False
    whisker_events = 0
    prev_whisker = False
    outage_steps = 0
    u_start = None
    u_end = 0.0
    vis_out_latest = {"raw_lplc2": 0.0, "residual_lplc2": 0.0, "residual_lc4": 0.0, "lateral_imbalance": 0.0}

    with build_scenario(scenario, seed, env_mode=env_mode) as sim:
        obs = sim.observe()
        for step in range(80):
            if not ORACLE_KEYS.isdisjoint(obs):
                raise AssertionError("Oracle field leaked into observation")
            dt = max(0.05, obs["time_s"] - prev_t)
            prev_t = obs["time_s"]

            rep = integrity_mon.step(obs, imu_enabled=True)
            snap_v1 = belief_v1.update(obs)
            w_any = bool(obs["whisker_left"] or obs["whisker_right"])
            if w_any and not prev_whisker:
                whisker_events += 1
            prev_whisker = w_any
            if not rep.camera_healthy:
                outage_steps += 1

            dyaw_compass = rep.delta_yaw_rad if rep.delta_yaw_rad is not None else 0.0
            if obs["capture_s"] != last_cap:
                dt_bio = 0.1 if last_cap is None else max(0.05, obs["capture_s"] - last_cap)
                b_ang = bearing(obs["rgb"])
                if anchor is None and b_ang is not None:
                    anchor = b_ang
                cue = None if (b_ang is None or anchor is None) else (b_ang - anchor)
                _, comp_conc = compass.step(dyaw_compass, cue)
                vis_out_latest = vis_v2.step(obs["rgb"], dt_bio, rep.imu_gyro_z, rep.imu_gyro_y)
                last_cap = obs["capture_s"]
            else:
                comp_conc = belief_v2.compass_conc

            snap_v2 = belief_v2.update(
                obs,
                delta_yaw_rad=rep.delta_yaw_rad,
                heading_unc_delta_rad=rep.heading_uncertainty_delta_rad,
                compass_conc=comp_conc,
            )
            if u_start is None:
                u_start = snap_v2["task_uncertainty"]
            u_end = snap_v2["task_uncertainty"]

            ipm_out = estimate_ground_contact_clearance(obs["rgb"], pitch_rad=0.0)
            feat_v3, lstate = extract_features_v3(obs, snap_v1, snap_v2, vis_out_latest, ipm_out, lstate, dt)
            hist.append(feat_v3)

            m4_preds = m8_preds = None
            if policy is not None and step >= 8:
                seq = np.zeros((24, 24), dtype=np.float32)
                recent = hist[-24:]
                seq[-len(recent) :] = np.stack(recent)
                raw_preds = policy.predict_actions(seq)
                m4_preds, m8_preds = build_p6_arbitration_inputs(raw_preds, variant, use_recovery=use_recovery)

            action, rsn = select_protected_action_p5(
                obs=obs,
                belief_snap=snap_v2,
                integrity=rep,
                vis_out=vis_out_latest,
                speed_m_s=sim.evaluator()["speed_m_s"],
                m4_preds=m4_preds,
                m8_preds=m8_preds,
                prev_action=prev_action,
                feasibility_checker=checker,
                cfg=cfg,
                use_visual_fusion=True,
                use_risk_arbitration=True,
                use_feasibility_check=True,
                use_revised_utility=True,
            )
            if "abstain" in rsn:
                abstained = True

            integrity_mon.record_action(action, outage_active=(not rep.camera_healthy))
            prev_action = action
            actions_seq.append(action)
            obs = sim.step(action)

        ev = sim.evaluator()

    rec = classify_episode(
        scenario=scenario,
        x_m=ev["x_m"],
        y_m=ev["y_m"],
        body_collisions=int(ev["body_contact_events"]),
        whisker_events=whisker_events,
        abstained=abstained,
        outage_steps=outage_steps,
        actions_seq=actions_seq,
        u_task_start=float(u_start or 0.0),
        u_task_end=float(u_end),
    )
    rec["config"] = config_name
    rec["seed"] = seed
    rec["suite"] = "r05" if scenario in R05_SCENARIOS else "g08"
    return rec


def run_p6_pilot(
    out_dir: Path,
    seeds: int = 4,
    v3_weights_path: Path = Path("runs/pilot-v3-models/pilot_weights_v3.npz"),
    old_weights_path: Optional[Path] = Path("runs/kaggle-nb76b77cb2ff-output/portable_weights.npz"),
) -> dict:
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

    g08_scenarios = ("open_healthy", "looming_post", "near_miss_post", "freeze_outage_clear", "blocked_outage", "occluded_hazard")
    old_w = old_weights_path if (old_weights_path is not None and old_weights_path.exists()) else None

    cached_policies: Dict[str, _PortableV3Policy] = {}
    for cfg_name in P6_CONFIGS:
        if cfg_name.startswith("P6_") and cfg_name != "P6_C_Plus_IPMFeas":
            run_name, variant, _ = _v3_policy_spec(cfg_name)
            cached_policies[cfg_name] = _PortableV3Policy(v3_weights_path, run_name, variant)

    all_rows = []
    report = {
        "seeds_per_scenario": seeds,
        "episodes_per_suite": seeds * 6,
        "v3_weights_path": str(v3_weights_path),
        "g08": {},
        "r05": {},
        "scenario_breakdown": {},
    }

    for cfg_name in P6_CONFIGS:
        g08_rows, r05_rows = [], []
        policy = cached_policies.get(cfg_name)
        for sc in g08_scenarios:
            for s in range(seeds):
                g08_rows.append(
                    run_p6_episode(sc, cfg_name, s, escape_assets, compass_model, "rover_sim", v3_weights_path, old_w, policy)
                )
        for sc in R05_SCENARIOS:
            for s in range(seeds):
                r05_rows.append(
                    run_p6_episode(sc, cfg_name, s, escape_assets, compass_model, "real_world", v3_weights_path, old_w, policy)
                )
        report["g08"][cfg_name] = aggregate_episodes(g08_rows)
        report["r05"][cfg_name] = aggregate_episodes(r05_rows)

        sc_map = {}
        for row in g08_rows + r05_rows:
            sc_map.setdefault(row["scenario"], []).append(row)
        report["scenario_breakdown"][cfg_name] = {
            sc: {
                "n": len(rows),
                "collision_episodes": sum(1 for r in rows if r["body_collisions"] > 0),
                "total_body_collision_events": sum(int(r["body_collisions"]) for r in rows),
                "progress_mean": float(np.mean([r["x_m"] for r in rows])),
                "goal_reached": sum(1 for r in rows if r["collision_free_goal_completion"]),
                "permanent_stalls": sum(1 for r in rows if r["permanent_stall"]),
                "appropriate_abstentions": sum(1 for r in rows if r["appropriate_abstention"]),
                "unnecessary_abstentions": sum(1 for r in rows if r["unnecessary_abstention"]),
                "timeouts": sum(1 for r in rows if r["collision_free_timeout"]),
            }
            for sc, rows in sc_map.items()
        }
        all_rows.extend(g08_rows + r05_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "episodes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    (out_dir / "pilot_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("runs/p6-pilot-closed-loop"))
    p.add_argument("--weights", type=Path, default=Path("runs/pilot-v3-models/pilot_weights_v3.npz"))
    p.add_argument("--old-weights", type=Path, default=Path("runs/kaggle-nb76b77cb2ff-output/portable_weights.npz"))
    p.add_argument("--seeds", type=int, default=4)
    args = p.parse_args()

    rep = run_p6_pilot(args.out, seeds=args.seeds, v3_weights_path=args.weights, old_weights_path=args.old_weights)
    for suite in ("g08", "r05"):
        print(f"\n=== P6.8 CLOSED-LOOP PILOT ({suite.upper()}, N={rep['episodes_per_suite']} eps/policy) ===")
        for cfg_name, m in rep[suite].items():
            ci_l, ci_u = m["collision_rate_clopper_pearson_95ci"]
            print(
                f"{cfg_name:28s} | col_eps={m['collision_episodes']:2d}/{m['episodes']} "
                f"CI=[{ci_l:.3f},{ci_u:.3f}] | col_ev={m['total_body_collision_events']:2d} "
                f"| goal_cf={m['collision_free_goal_completion_rate']:.3f} "
                f"| prog={m['mean_progress_m']:.3f}m | sw={m['mean_action_switches']:.1f} "
                f"| stall={m['permanent_stall_rate']:.2f}"
            )


if __name__ == "__main__":
    main()
