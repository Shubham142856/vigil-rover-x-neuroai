"""P6.9.6 & P6.9.7 Matched Multi-Seed Training, Offline Calibration & Closed-Loop Acceptance Evaluation.

Supports:
  - Both joint recurrent cores (Bio / Rewired / GRU) and modular adaptive-tau
    backbones with decoupled risk & utility readouts (_ModularAdaptiveBioModel, _ModularGRUModel).
  - Multi-seed training (seeds=(7, 17, 27)) with identical optimizer budgets.
  - Epistemic ensemble spread sigma_G(a) = std_seeds(delta_u(a)) feeding conservative
    information gain G_cons(a) = G_hat(a) - gamma * sigma_G(a) in P6.9.3 safety arbitration.
  - Clean separation of:
      (1) Multi-task training ablation (NoRecovTrain vs Full_Bio_Modular)
      (2) Inference-time recovery ablation (Bio_Mod_NoRecovInf vs Full_Bio_Modular)
      (3) Short vs Multi-horizon vs Protected risk (Single24s vs Full_Bio)
      (4) Architecture comparison (Bio vs Rewired7/19/41 vs GRU Modular)
  - Closed-loop evaluation on Development suites (G08, R05) AND an Untouched Held-Out suite
    (chicane_slalom, blind_alcove_emergence, dynamic_crosswalk, ramp_chicanery).
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import load_npz

from blindspot_dataset import ORACLE_KEYS, SectorBelief
from camera_compass import bearing
from camera_escape import Network
from closed_loop_benchmark import R05_SCENARIOS, build_scenario
from compass_benchmark import ConnectomeCompass, load_compass_connectome
from evaluation_metrics_v2 import (
    aggregate_episodes,
    classify_episode,
    clopper_pearson_one_sided_upper,
)
from ground_range_estimator import estimate_ground_contact_clearance
from p69_bio_diagnostics import _ModularAdaptiveBioModel, _ModularGRUModel, _train_model
from p69_dataset_audit import binary_metrics
from p69_targeted_dataset import UNTOUCHED_WORLD_FAMILIES, build_p69_world
from p6_closed_loop_pilot import build_p6_arbitration_inputs
from pilot_dataset_v3 import ACTIONS, HORIZONS_S, extract_features_v3
from pilot_train_v3 import (
    _BioRecurrentModel,
    _GRURecurrentModel,
    _compute_cumulative_risks,
    _load_nt_signs,
    _sigmoid,
    _softplus,
)
from safety_arbitrator_v2 import (
    ActionFeasibilityChecker,
    ArbitrationConfig,
    BoundedExplorationMonitor,
    ObservationIntegrityMonitor,
    select_protected_action_p5,
)
from spatial_belief_v2 import FactorizedSpatialBelief
from t4_t5_vision import DualPathwayT4T5Vision


class _PortableP69Policy:
    """Vectorized NumPy evaluator supporting joint & modular adaptive-tau P6.9 checkpoints."""

    def __init__(self, arr: dict, key_prefix: str, variant: str):
        self.key_prefix = key_prefix
        self.variant = variant
        self.feat_mean = arr["feat_mean"].astype(np.float64)
        self.feat_std = np.maximum(arr["feat_std"].astype(np.float64), 1e-4)
        p = f"{key_prefix}__"
        self.core_type = str(arr[f"{p}core_type"].item())
        self.act_emb = arr[f"{p}act_emb"].astype(np.float64)

        if self.core_type in ("bio", "rewired", "bio_modular", "bio_modular_frozen"):
            adj = arr[f"{p}adj"].astype(np.float64)
            sign = arr[f"{p}sign"].astype(np.float64)
            theta = arr[f"{p}theta"].astype(np.float64)
            self.w_eff = adj * sign * _softplus(theta)
            self.enc_w = arr[f"{p}enc_w"].astype(np.float64)
            self.enc_b = arr[f"{p}enc_b"].astype(np.float64)
            self.n_nodes = self.w_eff.shape[0]
            if f"{p}tau_logit" in arr:
                self.alpha = 0.05 + 0.85 * _sigmoid(arr[f"{p}tau_logit"].astype(np.float64))
            else:
                self.alpha = np.full(self.n_nodes, 0.20, dtype=np.float64)
        elif self.core_type in ("gru", "gru_modular"):
            self.x_w = arr[f"{p}x_w"].astype(np.float64)
            self.h_w = arr[f"{p}h_w"].astype(np.float64)
            self.b = arr[f"{p}b"].astype(np.float64)
            self.hidden_dim = self.h_w.shape[1]
        else:
            raise ValueError(f"Unsupported core_type {self.core_type}")

        self.is_modular = self.core_type in ("bio_modular", "bio_modular_frozen", "gru_modular")
        if self.is_modular:
            self.belief_skip_dim = int(arr[f"{p}belief_skip_dim"].item())
            self.hr0_w = arr[f"{p}hr0_w"].astype(np.float64)
            self.hr0_b = arr[f"{p}hr0_b"].astype(np.float64)
            self.hr2_w = arr[f"{p}hr2_w"].astype(np.float64)
            self.hr2_b = arr[f"{p}hr2_b"].astype(np.float64)
            self.hu0_w = arr[f"{p}hu0_w"].astype(np.float64)
            self.hu0_b = arr[f"{p}hu0_b"].astype(np.float64)
            self.hu2_w = arr[f"{p}hu2_w"].astype(np.float64)
            self.hu2_b = arr[f"{p}hu2_b"].astype(np.float64)
        else:
            self.h0_w = arr[f"{p}h0_w"].astype(np.float64)
            self.h0_b = arr[f"{p}h0_b"].astype(np.float64)
            self.h2_w = arr[f"{p}h2_w"].astype(np.float64)
            self.h2_b = arr[f"{p}h2_b"].astype(np.float64)

    def encode_norm(self, x_norm: np.ndarray) -> np.ndarray:
        seq_len = x_norm.shape[0]
        if self.core_type in ("bio", "rewired", "bio_modular", "bio_modular_frozen"):
            z = np.zeros(self.n_nodes, dtype=np.float64)
            drives = x_norm @ self.enc_w.T + self.enc_b
            w_eff_t = self.w_eff.T
            alpha = self.alpha
            one_m = 1.0 - alpha
            for t in range(seq_len):
                z = one_m * z + alpha * np.tanh(drives[t] + z @ w_eff_t)
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
        x_norm = (seq_24x24.astype(np.float64) - self.feat_mean[None, :]) / self.feat_std[None, :]
        z_end = self.encode_norm(x_norm)
        z_rep = np.broadcast_to(z_end[None, :], (len(ACTIONS), z_end.shape[0]))
        if self.is_modular:
            b_last = np.broadcast_to(x_norm[-1, : self.belief_skip_dim][None, :], (len(ACTIONS), self.belief_skip_dim))
            ro_in = np.concatenate([z_rep, b_last, self.act_emb], axis=1)
            hr0 = np.maximum(0.0, ro_in @ self.hr0_w.T + self.hr0_b)
            out_r = hr0 @ self.hr2_w.T + self.hr2_b
            hu0 = np.maximum(0.0, ro_in @ self.hu0_w.T + self.hu0_b)
            out_u = hu0 @ self.hu2_w.T + self.hu2_b
            out = np.concatenate([out_r, out_u], axis=1)
        else:
            h_in = np.concatenate([z_rep, self.act_emb], axis=1)
            h0 = np.maximum(0.0, h_in @ self.h0_w.T + self.h0_b)
            out = h0 @ self.h2_w.T + self.h2_b

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


class _EnsembleP69Policy:
    """Multi-seed ensemble delivering mean predictions AND epistemic std sigma_G(a) for conservative info gain."""

    def __init__(self, policies: List[_PortableP69Policy]):
        self.policies = policies
        self.variant = policies[0].variant

    def predict_actions(self, seq_24x24: np.ndarray) -> Dict[str, dict]:
        all_p = [pol.predict_actions(seq_24x24) for pol in self.policies]
        out: Dict[str, dict] = {}
        for act in ACTIONS:
            qs = np.mean([p[act]["q_hazards"] for p in all_p], axis=0)
            rc = np.mean([p[act]["r_cum"] for p in all_p], axis=0)
            rp = float(np.mean([p[act]["r_prot"] for p in all_p]))
            pr = float(np.mean([p[act]["progress"] for p in all_p]))
            dus = [p[act]["delta_u"] for p in all_p]
            out[act] = {
                "q_hazards": qs.tolist(),
                "r_cum": rc.tolist(),
                "r_prot": rp,
                "progress": pr,
                "delta_u": float(np.mean(dus)),
                "delta_u_std": float(np.std(dus)) if len(dus) > 1 else 0.0,
            }
        return out


def _build_p69_model(arch: str, adj_map: dict, sgn_map: dict, in_dim: int, seed: int):
    if arch == "Single24s_Bio":
        return _BioRecurrentModel(adj_map["bio"], sgn_map["bio"], in_dim, seed=seed)
    if arch == "Full_Bio_Joint":
        return _BioRecurrentModel(adj_map["bio"], sgn_map["bio"], in_dim, seed=seed)
    if arch in ("MultiProt_NoRecovTrain_Bio", "Full_Bio_Modular"):
        return _ModularAdaptiveBioModel(adj_map["bio"], sgn_map["bio"], in_dim, seed=seed, freeze_core=False)
    if arch == "Full_Rewired7_Modular":
        return _ModularAdaptiveBioModel(adj_map["rew7"], sgn_map["rew7"], in_dim, seed=seed, freeze_core=False)
    if arch == "Full_Rewired19_Modular":
        return _ModularAdaptiveBioModel(adj_map["rew19"], sgn_map["rew19"], in_dim, seed=seed, freeze_core=False)
    if arch == "Full_Rewired41_Modular":
        return _ModularAdaptiveBioModel(adj_map["rew41"], sgn_map["rew41"], in_dim, seed=seed, freeze_core=False)
    if arch == "Full_GRU_Modular":
        return _ModularGRUModel(in_dim, hidden_dim=64, seed=seed)
    raise ValueError(arch)


def _train_one_job(task: Tuple[str, str, int, str, str, str, int]) -> dict:
    arch, variant, s, data_dir_s, bio_dir_s, ann_csv_s, epochs = task
    data_dir, bio_dir, ann_csv = Path(data_dir_s), Path(bio_dir_s), Path(ann_csv_s)
    pre_signs = _load_nt_signs(ann_csv, bio_dir / "nodes.txt")

    def _load(sp: str):
        with np.load(data_dir / f"{sp}.npz") as d:
            return {
                "X": d["X"].astype(np.float64),
                "action": d["action"].astype(np.int64),
                "y_hazard_bin": d["y_hazard_bin"].astype(np.int64),
                "y_risk_horizons": d["y_risk_horizons"].astype(np.float64),
                "y_risk_protected": d["y_risk_protected"].astype(np.float64),
                "y_progress_protected": d["y_progress_protected"].astype(np.float64),
                "y_signed_delta_u": d["y_signed_delta_u"].astype(np.float64),
            }

    tr, va, te = _load("train"), _load("val"), _load("test")
    mu = tr["X"].mean(axis=(0, 1), keepdims=True)
    sd = np.maximum(tr["X"].std(axis=(0, 1), keepdims=True), 1e-4)
    for d in (tr, va, te):
        d["X_norm"] = (d["X"] - mu) / sd
    in_dim = tr["X"].shape[-1]

    def _masks(p: Path):
        mat = load_npz(p).toarray()
        return (mat != 0).astype(np.float64), np.broadcast_to(pre_signs[None, :], mat.shape).copy()

    adj_map = {
        "bio": _masks(bio_dir / "biological.npz")[0],
        "rew7": _masks(bio_dir / "rewired-7.npz")[0],
        "rew19": _masks(bio_dir / "rewired-19.npz")[0],
        "rew41": _masks(bio_dir / "rewired-41.npz")[0],
    }
    sgn_val = np.broadcast_to(pre_signs[None, :], adj_map["bio"].shape).copy()
    sgn_map = {k: sgn_val for k in adj_map}

    mod = _build_p69_model(arch, adj_map, sgn_map, in_dim, s)
    _train_model(mod, tr, variant=variant, epochs=epochs, seed=s, batch_size=64)

    key_p = f"{arch}__s{s}"
    weights = {f"{key_p}__core_type": np.array([mod.core_type])}
    if hasattr(mod, "adj"):
        weights[f"{key_p}__adj"] = mod.adj.astype(np.float32)
        weights[f"{key_p}__sign"] = mod.sign.astype(np.float32)
    if hasattr(mod, "belief_skip_dim"):
        weights[f"{key_p}__belief_skip_dim"] = np.array([mod.belief_skip_dim], dtype=np.int64)
    for pk, pv in mod.params.items():
        weights[f"{key_p}__{pk}"] = pv.astype(np.float32)

    p_const_24 = float(np.mean(tr["y_risk_horizons"][:, 4] > 0.5))
    p_const_prot = float(np.mean(tr["y_risk_protected"] > 0.5))
    p_const_03 = float(np.mean(tr["y_risk_horizons"][:, 1] > 0.5))

    metrics_sp = {}
    for sp_name, ds in (("val", va), ("test", te)):
        _, out_sp, _ = mod.forward_and_grads(ds["X_norm"], ds["action"], ds, variant)
        if variant == "single_24s":
            r24 = _sigmoid(out_sp[:, 4])
            metrics_sp[sp_name] = {
                "risk_24s": binary_metrics(ds["y_risk_horizons"][:, 4], r24, p_const_24),
                "cross_protected": binary_metrics(ds["y_risk_protected"], r24, p_const_prot),
            }
        else:
            rc = _compute_cumulative_risks(out_sp)
            pr = _sigmoid(out_sp[:, 5])
            metrics_sp[sp_name] = {
                "protected_0_30s": binary_metrics(ds["y_risk_protected"], pr, p_const_prot),
                "cum_0_30s": binary_metrics(ds["y_risk_horizons"][:, 1], rc[:, 1], p_const_03),
                "cum_2_40s": binary_metrics(ds["y_risk_horizons"][:, 4], rc[:, 4], p_const_24),
                "progress_mae": float(np.mean(np.abs(out_sp[:, 6] - ds["y_progress_protected"]))),
                "delta_u_mae": float(np.mean(np.abs(out_sp[:, 7] - ds["y_signed_delta_u"]))),
                "survival_monotonic": bool(np.all(np.diff(rc, axis=1) >= -1e-9)),
            }
    return {"arch": arch, "variant": variant, "seed": s, "weights": weights, "metrics": metrics_sp}


def train_p69_matched_suite(
    data_dir: Path,
    bio_dir: Path,
    ann_csv: Path,
    out_dir: Path,
    epochs: int = 80,
    seeds: Tuple[int, ...] = (7, 17, 27),
    workers: int = 4,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(data_dir / "train.npz") as d:
        X_tr = d["X"].astype(np.float64)
    mu = X_tr.mean(axis=(0, 1), keepdims=True)
    sd = np.maximum(X_tr.std(axis=(0, 1), keepdims=True), 1e-4)

    specs = [
        ("Single24s_Bio", "single_24s"),
        ("MultiProt_NoRecovTrain_Bio", "multi_horizon_prot"),
        ("Full_Bio_Joint", "full_v3_neurocover"),
        ("Full_Bio_Modular", "full_v3_neurocover"),
        ("Full_Rewired7_Modular", "full_v3_neurocover"),
        ("Full_Rewired19_Modular", "full_v3_neurocover"),
        ("Full_Rewired41_Modular", "full_v3_neurocover"),
        ("Full_GRU_Modular", "full_v3_neurocover"),
    ]
    tasks = [(arch, variant, s, str(data_dir), str(bio_dir), str(ann_csv), epochs) for arch, variant in specs for s in seeds]
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_train_one_job, tasks))
    else:
        results = [_train_one_job(t) for t in tasks]

    exported = {"feat_mean": mu.squeeze().astype(np.float32), "feat_std": sd.squeeze().astype(np.float32)}
    grouped: Dict[str, List[dict]] = {arch: [] for arch, _ in specs}
    for res in results:
        exported.update(res["weights"])
        grouped[res["arch"]].append(res)

    report = {"epochs": epochs, "seeds": list(seeds), "models": {}}
    for arch, variant in specs:
        items = sorted(grouped[arch], key=lambda x: x["seed"])
        seed_metrics = {"val": [it["metrics"]["val"] for it in items], "test": [it["metrics"]["test"] for it in items]}
        agg = {"variant": variant}
        for sp_name in ("val", "test"):
            rows = seed_metrics[sp_name]
            if variant == "single_24s":
                agg[sp_name] = {
                    "risk_24s_brier_mean": float(np.mean([r["risk_24s"]["brier"] for r in rows])),
                    "risk_24s_bss_mean": float(np.mean([r["risk_24s"]["brier_skill_score"] for r in rows])),
                    "risk_24s_pr_auc_mean": float(np.mean([r["risk_24s"]["pr_auc"] for r in rows if r["risk_24s"]["pr_auc"] is not None])),
                    "cross_prot_bss_mean": float(np.mean([r["cross_protected"]["brier_skill_score"] for r in rows])),
                }
            else:
                agg[sp_name] = {
                    "prot_brier_mean": float(np.mean([r["protected_0_30s"]["brier"] for r in rows])),
                    "prot_bss_mean": float(np.mean([r["protected_0_30s"]["brier_skill_score"] for r in rows])),
                    "prot_pr_auc_mean": float(np.mean([r["protected_0_30s"]["pr_auc"] for r in rows if r["protected_0_30s"]["pr_auc"] is not None])),
                    "prot_recall_050_mean": float(np.mean([r["protected_0_30s"]["recall_at_0_50"] for r in rows if r["protected_0_30s"]["recall_at_0_50"] is not None])),
                    "prot_fn_mean": float(np.mean([r["protected_0_30s"]["dangerous_fn_lt_0_20"] for r in rows])),
                    "cum24_brier_mean": float(np.mean([r["cum_2_40s"]["brier"] for r in rows])),
                    "cum24_bss_mean": float(np.mean([r["cum_2_40s"]["brier_skill_score"] for r in rows])),
                    "cum24_pr_auc_mean": float(np.mean([r["cum_2_40s"]["pr_auc"] for r in rows if r["cum_2_40s"]["pr_auc"] is not None])),
                    "progress_mae_mean": float(np.mean([r["progress_mae"] for r in rows])),
                    "delta_u_mae_mean": float(np.mean([r["delta_u_mae"] for r in rows])),
                    "delta_u_mae_std": float(np.std([r["delta_u_mae"] for r in rows])),
                    "survival_monotonic_all": all(r["survival_monotonic"] for r in rows),
                }
        agg["per_seed"] = seed_metrics
        report["models"][arch] = agg

    np.savez_compressed(out_dir / "p69_weights.npz", **exported)
    (out_dir / "p69_offline_metrics.json").write_text(json.dumps(report, indent=2))
    return report


CLOSED_LOOP_POLICIES = (
    "P69_Det_IPMFeas",
    "P69_Single24s_Bio",
    "P69_Bio_NoRecovTrain",
    "P69_Bio_Mod_NoRecovInf",
    "P69_Full_Bio_Joint",
    "P69_Full_Bio_Modular",
    "P69_Full_Rewired7_Modular",
    "P69_Full_GRU_Modular",
)


def _load_ensemble_for_config(arr: dict, cfg_name: str, seeds: Tuple[int, ...]) -> Tuple[Optional[_EnsembleP69Policy], str, bool]:
    spec_map = {
        "P69_Det_IPMFeas": (None, "none", False),
        "P69_Single24s_Bio": ("Single24s_Bio", "single_24s", False),
        "P69_Bio_NoRecovTrain": ("MultiProt_NoRecovTrain_Bio", "multi_horizon_prot", False),
        "P69_Bio_Mod_NoRecovInf": ("Full_Bio_Modular", "full_v3_neurocover", False),
        "P69_Full_Bio_Joint": ("Full_Bio_Joint", "full_v3_neurocover", True),
        "P69_Full_Bio_Modular": ("Full_Bio_Modular", "full_v3_neurocover", True),
        "P69_Full_Rewired7_Modular": ("Full_Rewired7_Modular", "full_v3_neurocover", True),
        "P69_Full_GRU_Modular": ("Full_GRU_Modular", "full_v3_neurocover", True),
    }
    arch, variant, use_recov = spec_map[cfg_name]
    if arch is None:
        return None, variant, use_recov
    pols = [_PortableP69Policy(arr, f"{arch}__s{s}", variant) for s in seeds if f"{arch}__s{s}__core_type" in arr]
    return _EnsembleP69Policy(pols), variant, use_recov


def run_p69_episode(
    scenario: str,
    config_name: str,
    seed: int,
    escape_assets: dict,
    compass_model: dict,
    env_mode: str,
    ensemble: Optional[_EnsembleP69Policy],
    variant: str,
    use_recovery: bool,
    suite_label: str,
) -> dict:
    belief_v1 = SectorBelief(max_age_s=0.45)
    belief_v2 = FactorizedSpatialBelief(max_age_s=0.60)
    vis_v2 = DualPathwayT4T5Vision()
    integrity_mon = ObservationIntegrityMonitor(max_freeze_probes=2)
    checker = ActionFeasibilityChecker()
    exploration_mon = BoundedExplorationMonitor()
    cfg = ArbitrationConfig()
    compass = ConnectomeCompass(compass_model)
    _ = Network(escape_assets["npz"], escape_assets["labels"])

    lstate = None
    hist = []
    last_cap = None
    prev_t = 0.0
    anchor = None
    cue_latest = None
    prev_action = "brake"
    actions_seq = []
    abstained = False
    whisker_events = 0
    prev_whisker = False
    outage_steps = 0
    u_start = None
    u_end = 0.0
    vis_out_latest = {"raw_lplc2": 0.0, "residual_lplc2": 0.0, "residual_lc4": 0.0, "lateral_imbalance": 0.0}

    ctx = build_p69_world(scenario, seed) if scenario in UNTOUCHED_WORLD_FAMILIES else build_scenario(scenario, seed, env_mode=env_mode)
    with ctx as sim:
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

            eff_dyaw = (1.0 if prev_action == "left" else (-1.0 if prev_action == "right" else 0.0)) * dt
            dyaw_compass = rep.delta_yaw_rad if rep.delta_yaw_rad is not None else eff_dyaw
            if obs["capture_s"] != last_cap:
                dt_bio = 0.1 if last_cap is None else max(0.05, obs["capture_s"] - last_cap)
                vis_out_latest = vis_v2.step(obs["rgb"], dt_bio, rep.imu_gyro_z, rep.imu_gyro_y)
                if rep.camera_healthy:
                    b_ang = bearing(obs["rgb"])
                    if anchor is None and b_ang is not None and abs(vis_out_latest.get("residual_lc4", 0.0)) < 0.35:
                        anchor = b_ang
                    cue_latest = None if (b_ang is None or anchor is None or abs(vis_out_latest.get("residual_lc4", 0.0)) >= 0.35) else (b_ang - anchor)
                else:
                    cue_latest = None
                _, comp_conc = compass.step(dyaw_compass, cue_latest)
                last_cap = obs["capture_s"]
            else:
                comp_conc = belief_v2.compass_conc

            vis_motion_stable = bool(
                rep.camera_healthy
                and float(obs.get("contrast", 0.0)) >= 0.04
                and abs(float(vis_out_latest.get("residual_lc4", 0.0))) < 0.18
            )
            snap_v2 = belief_v2.update(
                obs,
                delta_yaw_rad=rep.delta_yaw_rad,
                heading_unc_delta_rad=rep.heading_uncertainty_delta_rad,
                compass_conc=comp_conc,
                action_cmd=prev_action,
                visual_yaw_rad=cue_latest,
                visual_yaw_kappa=10.0 if (rep.camera_healthy and cue_latest is not None) else 0.0,
                visual_motion_stable=vis_motion_stable,
            )
            if u_start is None:
                u_start = snap_v2["task_uncertainty"]
            u_end = snap_v2["task_uncertainty"]

            ipm_out = estimate_ground_contact_clearance(obs["rgb"], pitch_rad=0.0) if rep.camera_healthy else {
                "obstacle_detected": False, "estimated_clearance_m": 4.0, "ipm_reliable": False
            }
            feat_v3, lstate = extract_features_v3(obs, snap_v1, snap_v2, vis_out_latest, ipm_out, lstate, dt)
            hist.append(feat_v3)

            m4_preds = m8_preds = None
            if ensemble is not None and step >= 8:
                seq = np.zeros((24, 24), dtype=np.float32)
                recent = hist[-24:]
                seq[-len(recent) :] = np.stack(recent)
                raw_preds = ensemble.predict_actions(seq)
                m4_preds, m8_preds = build_p6_arbitration_inputs(raw_preds, variant, use_recovery=use_recovery)
                # Inject ensemble epistemic spread sigma_G(a) for conservative info gain
                if use_recovery:
                    for a_k in m8_preds:
                        m8_preds[a_k]["delta_u_std"] = float(raw_preds[a_k].get("delta_u_std", 0.0))

            speed_est = float(
                obs["encoder_speed_m_s"]
                if ("encoder_speed_m_s" in obs and obs["encoder_speed_m_s"] is not None)
                else (0.35 if prev_action == "forward" else (0.12 if prev_action == "slow" else 0.0))
            )
            action, rsn = select_protected_action_p5(
                obs=obs,
                belief_snap=snap_v2,
                integrity=rep,
                vis_out=vis_out_latest,
                speed_m_s=speed_est,
                m4_preds=m4_preds,
                m8_preds=m8_preds,
                prev_action=prev_action,
                feasibility_checker=checker,
                cfg=cfg,
                use_visual_fusion=True,
                use_risk_arbitration=True,
                use_feasibility_check=True,
                use_revised_utility=True,
                exploration_monitor=exploration_mon,
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
    rec["suite"] = suite_label
    return rec


def _eval_cfg_worker(task: Tuple[str, str, int, int, Tuple[int, ...]]) -> dict:
    cfg_name, weights_s, seeds_dev, seeds_untouched, train_seeds = task
    weights_npz = Path(weights_s)
    escape_dir = Path("runs/graph-controls/escape")
    ann_path = Path("runs/connectome/escape/annotations.csv")
    compass_dir = Path("runs/connectome/compass")
    with ann_path.open() as f:
        types = {r["root_id"]: r["primary_type"] for r in csv.DictReader(f)}
    ids = (escape_dir / "nodes.txt").read_text().splitlines()
    escape_assets = {"npz": escape_dir / "biological.npz", "labels": np.array([types[i] for i in ids]), "threshold": 0.0238}
    compass_model = load_compass_connectome(compass_dir)
    with np.load(weights_npz, allow_pickle=False) as d:
        arr = {k: d[k] for k in d.files}

    g08_scenarios = ("open_healthy", "looming_post", "near_miss_post", "freeze_outage_clear", "blocked_outage", "occluded_hazard")
    ens, variant, use_rec = _load_ensemble_for_config(arr, cfg_name, train_seeds)
    g08_r, r05_r, unt_r = [], [], []
    for sc in g08_scenarios:
        for s in range(seeds_dev):
            g08_r.append(run_p69_episode(sc, cfg_name, s, escape_assets, compass_model, "rover_sim", ens, variant, use_rec, "g08"))
    for sc in R05_SCENARIOS:
        for s in range(seeds_dev):
            r05_r.append(run_p69_episode(sc, cfg_name, s, escape_assets, compass_model, "real_world", ens, variant, use_rec, "r05"))
    for sc in UNTOUCHED_WORLD_FAMILIES:
        for s in range(seeds_untouched):
            unt_r.append(run_p69_episode(sc, cfg_name, 9000 + s, escape_assets, compass_model, "untouched", ens, variant, use_rec, "untouched"))
    return {"cfg_name": cfg_name, "g08": g08_r, "r05": r05_r, "untouched": unt_r}


def run_p69_closed_loop_gate(
    weights_npz: Path,
    out_dir: Path,
    seeds_dev: int = 4,
    seeds_untouched: int = 6,
    train_seeds: Tuple[int, ...] = (7, 17, 27),
    workers: int = 4,
) -> dict:
    tasks = [(cfg_name, str(weights_npz), seeds_dev, seeds_untouched, train_seeds) for cfg_name in CLOSED_LOOP_POLICIES]
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_eval_cfg_worker, tasks))
    else:
        results = [_eval_cfg_worker(t) for t in tasks]

    all_rows = []
    report = {
        "weights_path": str(weights_npz),
        "g08": {},
        "r05": {},
        "untouched": {},
        "scenario_breakdown": {},
    }

    for res in results:
        cfg_name = res["cfg_name"]
        g08_r, r05_r, unt_r = res["g08"], res["r05"], res["untouched"]
        for sname, rows in (("g08", g08_r), ("r05", r05_r), ("untouched", unt_r)):
            agg = aggregate_episodes(rows)
            agg["collision_rate_one_sided_95upper"] = clopper_pearson_one_sided_upper(agg["collision_episodes"], agg["episodes"])
            report[sname][cfg_name] = agg

        sc_map = {}
        for row in g08_r + r05_r + unt_r:
            sc_map.setdefault(row["scenario"], []).append(row)
        report["scenario_breakdown"][cfg_name] = {
            sc: {
                "n": len(rows),
                "collision_episodes": sum(1 for r in rows if r["body_collisions"] > 0),
                "progress_mean": float(np.mean([r["x_m"] for r in rows])),
                "goal_reached": sum(1 for r in rows if r["collision_free_goal_completion"]),
                "permanent_stalls": sum(1 for r in rows if r["permanent_stall"]),
            }
            for sc, rows in sc_map.items()
        }
        all_rows.extend(g08_r + r05_r + unt_r)

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "episodes_p69.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    (out_dir / "p69_closed_loop_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("runs/pilot-v3-p69"))
    p.add_argument("--bio-dir", type=Path, default=Path("runs/graph-controls/escape"))
    p.add_argument("--ann", type=Path, default=Path("runs/connectome/escape/annotations.csv"))
    p.add_argument("--models-out", type=Path, default=Path("runs/p69-models"))
    p.add_argument("--eval-out", type=Path, default=Path("runs/p69-closed-loop"))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    off = train_p69_matched_suite(args.data, args.bio_dir, args.ann, args.models_out, epochs=args.epochs, workers=args.workers)
    print("\n=== P6.9.6 OFFLINE MATCHED TRAINING (3 seeds, TEST UNTOUCHED/UN-MINED) ===")
    for name, m in off["models"].items():
        te = m["test"]
        if m["variant"] == "single_24s":
            print(f"{name:26s} | 2.4s_BSS={te['risk_24s_bss_mean']:+.3f} PRAUC={te['risk_24s_pr_auc_mean']:.3f} | xProt_BSS={te['cross_prot_bss_mean']:+.3f}")
        else:
            print(
                f"{name:26s} | prot_BSS={te['prot_bss_mean']:+.3f} PRAUC={te['prot_pr_auc_mean']:.3f} FN={te['prot_fn_mean']:.1f} "
                f"| 2.4s_BSS={te['cum24_bss_mean']:+.3f} | progMAE={te['progress_mae_mean']:.4f} duMAE={te['delta_u_mae_mean']:.4f}"
            )

    cl = run_p69_closed_loop_gate(args.models_out / "p69_weights.npz", args.eval_out, workers=args.workers)
    for suite in ("g08", "r05", "untouched"):
        print(f"\n=== P6.9.7 CLOSED-LOOP ({suite.upper()}) ===")
        for cfg_name, m in cl[suite].items():
            ci_l, ci_u = m["collision_rate_clopper_pearson_95ci"]
            print(
                f"{cfg_name:26s} | col={m['collision_episodes']:2d}/{m['episodes']} CI2=[{ci_l:.3f},{ci_u:.3f}] "
                f"| goal_cf={m['collision_free_goal_completion_rate']:.3f} | prog={m['mean_progress_m']:.3f}m "
                f"| stall={m['permanent_stall_rate']:.2f} | sw={m['mean_action_switches']:.1f}"
            )


if __name__ == "__main__":
    main()
