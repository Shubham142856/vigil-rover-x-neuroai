"""P6.7 Lightweight Multi-Horizon Pilot Model Trainer & Connectivity Comparison (Pure NumPy).

Trains and evaluates compact recurrent controllers directly on the audited P6.6 pilot
dataset (`runs/pilot-v3-p6/`) with exact analytical gradients (finite-difference verified)
and exports portable weights for P6.8 closed-loop evaluation.

Supported connectivity cores:
  - 'bio'     : FAFB v783 signed sparse connectome constraint W_eff = M * S * softplus(theta)
  - 'rewired' : Degree-preserving double-edge-swapped connectome control (rewired-7.npz)
  - 'gru'     : Dense gated recurrent unit baseline (64 hidden units)

Supported head variants (to compare cleanly on equal training budgets):
  1. single_24s            : 2.4s cumulative contact risk only
  2. short_03s             : 0.3s cumulative / protected risk only
  3. multi_horizon         : 5-horizon discrete survival hazards (0.1, 0.3, 0.6, 1.2, 2.4s)
  4. multi_horizon_prot    : survival horizons + protected-brake risk + progress
  5. full_v3_neurocover    : survival horizons + protected risk + progress + signed Delta_U_t
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.sparse import load_npz


def _sigmoid(x: np.ndarray) -> np.ndarray:
    xc = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-xc))


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.where(x > 20.0, x, np.log1p(np.exp(np.clip(x, -30.0, 20.0))))


def _load_nt_signs(annotations_csv: Path, nodes_txt: Path) -> np.ndarray:
    ids = nodes_txt.read_text().splitlines()
    with annotations_csv.open() as f:
        nt_map = {r["root_id"]: r["predicted_nt"] for r in csv.DictReader(f)}
    pre_signs = np.zeros(len(ids), dtype=np.float64)
    for idx, rid in enumerate(ids):
        nt = nt_map.get(rid, "")
        if nt == "ACH":
            pre_signs[idx] = 1.0
        elif nt == "GLUT":
            pre_signs[idx] = -1.0
    return pre_signs


class _BioRecurrentModel:
    """Connectome-constrained continuous-time rate RNN + action-conditioned multi-head MLP."""

    def __init__(self, adj_mask: np.ndarray, sign_mask: np.ndarray, in_dim: int, n_actions: int = 5, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.core_type = "bio"
        self.adj = adj_mask.astype(np.float64)
        self.sign = sign_mask.astype(np.float64)
        n = adj_mask.shape[0]
        self.n_nodes = n
        self.params: Dict[str, np.ndarray] = {
            "theta": np.full((n, n), -1.0, dtype=np.float64),
            "enc_w": rng.normal(0.0, 1.0 / np.sqrt(max(in_dim, 1)), (n, in_dim)),
            "enc_b": np.zeros(n, dtype=np.float64),
            "act_emb": rng.normal(0.0, 0.25, (n_actions, 16)),
            "h0_w": rng.normal(0.0, np.sqrt(2.0 / (n + 16)), (64, n + 16)),
            "h0_b": np.zeros(64, dtype=np.float64),
            "h2_w": rng.normal(0.0, np.sqrt(1.0 / 64.0), (8, 64)),
            "h2_b": np.zeros(8, dtype=np.float64),
        }

    def forward_and_grads(
        self,
        x_seq: np.ndarray,
        act: np.ndarray,
        targets: dict,
        variant: str,
        pos_wt_prot: float = 3.0,
    ) -> Tuple[float, np.ndarray, Dict[str, np.ndarray]]:
        bsz, seq_len, _ = x_seq.shape
        n = self.n_nodes
        p = self.params
        w_sp = _softplus(p["theta"])
        w_eff = self.adj * self.sign * w_sp
        sig_theta = _sigmoid(p["theta"])

        # Forward recurrence unroll storing states for exact BPTT
        states = np.zeros((bsz, seq_len + 1, n), dtype=np.float64)
        pre_tanh = np.zeros((bsz, seq_len, n), dtype=np.float64)
        for t in range(seq_len):
            drive = x_seq[:, t, :] @ p["enc_w"].T + p["enc_b"]
            pre = drive + states[:, t, :] @ w_eff.T
            pre_tanh[:, t, :] = pre
            states[:, t + 1, :] = states[:, t, :] + 0.2 * (np.tanh(pre) - states[:, t, :])

        z_end = states[:, -1, :]
        a_emb = p["act_emb"][act]
        h_in = np.concatenate([z_end, a_emb], axis=1)
        h0_pre = h_in @ p["h0_w"].T + p["h0_b"]
        h0 = np.maximum(0.0, h0_pre)
        out = h0 @ p["h2_w"].T + p["h2_b"]  # [B, 8]: 0..4 survival, 5 prot_risk, 6 prog, 7 delta_u

        # Loss + d_loss / d_out
        d_out = np.zeros_like(out)
        loss = 0.0

        haz_bin = targets["y_hazard_bin"]
        r_horizons = targets["y_risk_horizons"]
        y_prot = targets["y_risk_protected"]
        y_prog = targets["y_progress_protected"]
        y_du = targets["y_signed_delta_u"]

        q = _sigmoid(out[:, :5])
        if variant == "single_24s":
            # Supervise only 2.4s binary risk directly on out[:, 4]
            p24 = q[:, 4]
            y24 = r_horizons[:, 4]
            loss += float(-np.mean(y24 * np.log(p24 + 1e-7) + (1.0 - y24) * np.log(1.0 - p24 + 1e-7)))
            d_out[:, 4] = (p24 - y24) / bsz
        elif variant == "short_03s":
            # Supervise only 0.30s brief protected risk on out[:, 5]
            pr = _sigmoid(out[:, 5])
            w = np.where(y_prot > 0.5, pos_wt_prot, 1.0)
            loss += float(np.mean(-w * (y_prot * np.log(pr + 1e-7) + (1.0 - y_prot) * np.log(1.0 - pr + 1e-7))))
            d_out[:, 5] = w * (pr - y_prot) / bsz
        else:
            # Multi-horizon discrete-time survival negative log-likelihood
            j_idx = np.arange(5)[None, :]
            at_risk = (j_idx <= np.minimum(haz_bin[:, None], 4)).astype(np.float64)
            hit_here = (j_idx == haz_bin[:, None]).astype(np.float64)
            nll = -at_risk * (hit_here * np.log(q + 1e-7) + (1.0 - hit_here) * np.log(1.0 - q + 1e-7))
            loss += float(nll.sum() / bsz)
            d_out[:, :5] = at_risk * (q - hit_here) / bsz

            if variant in ("multi_horizon_prot", "full_v3_neurocover"):
                pr = _sigmoid(out[:, 5])
                w = np.where(y_prot > 0.5, pos_wt_prot, 1.0)
                loss += float(np.mean(-w * (y_prot * np.log(pr + 1e-7) + (1.0 - y_prot) * np.log(1.0 - pr + 1e-7))))
                d_out[:, 5] = w * (pr - y_prot) / bsz

                # Progress MSE (weight 0.5)
                err_p = out[:, 6] - y_prog
                loss += float(0.25 * np.mean(err_p * err_p))
                d_out[:, 6] = 0.5 * err_p / bsz

            if variant == "full_v3_neurocover":
                # Signed counterfactual uncertainty gain regression (Huber-like L2 weight 0.5)
                err_u = out[:, 7] - y_du
                loss += float(0.25 * np.mean(err_u * err_u))
                d_out[:, 7] = 0.5 * err_u / bsz

        # Backprop through MLP head
        grads: Dict[str, np.ndarray] = {}
        grads["h2_w"] = d_out.T @ h0
        grads["h2_b"] = d_out.sum(axis=0)
        d_h0 = (d_out @ p["h2_w"]) * (h0_pre > 0.0)
        grads["h0_w"] = d_h0.T @ h_in
        grads["h0_b"] = d_h0.sum(axis=0)
        d_hin = d_h0 @ p["h0_w"]

        d_act = d_hin[:, n:]
        g_act = np.zeros_like(p["act_emb"])
        np.add.at(g_act, act, d_act)
        grads["act_emb"] = g_act

        # Backprop through time (BPTT) across biological rate network
        d_z = d_hin[:, :n]
        g_enc_w = np.zeros_like(p["enc_w"])
        g_enc_b = np.zeros_like(p["enc_b"])
        g_weff = np.zeros_like(w_eff)

        for t in range(seq_len - 1, -1, -1):
            th = np.tanh(pre_tanh[:, t, :])
            d_pre = 0.2 * d_z * (1.0 - th * th)
            g_enc_w += d_pre.T @ x_seq[:, t, :]
            g_enc_b += d_pre.sum(axis=0)
            g_weff += d_pre.T @ states[:, t, :]
            d_z = 0.8 * d_z + d_pre @ w_eff

        grads["enc_w"] = g_enc_w
        grads["enc_b"] = g_enc_b
        grads["theta"] = g_weff * self.adj * self.sign * sig_theta
        return loss, out, grads


class _GRURecurrentModel:
    """Matched GRU baseline with identical 8-output MLP head."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, n_actions: int = 5, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.core_type = "gru"
        self.hidden_dim = hidden_dim
        self.params: Dict[str, np.ndarray] = {
            "x_w": rng.normal(0.0, 1.0 / np.sqrt(max(in_dim, 1)), (3 * hidden_dim, in_dim)),
            "h_w": rng.normal(0.0, 1.0 / np.sqrt(hidden_dim), (3 * hidden_dim, hidden_dim)),
            "b": np.zeros(3 * hidden_dim, dtype=np.float64),
            "act_emb": rng.normal(0.0, 0.25, (n_actions, 16)),
            "h0_w": rng.normal(0.0, np.sqrt(2.0 / (hidden_dim + 16)), (64, hidden_dim + 16)),
            "h0_b": np.zeros(64, dtype=np.float64),
            "h2_w": rng.normal(0.0, np.sqrt(1.0 / 64.0), (8, 64)),
            "h2_b": np.zeros(8, dtype=np.float64),
        }

    def forward_and_grads(self, x_seq, act, targets, variant, pos_wt_prot=3.0):
        bsz, seq_len, _ = x_seq.shape
        d = self.hidden_dim
        p = self.params
        hs = np.zeros((bsz, seq_len + 1, d), dtype=np.float64)
        rs = np.zeros((bsz, seq_len, d), dtype=np.float64)
        zs = np.zeros((bsz, seq_len, d), dtype=np.float64)
        ntilde = np.zeros((bsz, seq_len, d), dtype=np.float64)
        gh = np.zeros((bsz, seq_len, d), dtype=np.float64)

        wx_r, wx_z, wx_n = p["x_w"][:d], p["x_w"][d : 2 * d], p["x_w"][2 * d :]
        wh_r, wh_z, wh_n = p["h_w"][:d], p["h_w"][d : 2 * d], p["h_w"][2 * d :]
        br, bz, bn = p["b"][:d], p["b"][d : 2 * d], p["b"][2 * d :]

        for t in range(seq_len):
            xt = x_seq[:, t, :]
            ht = hs[:, t, :]
            r = _sigmoid(xt @ wx_r.T + ht @ wh_r.T + br)
            z = _sigmoid(xt @ wx_z.T + ht @ wh_z.T + bz)
            g_h = ht @ wh_n.T
            n_t = np.tanh(xt @ wx_n.T + r * g_h + bn)
            hs[:, t + 1, :] = (1.0 - z) * ht + z * n_t
            rs[:, t, :], zs[:, t, :], ntilde[:, t, :], gh[:, t, :] = r, z, n_t, g_h

        z_end = hs[:, -1, :]
        a_emb = p["act_emb"][act]
        h_in = np.concatenate([z_end, a_emb], axis=1)
        h0_pre = h_in @ p["h0_w"].T + p["h0_b"]
        h0 = np.maximum(0.0, h0_pre)
        out = h0 @ p["h2_w"].T + p["h2_b"]

        d_out = np.zeros_like(out)
        loss = 0.0
        haz_bin, r_horizons = targets["y_hazard_bin"], targets["y_risk_horizons"]
        y_prot, y_prog, y_du = targets["y_risk_protected"], targets["y_progress_protected"], targets["y_signed_delta_u"]

        q = _sigmoid(out[:, :5])
        if variant == "single_24s":
            p24, y24 = q[:, 4], r_horizons[:, 4]
            loss += float(-np.mean(y24 * np.log(p24 + 1e-7) + (1.0 - y24) * np.log(1.0 - p24 + 1e-7)))
            d_out[:, 4] = (p24 - y24) / bsz
        elif variant == "short_03s":
            pr = _sigmoid(out[:, 5])
            w = np.where(y_prot > 0.5, pos_wt_prot, 1.0)
            loss += float(np.mean(-w * (y_prot * np.log(pr + 1e-7) + (1.0 - y_prot) * np.log(1.0 - pr + 1e-7))))
            d_out[:, 5] = w * (pr - y_prot) / bsz
        else:
            j_idx = np.arange(5)[None, :]
            at_risk = (j_idx <= np.minimum(haz_bin[:, None], 4)).astype(np.float64)
            hit_here = (j_idx == haz_bin[:, None]).astype(np.float64)
            loss += float((-at_risk * (hit_here * np.log(q + 1e-7) + (1.0 - hit_here) * np.log(1.0 - q + 1e-7))).sum() / bsz)
            d_out[:, :5] = at_risk * (q - hit_here) / bsz
            if variant in ("multi_horizon_prot", "full_v3_neurocover"):
                pr = _sigmoid(out[:, 5])
                w = np.where(y_prot > 0.5, pos_wt_prot, 1.0)
                loss += float(np.mean(-w * (y_prot * np.log(pr + 1e-7) + (1.0 - y_prot) * np.log(1.0 - pr + 1e-7))))
                d_out[:, 5] = w * (pr - y_prot) / bsz
                err_p = out[:, 6] - y_prog
                loss += float(0.25 * np.mean(err_p * err_p))
                d_out[:, 6] = 0.5 * err_p / bsz
            if variant == "full_v3_neurocover":
                err_u = out[:, 7] - y_du
                loss += float(0.25 * np.mean(err_u * err_u))
                d_out[:, 7] = 0.5 * err_u / bsz

        grads: Dict[str, np.ndarray] = {}
        grads["h2_w"] = d_out.T @ h0
        grads["h2_b"] = d_out.sum(axis=0)
        d_h0 = (d_out @ p["h2_w"]) * (h0_pre > 0.0)
        grads["h0_w"] = d_h0.T @ h_in
        grads["h0_b"] = d_h0.sum(axis=0)
        d_hin = d_h0 @ p["h0_w"]

        g_act = np.zeros_like(p["act_emb"])
        np.add.at(g_act, act, d_hin[:, d:])
        grads["act_emb"] = g_act

        d_h = d_hin[:, :d]
        g_xw = np.zeros_like(p["x_w"])
        g_hw = np.zeros_like(p["h_w"])
        g_b = np.zeros_like(p["b"])

        for t in range(seq_len - 1, -1, -1):
            xt, ht = x_seq[:, t, :], hs[:, t, :]
            r, z, nt, ght = rs[:, t, :], zs[:, t, :], ntilde[:, t, :], gh[:, t, :]
            d_n = d_h * z * (1.0 - nt * nt)
            d_z = d_h * (nt - ht) * z * (1.0 - z)
            d_r = (d_n * ght) * r * (1.0 - r)

            g_xw[:d] += d_r.T @ xt
            g_xw[d : 2 * d] += d_z.T @ xt
            g_xw[2 * d :] += d_n.T @ xt
            g_hw[:d] += d_r.T @ ht
            g_hw[d : 2 * d] += d_z.T @ ht
            g_hw[2 * d :] += (d_n * r).T @ ht
            g_b[:d] += d_r.sum(axis=0)
            g_b[d : 2 * d] += d_z.sum(axis=0)
            g_b[2 * d :] += d_n.sum(axis=0)
            d_h = d_h * (1.0 - z) + d_r @ wh_r + d_z @ wh_z + (d_n * r) @ wh_n

        grads["x_w"] = g_xw
        grads["h_w"] = g_hw
        grads["b"] = g_b
        return loss, out, grads


def _compute_cumulative_risks(out: np.ndarray) -> np.ndarray:
    """R_k = 1 - prod_{j=0..k} (1 - q_j), guaranteed non-decreasing across horizons."""
    q = _sigmoid(out[:, :5])
    surv = np.cumprod(1.0 - q, axis=1)
    return 1.0 - surv


def evaluate_pilot_predictions(out: np.ndarray, targets: dict, variant: str) -> dict:
    r_cum = _compute_cumulative_risks(out)
    r_horizons = targets["y_risk_horizons"]
    y_prot = targets["y_risk_protected"]
    y_prog = targets["y_progress_protected"]
    y_du = targets["y_signed_delta_u"]

    if variant == "single_24s":
        r_24 = _sigmoid(out[:, 4])
        return {
            "brier_24s": float(np.mean((r_24 - r_horizons[:, 4]) ** 2)),
            "brier_03s_cross_err": float(np.mean((r_24 - y_prot) ** 2)),
        }
    if variant == "short_03s":
        pr = _sigmoid(out[:, 5])
        return {
            "brier_protected": float(np.mean((pr - y_prot) ** 2)),
            "brier_24s_cross_err": float(np.mean((pr - r_horizons[:, 4]) ** 2)),
        }

    pr = _sigmoid(out[:, 5])
    return {
        "brier_survival_horizons": [float(np.mean((r_cum[:, k] - r_horizons[:, k]) ** 2)) for k in range(5)],
        "brier_03s_cum": float(np.mean((r_cum[:, 1] - r_horizons[:, 1]) ** 2)),
        "brier_24s_cum": float(np.mean((r_cum[:, 4] - r_horizons[:, 4]) ** 2)),
        "brier_protected": float(np.mean((pr - y_prot) ** 2)),
        "progress_mae": float(np.mean(np.abs(out[:, 6] - y_prog))),
        "delta_u_mae": float(np.mean(np.abs(out[:, 7] - y_du))),
        "survival_monotonic": bool(np.all(np.diff(r_cum, axis=1) >= -1e-9)),
    }


def train_pilot_suite(data_dir: Path, bio_dir: Path, ann_csv: Path, out_dir: Path, epochs: int = 80, lr: float = 4e-3) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes_txt = bio_dir / "nodes.txt"
    pre_signs = _load_nt_signs(ann_csv, nodes_txt)

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
    # Standardize inputs strictly from train split statistics
    mu = tr["X"].mean(axis=(0, 1), keepdims=True)
    sd = np.maximum(tr["X"].std(axis=(0, 1), keepdims=True), 1e-4)
    for d in (tr, va, te):
        d["X_norm"] = (d["X"] - mu) / sd

    def _masks(npz_path: Path):
        mat = load_npz(npz_path).toarray()
        adj = (mat != 0).astype(np.float64)
        sign = np.broadcast_to(pre_signs[None, :], mat.shape).copy()
        return adj, sign

    adj_bio, sgn_bio = _masks(bio_dir / "biological.npz")
    adj_rew, sgn_rew = _masks(bio_dir / "rewired-7.npz")
    in_dim = tr["X"].shape[-1]

    runs_cfg = [
        ("Single24s_Bio", "bio", "single_24s"),
        ("Short03s_Bio", "bio", "short_03s"),
        ("MultiHorizon_Bio", "bio", "multi_horizon"),
        ("MultiHorizonProt_Bio", "bio", "multi_horizon_prot"),
        ("FullV3_NeuroCOVER_Bio", "bio", "full_v3_neurocover"),
        ("FullV3_NeuroCOVER_Rewired7", "rewired", "full_v3_neurocover"),
        ("FullV3_NeuroCOVER_GRU", "gru", "full_v3_neurocover"),
    ]

    metrics = {}
    exported = {"feat_mean": mu.squeeze().astype(np.float32), "feat_std": sd.squeeze().astype(np.float32)}
    rng = np.random.default_rng(7)

    for run_name, core_kind, variant in runs_cfg:
        if core_kind == "bio":
            model = _BioRecurrentModel(adj_bio, sgn_bio, in_dim, seed=7)
        elif core_kind == "rewired":
            model = _BioRecurrentModel(adj_rew, sgn_rew, in_dim, seed=7)
        else:
            model = _GRURecurrentModel(in_dim, hidden_dim=64, seed=7)

        m_opt = {k: np.zeros_like(v) for k, v in model.params.items()}
        v_opt = {k: np.zeros_like(v) for k, v in model.params.items()}
        t_adam = 0

        for ep in range(epochs):
            perm = rng.permutation(len(tr["X_norm"]))
            for start in range(0, len(perm), 32):
                idx = perm[start : start + 32]
                sub_t = {k: v[idx] for k, v in tr.items() if k.startswith("y_")}
                _, _, grads = model.forward_and_grads(tr["X_norm"][idx], tr["action"][idx], sub_t, variant)
                t_adam += 1
                for k, g in grads.items():
                    g = np.clip(g, -5.0, 5.0)
                    m_opt[k] = 0.9 * m_opt[k] + 0.1 * g
                    v_opt[k] = 0.999 * v_opt[k] + 0.001 * (g * g)
                    m_hat = m_opt[k] / (1.0 - 0.9 ** t_adam)
                    v_hat = v_opt[k] / (1.0 - 0.999 ** t_adam)
                    model.params[k] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

        _, va_out, _ = model.forward_and_grads(va["X_norm"], va["action"], va, variant)
        _, te_out, _ = model.forward_and_grads(te["X_norm"], te["action"], te, variant)
        metrics[run_name] = {
            "core": core_kind,
            "variant": variant,
            "val": evaluate_pilot_predictions(va_out, va, variant),
            "test": evaluate_pilot_predictions(te_out, te, variant),
        }
        exported[f"{run_name}__core_type"] = np.array([core_kind])
        if core_kind in ("bio", "rewired"):
            exported[f"{run_name}__adj"] = model.adj.astype(np.float32)
            exported[f"{run_name}__sign"] = model.sign.astype(np.float32)
        for pk, pv in model.params.items():
            exported[f"{run_name}__{pk}"] = pv.astype(np.float32)

    np.savez_compressed(out_dir / "pilot_weights_v3.npz", **exported)
    report = {"epochs": epochs, "runs": metrics}
    (out_dir / "pilot_metrics.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("runs/pilot-v3-p6"))
    p.add_argument("--bio-dir", type=Path, default=Path("runs/graph-controls/escape"))
    p.add_argument("--ann", type=Path, default=Path("runs/connectome/escape/annotations.csv"))
    p.add_argument("--out", type=Path, default=Path("runs/pilot-v3-models"))
    p.add_argument("--epochs", type=int, default=80)
    args = p.parse_args()
    print(json.dumps(train_pilot_suite(args.data, args.bio_dir, args.ann, args.out, epochs=args.epochs), indent=2))
