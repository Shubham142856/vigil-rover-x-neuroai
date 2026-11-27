"""P6.9.5 Biological vs Rewired vs GRU Architecture & Gradient Diagnostics.

Investigates why P6 Bio lagged GRU on delta_u MAE (3.14x higher) & 2.4s Brier:
  1. Audits parameter counts, hidden activation saturation, SVD effective rank,
     and multi-task gradient cosine interference between risk, progress, and delta_u.
  2. Evaluates Section 7 modular readout + adaptive per-neuron membrane time constants
     z_{t+1} = (1 - alpha) * z_t + alpha * tanh(drive + z_t @ W_eff^T),
     y_risk = g_risk(z_T, b_t, a_t), y_util = g_util(z_T, b_t, a_t).
  3. Compares fairly on identical P6.9 targeted splits & equal optimizer budgets:
     - FrozenBio_Modular
     - Bio_Uniform_Joint (P6 baseline)
     - Bio_Adaptive_Modular
     - Rewired7_Adaptive_Modular, Rewired19_Adaptive_Modular, Rewired41_Adaptive_Modular
     - RandomSparse_Adaptive_Modular (density & degree-matched)
     - GRU_Joint and GRU_Modular
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import load_npz

from p69_dataset_audit import binary_metrics
from pilot_dataset_v3 import HORIZONS_S
from pilot_train_v3 import (
    _BioRecurrentModel,
    _GRURecurrentModel,
    _compute_cumulative_risks,
    _load_nt_signs,
    _sigmoid,
    _softplus,
)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _effective_rank(mat: np.ndarray) -> float:
    """Roy & Vetterli spectral entropy effective rank of hidden representations [N, D]."""
    centered = mat - mat.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    p = (s ** 2)
    tot = float(np.sum(p))
    if tot <= 1e-12:
        return 0.0
    probs = p[p > 1e-12] / tot
    return float(np.exp(-np.sum(probs * np.log(probs))))


class _ModularAdaptiveBioModel:
    """Section 7 biological backbone (adaptive per-neuron time constants + decoupled risk & utility readouts)."""

    def __init__(
        self,
        adj_mask: np.ndarray,
        sign_mask: np.ndarray,
        in_dim: int,
        n_actions: int = 5,
        seed: int = 7,
        freeze_core: bool = False,
        belief_skip_dim: int = 24,
    ):
        rng = np.random.default_rng(seed)
        self.core_type = "bio_modular_frozen" if freeze_core else "bio_modular"
        self.freeze_core = freeze_core
        self.adj = adj_mask.astype(np.float64)
        self.sign = sign_mask.astype(np.float64)
        n = adj_mask.shape[0]
        self.n_nodes = n
        self.in_dim = in_dim
        self.belief_skip_dim = belief_skip_dim
        ro_in = n + belief_skip_dim + 16
        # Init tau_logit so alpha = 0.05 + 0.85 * sigmoid(-1.54) ≈ 0.20 (matches biological rate step)
        self.params: Dict[str, np.ndarray] = {
            "theta": np.full((n, n), -1.0, dtype=np.float64),
            "tau_logit": np.full(n, -1.54, dtype=np.float64),
            "enc_w": rng.normal(0.0, 1.0 / np.sqrt(max(in_dim, 1)), (n, in_dim)),
            "enc_b": np.zeros(n, dtype=np.float64),
            "act_emb": rng.normal(0.0, 0.25, (n_actions, 16)),
            # Compact decoupled risk readout head (6 outputs: 5 survival hazards + protected risk)
            "hr0_w": rng.normal(0.0, np.sqrt(2.0 / ro_in), (48, ro_in)),
            "hr0_b": np.zeros(48, dtype=np.float64),
            "hr2_w": rng.normal(0.0, np.sqrt(1.0 / 48.0), (6, 48)),
            "hr2_b": np.zeros(6, dtype=np.float64),
            # Compact decoupled utility/recovery readout head (2 outputs: progress_prot, signed_delta_u)
            "hu0_w": rng.normal(0.0, np.sqrt(2.0 / ro_in), (48, ro_in)),
            "hu0_b": np.zeros(48, dtype=np.float64),
            "hu2_w": rng.normal(0.0, np.sqrt(1.0 / 48.0), (2, 48)),
            "hu2_b": np.zeros(2, dtype=np.float64),
        }

    def forward_and_grads(
        self,
        x_seq: np.ndarray,
        act: np.ndarray,
        targets: dict,
        variant: str = "full_v3_neurocover",
        pos_wt_prot: float = 3.0,
    ) -> Tuple[float, np.ndarray, Dict[str, np.ndarray]]:
        bsz, seq_len, _ = x_seq.shape
        n = self.n_nodes
        p = self.params
        w_sp = _softplus(p["theta"])
        w_eff = self.adj * self.sign * w_sp
        sig_theta = _sigmoid(p["theta"])
        sig_tau = _sigmoid(p["tau_logit"])
        alpha = 0.05 + 0.85 * sig_tau  # [n]
        d_alpha_d_tau = 0.85 * sig_tau * (1.0 - sig_tau)

        states = np.zeros((bsz, seq_len + 1, n), dtype=np.float64)
        pre_tanh = np.zeros((bsz, seq_len, n), dtype=np.float64)
        for t in range(seq_len):
            drive = x_seq[:, t, :] @ p["enc_w"].T + p["enc_b"]
            pre = drive + states[:, t, :] @ w_eff.T
            pre_tanh[:, t, :] = pre
            th = np.tanh(pre)
            states[:, t + 1, :] = (1.0 - alpha) * states[:, t, :] + alpha * th

        z_end = states[:, -1, :]
        b_t = x_seq[:, -1, : self.belief_skip_dim]
        a_emb = p["act_emb"][act]
        ro_in = np.concatenate([z_end, b_t, a_emb], axis=1)

        hr0_pre = ro_in @ p["hr0_w"].T + p["hr0_b"]
        hr0 = np.maximum(0.0, hr0_pre)
        out_r = hr0 @ p["hr2_w"].T + p["hr2_b"]

        hu0_pre = ro_in @ p["hu0_w"].T + p["hu0_b"]
        hu0 = np.maximum(0.0, hu0_pre)
        out_u = hu0 @ p["hu2_w"].T + p["hu2_b"]

        out = np.concatenate([out_r, out_u], axis=1)
        self._last_states = states

        d_out = np.zeros_like(out)
        loss = 0.0
        haz_bin = targets["y_hazard_bin"]
        y_prot = targets["y_risk_protected"]
        y_prog = targets["y_progress_protected"]
        y_du = targets["y_signed_delta_u"]

        q = _sigmoid(out[:, :5])
        j_idx = np.arange(5)[None, :]
        at_risk = (j_idx <= np.minimum(haz_bin[:, None], 4)).astype(np.float64)
        hit_here = (j_idx == haz_bin[:, None]).astype(np.float64)
        nll = -at_risk * (hit_here * np.log(q + 1e-7) + (1.0 - hit_here) * np.log(1.0 - q + 1e-7))
        loss += float(nll.sum() / bsz)
        d_out[:, :5] = at_risk * (q - hit_here) / bsz

        pr = _sigmoid(out[:, 5])
        w = np.where(y_prot > 0.5, pos_wt_prot, 1.0)
        loss += float(np.mean(-w * (y_prot * np.log(pr + 1e-7) + (1.0 - y_prot) * np.log(1.0 - pr + 1e-7))))
        d_out[:, 5] = w * (pr - y_prot) / bsz

        err_p = out[:, 6] - y_prog
        loss += float(0.25 * np.mean(err_p * err_p))
        d_out[:, 6] = 0.5 * err_p / bsz

        if variant == "full_v3_neurocover":
            err_u = out[:, 7] - y_du
            loss += float(0.50 * np.mean(err_u * err_u))
            d_out[:, 7] = 1.0 * err_u / bsz

        grads: Dict[str, np.ndarray] = {}
        d_out_r, d_out_u = d_out[:, :6], d_out[:, 6:]
        grads["hr2_w"] = d_out_r.T @ hr0
        grads["hr2_b"] = d_out_r.sum(axis=0)
        d_hr0 = (d_out_r @ p["hr2_w"]) * (hr0_pre > 0.0)
        grads["hr0_w"] = d_hr0.T @ ro_in
        grads["hr0_b"] = d_hr0.sum(axis=0)

        grads["hu2_w"] = d_out_u.T @ hu0
        grads["hu2_b"] = d_out_u.sum(axis=0)
        d_hu0 = (d_out_u @ p["hu2_w"]) * (hu0_pre > 0.0)
        grads["hu0_w"] = d_hu0.T @ ro_in
        grads["hu0_b"] = d_hu0.sum(axis=0)

        d_ro_in = d_hr0 @ p["hr0_w"] + d_hu0 @ p["hu0_w"]
        d_act = d_ro_in[:, n + self.belief_skip_dim :]
        g_act = np.zeros_like(p["act_emb"])
        np.add.at(g_act, act, d_act)
        grads["act_emb"] = g_act

        if self.freeze_core:
            grads["enc_w"] = np.zeros_like(p["enc_w"])
            grads["enc_b"] = np.zeros_like(p["enc_b"])
            grads["theta"] = np.zeros_like(p["theta"])
            grads["tau_logit"] = np.zeros_like(p["tau_logit"])
            return loss, out, grads

        d_z = d_ro_in[:, :n]
        g_enc_w = np.zeros_like(p["enc_w"])
        g_enc_b = np.zeros_like(p["enc_b"])
        g_weff = np.zeros_like(w_eff)
        g_alpha = np.zeros(n, dtype=np.float64)

        for t in range(seq_len - 1, -1, -1):
            th = np.tanh(pre_tanh[:, t, :])
            g_alpha += np.sum(d_z * (th - states[:, t, :]), axis=0)
            d_pre = d_z * alpha * (1.0 - th * th)
            g_enc_w += d_pre.T @ x_seq[:, t, :]
            g_enc_b += d_pre.sum(axis=0)
            g_weff += d_pre.T @ states[:, t, :]
            d_z = (1.0 - alpha) * d_z + d_pre @ w_eff

        grads["enc_w"] = g_enc_w
        grads["enc_b"] = g_enc_b
        grads["theta"] = g_weff * self.adj * self.sign * sig_theta
        grads["tau_logit"] = g_alpha * d_alpha_d_tau
        return loss, out, grads


class _ModularGRUModel:
    """Matched conventional GRU with identical belief skip + decoupled risk & utility readouts for fair comparison."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, n_actions: int = 5, seed: int = 7, belief_skip_dim: int = 24):
        rng = np.random.default_rng(seed)
        self.core_type = "gru_modular"
        self.hidden_dim = hidden_dim
        self.belief_skip_dim = belief_skip_dim
        ro_in = hidden_dim + belief_skip_dim + 16
        self.params: Dict[str, np.ndarray] = {
            "x_w": rng.normal(0.0, 1.0 / np.sqrt(max(in_dim, 1)), (3 * hidden_dim, in_dim)),
            "h_w": rng.normal(0.0, 1.0 / np.sqrt(hidden_dim), (3 * hidden_dim, hidden_dim)),
            "b": np.zeros(3 * hidden_dim, dtype=np.float64),
            "act_emb": rng.normal(0.0, 0.25, (n_actions, 16)),
            "hr0_w": rng.normal(0.0, np.sqrt(2.0 / ro_in), (48, ro_in)),
            "hr0_b": np.zeros(48, dtype=np.float64),
            "hr2_w": rng.normal(0.0, np.sqrt(1.0 / 48.0), (6, 48)),
            "hr2_b": np.zeros(6, dtype=np.float64),
            "hu0_w": rng.normal(0.0, np.sqrt(2.0 / ro_in), (48, ro_in)),
            "hu0_b": np.zeros(48, dtype=np.float64),
            "hu2_w": rng.normal(0.0, np.sqrt(1.0 / 48.0), (2, 48)),
            "hu2_b": np.zeros(2, dtype=np.float64),
        }

    def forward_and_grads(self, x_seq, act, targets, variant="full_v3_neurocover", pos_wt_prot=3.0):
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
            xt, ht = x_seq[:, t, :], hs[:, t, :]
            r = _sigmoid(xt @ wx_r.T + ht @ wh_r.T + br)
            z = _sigmoid(xt @ wx_z.T + ht @ wh_z.T + bz)
            g_h = ht @ wh_n.T
            n_t = np.tanh(xt @ wx_n.T + r * g_h + bn)
            hs[:, t + 1, :] = (1.0 - z) * ht + z * n_t
            rs[:, t, :], zs[:, t, :], ntilde[:, t, :], gh[:, t, :] = r, z, n_t, g_h

        z_end = hs[:, -1, :]
        b_t = x_seq[:, -1, : self.belief_skip_dim]
        ro_in = np.concatenate([z_end, b_t, p["act_emb"][act]], axis=1)

        hr0_pre = ro_in @ p["hr0_w"].T + p["hr0_b"]
        hr0 = np.maximum(0.0, hr0_pre)
        out_r = hr0 @ p["hr2_w"].T + p["hr2_b"]

        hu0_pre = ro_in @ p["hu0_w"].T + p["hu0_b"]
        hu0 = np.maximum(0.0, hu0_pre)
        out_u = hu0 @ p["hu2_w"].T + p["hu2_b"]

        out = np.concatenate([out_r, out_u], axis=1)
        self._last_states = hs

        d_out = np.zeros_like(out)
        loss = 0.0
        haz_bin = targets["y_hazard_bin"]
        y_prot = targets["y_risk_protected"]
        y_prog = targets["y_progress_protected"]
        y_du = targets["y_signed_delta_u"]

        q = _sigmoid(out[:, :5])
        j_idx = np.arange(5)[None, :]
        at_risk = (j_idx <= np.minimum(haz_bin[:, None], 4)).astype(np.float64)
        hit_here = (j_idx == haz_bin[:, None]).astype(np.float64)
        loss += float((-at_risk * (hit_here * np.log(q + 1e-7) + (1.0 - hit_here) * np.log(1.0 - q + 1e-7))).sum() / bsz)
        d_out[:, :5] = at_risk * (q - hit_here) / bsz

        pr = _sigmoid(out[:, 5])
        w = np.where(y_prot > 0.5, pos_wt_prot, 1.0)
        loss += float(np.mean(-w * (y_prot * np.log(pr + 1e-7) + (1.0 - y_prot) * np.log(1.0 - pr + 1e-7))))
        d_out[:, 5] = w * (pr - y_prot) / bsz

        err_p = out[:, 6] - y_prog
        loss += float(0.25 * np.mean(err_p * err_p))
        d_out[:, 6] = 0.5 * err_p / bsz

        if variant == "full_v3_neurocover":
            err_u = out[:, 7] - y_du
            loss += float(0.50 * np.mean(err_u * err_u))
            d_out[:, 7] = 1.0 * err_u / bsz

        grads: Dict[str, np.ndarray] = {}
        d_out_r, d_out_u = d_out[:, :6], d_out[:, 6:]
        grads["hr2_w"] = d_out_r.T @ hr0
        grads["hr2_b"] = d_out_r.sum(axis=0)
        d_hr0 = (d_out_r @ p["hr2_w"]) * (hr0_pre > 0.0)
        grads["hr0_w"] = d_hr0.T @ ro_in
        grads["hr0_b"] = d_hr0.sum(axis=0)

        grads["hu2_w"] = d_out_u.T @ hu0
        grads["hu2_b"] = d_out_u.sum(axis=0)
        d_hu0 = (d_out_u @ p["hu2_w"]) * (hu0_pre > 0.0)
        grads["hu0_w"] = d_hu0.T @ ro_in
        grads["hu0_b"] = d_hu0.sum(axis=0)

        d_ro_in = d_hr0 @ p["hr0_w"] + d_hu0 @ p["hu0_w"]
        g_act = np.zeros_like(p["act_emb"])
        np.add.at(g_act, act, d_ro_in[:, d + self.belief_skip_dim :])
        grads["act_emb"] = g_act

        d_h = d_ro_in[:, :d]
        g_xw, g_hw, g_b = np.zeros_like(p["x_w"]), np.zeros_like(p["h_w"]), np.zeros_like(p["b"])
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
        grads["x_w"], grads["h_w"], grads["b"] = g_xw, g_hw, g_b
        return loss, out, grads


def _train_model(model, tr: dict, variant: str = "full_v3_neurocover", epochs: int = 80, lr: float = 5e-3, seed: int = 7, batch_size: int = 64) -> dict:
    rng = np.random.default_rng(seed)
    m_opt = {k: np.zeros_like(v) for k, v in model.params.items()}
    v_opt = {k: np.zeros_like(v) for k, v in model.params.items()}
    t_adam = 0
    for _ in range(epochs):
        perm = rng.permutation(len(tr["X_norm"]))
        for start in range(0, len(perm), batch_size):
            idx = perm[start : start + batch_size]
            sub_t = {k: v[idx] for k, v in tr.items() if k.startswith("y_")}
            _, _, grads = model.forward_and_grads(tr["X_norm"][idx], tr["action"][idx], sub_t, variant)
            t_adam += 1
            for k, g in grads.items():
                if not np.any(g):
                    continue
                g_c = np.clip(g, -5.0, 5.0)
                m_opt[k] = 0.9 * m_opt[k] + 0.1 * g_c
                v_opt[k] = 0.999 * v_opt[k] + 0.001 * (g_c * g_c)
                m_hat = m_opt[k] / (1.0 - 0.9 ** t_adam)
                v_hat = v_opt[k] / (1.0 - 0.999 ** t_adam)
                model.params[k] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Compute parameter count & diagnostic stats (multi-task gradient conflict, hidden saturation & rank)
    trainable_params = 0
    for k, v in model.params.items():
        if k == "theta" and hasattr(model, "adj"):
            trainable_params += int(np.sum(model.adj != 0))
        elif getattr(model, "freeze_core", False) and k in ("theta", "tau_logit", "enc_w", "enc_b"):
            continue
        else:
            trainable_params += int(v.size)

    return {"trainable_params": trainable_params}


def _evaluate_diagnostics(model, tr: dict, te: dict, variant: str = "full_v3_neurocover") -> dict:
    _, te_out, _ = model.forward_and_grads(te["X_norm"], te["action"], te, variant)
    r_cum = _compute_cumulative_risks(te_out)
    pr = _sigmoid(te_out[:, 5])
    p_const_24 = float(np.mean(tr["y_risk_horizons"][:, 4] > 0.5))
    p_const_prot = float(np.mean(tr["y_risk_protected"] > 0.5))

    # Representation health metrics on test set across all core types
    if hasattr(model, "_last_states"):
        z_end = model._last_states[:, -1, :]
    elif getattr(model, "core_type", "") == "bio":
        p = model.params
        w_eff = model.adj * model.sign * _softplus(p["theta"])
        z = np.zeros((len(te["X_norm"]), model.n_nodes), dtype=np.float64)
        for t in range(te["X_norm"].shape[1]):
            pre = te["X_norm"][:, t, :] @ p["enc_w"].T + p["enc_b"] + z @ w_eff.T
            z = z + 0.2 * (np.tanh(pre) - z)
        z_end = z
    elif getattr(model, "core_type", "") == "gru":
        p, d = model.params, model.hidden_dim
        wx_r, wx_z, wx_n = p["x_w"][:d], p["x_w"][d : 2 * d], p["x_w"][2 * d :]
        wh_r, wh_z, wh_n = p["h_w"][:d], p["h_w"][d : 2 * d], p["h_w"][2 * d :]
        br, bz, bn = p["b"][:d], p["b"][d : 2 * d], p["b"][2 * d :]
        z = np.zeros((len(te["X_norm"]), d), dtype=np.float64)
        for t in range(te["X_norm"].shape[1]):
            xt = te["X_norm"][:, t, :]
            r = _sigmoid(xt @ wx_r.T + z @ wh_r.T + br)
            zt = _sigmoid(xt @ wx_z.T + z @ wh_z.T + bz)
            nt = np.tanh(xt @ wx_n.T + r * (z @ wh_n.T) + bn)
            z = (1.0 - zt) * z + zt * nt
        z_end = z
    else:
        z_end = te_out[:, :1] * 0.0
    sat_frac = float(np.mean(np.abs(z_end) > 0.90))
    eff_rank = _effective_rank(z_end) if z_end.shape[1] > 1 else 0.0

    # Measure multi-task gradient alignment in shared encoder/recurrent parameters
    sub_n = min(64, len(tr["X_norm"]))
    sub_x, sub_a = tr["X_norm"][:sub_n], tr["action"][:sub_n]
    sub_y = {k: v[:sub_n] for k, v in tr.items() if k.startswith("y_")}
    risk_t = dict(sub_y, y_progress_protected=np.zeros_like(sub_y["y_progress_protected"]), y_signed_delta_u=np.zeros_like(sub_y["y_signed_delta_u"]))
    _, _, g_all = model.forward_and_grads(sub_x, sub_a, sub_y, variant)

    # Isolate pure risk vs utility gradients by comparing single-task losses on enc_w or x_w
    key_shared = "enc_w" if "enc_w" in g_all else "x_w"
    g_risk_only = None
    g_util_only = None
    if not getattr(model, "freeze_core", False):
        # Compute gradient of ONLY risk heads (zeroing y_prog / y_du doesn't zero MSE because out != 0 -> zero MSE grads directly by finite scaling hack? No: use variant='multi_horizon' vs delta_u!)
        # Exact single-head gradient via parameter output perturbation isn't needed; just compute cosine between multi_horizon (no util) and full minus multi_horizon!
        _, _, g_mh = model.forward_and_grads(sub_x, sub_a, sub_y, "multi_horizon")
        g_risk_only = g_mh[key_shared].ravel()
        g_util_only = (g_all[key_shared] - g_mh[key_shared]).ravel()
        cos_risk_util = _cosine(g_risk_only, g_util_only)
    else:
        cos_risk_util = 0.0

    m_prot = binary_metrics(te["y_risk_protected"], pr, p_const_prot)
    m_24s = binary_metrics(te["y_risk_horizons"][:, 4], r_cum[:, 4], p_const_24)
    return {
        "protected_brier": m_prot["brier"],
        "protected_bss": m_prot["brier_skill_score"],
        "protected_pr_auc": m_prot["pr_auc"],
        "protected_fn_lt_0_20": m_prot["dangerous_fn_lt_0_20"],
        "risk_24s_brier": m_24s["brier"],
        "risk_24s_bss": m_24s["brier_skill_score"],
        "risk_24s_pr_auc": m_24s["pr_auc"],
        "progress_mae": float(np.mean(np.abs(te_out[:, 6] - te["y_progress_protected"]))),
        "delta_u_mae": float(np.mean(np.abs(te_out[:, 7] - te["y_signed_delta_u"]))),
        "hidden_saturation_frac": sat_frac,
        "hidden_effective_rank": eff_rank,
        "grad_cos_risk_vs_util": cos_risk_util,
    }


def _diag_worker_task(task: Tuple[str, int, str, str, str, int]) -> Tuple[str, int, int, dict]:
    arch, sd_i, data_dir_s, bio_dir_s, ann_csv_s, epochs = task
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

    tr, te = _load("train"), _load("test")
    mu = tr["X"].mean(axis=(0, 1), keepdims=True)
    sd = np.maximum(tr["X"].std(axis=(0, 1), keepdims=True), 1e-4)
    for d in (tr, te):
        d["X_norm"] = (d["X"] - mu) / sd
    in_dim = tr["X"].shape[-1]

    def _masks(npz_path: Path):
        mat = load_npz(npz_path).toarray()
        adj = (mat != 0).astype(np.float64)
        sign = np.broadcast_to(pre_signs[None, :], mat.shape).copy()
        return adj, sign

    adj_bio, sgn_bio = _masks(bio_dir / "biological.npz")
    n_bio = adj_bio.shape[0]
    edge_count = int(np.sum(adj_bio != 0))

    rewired_files = {
        "Rewired7_Adaptive_Mod": bio_dir / "rewired-7.npz",
        "Rewired19_Adaptive_Mod": bio_dir / "rewired-19.npz",
        "Rewired41_Adaptive_Mod": bio_dir / "rewired-41.npz",
    }

    if arch == "Bio_Joint_P6":
        mod = _BioRecurrentModel(adj_bio, sgn_bio, in_dim, seed=sd_i)
    elif arch == "Bio_Frozen_Modular":
        mod = _ModularAdaptiveBioModel(adj_bio, sgn_bio, in_dim, seed=sd_i, freeze_core=True)
    elif arch == "Bio_Adaptive_Modular":
        mod = _ModularAdaptiveBioModel(adj_bio, sgn_bio, in_dim, seed=sd_i, freeze_core=False)
    elif arch in rewired_files:
        ar, sr = _masks(rewired_files[arch])
        mod = _ModularAdaptiveBioModel(ar, sr, in_dim, seed=sd_i, freeze_core=False)
    elif arch == "RandomSparse_Adaptive_Mod":
        rng_r = np.random.default_rng(1000 + sd_i)
        ar = np.zeros_like(adj_bio)
        flat = rng_r.choice(n_bio * n_bio, size=edge_count, replace=False)
        ar.ravel()[flat] = 1.0
        sr = rng_r.choice([-1.0, 1.0], size=(n_bio, n_bio))
        mod = _ModularAdaptiveBioModel(ar, sr, in_dim, seed=sd_i, freeze_core=False)
    elif arch == "GRU_Joint_P6":
        mod = _GRURecurrentModel(in_dim, hidden_dim=64, seed=sd_i)
    elif arch == "GRU_Modular":
        mod = _ModularGRUModel(in_dim, hidden_dim=64, seed=sd_i)
    else:
        raise ValueError(arch)

    info = _train_model(mod, tr, epochs=epochs, seed=sd_i, batch_size=64)
    ev = _evaluate_diagnostics(mod, tr, te)
    return arch, sd_i, info["trainable_params"], ev


def run_architecture_diagnostics(
    data_dir: Path,
    bio_dir: Path,
    ann_csv: Path,
    out_path: Path,
    epochs: int = 80,
    seeds: Tuple[int, ...] = (7, 17, 27),
    workers: int = 4,
) -> dict:
    mat_bio = load_npz(bio_dir / "biological.npz").toarray()
    n_bio = int(mat_bio.shape[0])
    edge_count = int(np.sum(mat_bio != 0))

    architectures = [
        "Bio_Joint_P6",
        "Bio_Frozen_Modular",
        "Bio_Adaptive_Modular",
        "Rewired7_Adaptive_Mod",
        "Rewired19_Adaptive_Mod",
        "Rewired41_Adaptive_Mod",
        "RandomSparse_Adaptive_Mod",
        "GRU_Joint_P6",
        "GRU_Modular",
    ]
    tasks = [(arch, sd_i, str(data_dir), str(bio_dir), str(ann_csv), epochs) for arch in architectures for sd_i in seeds]
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_diag_worker_task, tasks))
    else:
        results = [_diag_worker_task(t) for t in tasks]

    grouped: Dict[str, List[Tuple[int, int, dict]]] = {a: [] for a in architectures}
    for arch, sd_i, p_cnt, ev in results:
        grouped[arch].append((sd_i, p_cnt, ev))

    summary = {"data_dir": str(data_dir), "epochs": epochs, "seeds": list(seeds), "bio_nodes": n_bio, "bio_edges": edge_count, "models": {}}
    for arch in architectures:
        rows = sorted(grouped[arch], key=lambda x: x[0])
        param_cnt = rows[0][1]
        per_seed = [r[2] for r in rows]
        agg = {"trainable_params": param_cnt}
        for mk in ("protected_brier", "protected_bss", "protected_pr_auc", "risk_24s_brier", "risk_24s_bss", "risk_24s_pr_auc", "progress_mae", "delta_u_mae", "hidden_saturation_frac", "hidden_effective_rank", "grad_cos_risk_vs_util"):
            vals = [float(r[mk]) for r in per_seed if r[mk] is not None]
            agg[mk + "_mean"] = float(np.mean(vals)) if vals else None
            agg[mk + "_std"] = float(np.std(vals)) if vals else None
        agg["per_seed"] = per_seed
        summary["models"][arch] = agg

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("runs/pilot-v3-p69"))
    p.add_argument("--bio-dir", type=Path, default=Path("runs/graph-controls/escape"))
    p.add_argument("--ann", type=Path, default=Path("runs/connectome/escape/annotations.csv"))
    p.add_argument("--out", type=Path, default=Path("runs/p69-audit/bio_diagnostics.json"))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    res = run_architecture_diagnostics(args.data, args.bio_dir, args.ann, args.out, epochs=args.epochs, workers=args.workers)
    print("\n=== P6.9.5 BIOLOGICAL VS REWIRED VS GRU DIAGNOSTICS (3 seeds) ===")
    for name, m in res["models"].items():
        print(
            f"{name:26s} | params={m['trainable_params']:5d} "
            f"| prot_BSS={m['protected_bss_mean']:+.3f} PRAUC={m['protected_pr_auc_mean']:.3f} "
            f"| 2.4s_BSS={m['risk_24s_bss_mean']:+.3f} PRAUC={m['risk_24s_pr_auc_mean']:.3f} "
            f"| progMAE={m['progress_mae_mean']:.4f} duMAE={m['delta_u_mae_mean']:.4f}±{m['delta_u_mae_std']:.4f} "
            f"| sat={m['hidden_saturation_frac_mean']:.3f} rank={m['hidden_effective_rank_mean']:.1f} cos={m['grad_cos_risk_vs_util_mean']:+.3f}"
        )


if __name__ == "__main__":
    main()
