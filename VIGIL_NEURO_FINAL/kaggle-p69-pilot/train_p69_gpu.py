"""P6.9.5 & P6.9.6 Kaggle GPU Trainer (PyTorch CUDA -> Exact Portable NumPy Weights + Diagnostics).

Clones pinned GitHub commit containing `runs/pilot-v3-p69/` and runs:
  1. P6.9.5 Architecture & Gradient Diagnostics (9 architectures x 3 seeds = 27 models):
       - Bio_Joint_P6, Bio_Frozen_Modular, Bio_Adaptive_Modular
       - Rewired7/19/41_Adaptive_Mod, RandomSparse_Adaptive_Mod
       - GRU_Joint_P6, GRU_Modular
     Exports `bio_diagnostics.json` with exact param counts, BSS, NLL, PR-AUC, FNs,
     hidden saturation, SVD effective rank, and risk-vs-utility gradient cosine alignment.
  2. P6.9.6 Matched Multi-Seed Training (8 policy ensembles x 3 seeds = 24 models):
       - Single24s_Bio, MultiProt_NoRecovTrain_Bio, Full_Bio_Joint, Full_Bio_Modular,
         Full_Rewired7/19/41_Modular, Full_GRU_Modular
     Exports `p69_weights.npz` (100% compatible with `_PortableP69Policy`) and `p69_offline_metrics.json`.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import load_npz

REPO_URL = "https://github.com/Shubham142856/vigil-rover-x-neuroai.git"
HORIZONS_S = (0.10, 0.30, 0.60, 1.20, 2.40)


def _load_nt_signs(annotations_csv: Path, nodes_txt: Path) -> np.ndarray:
    ids = nodes_txt.read_text().splitlines()
    with annotations_csv.open() as f:
        nt_map = {r["root_id"]: r["predicted_nt"] for r in csv.DictReader(f)}
    pre_signs = np.zeros(len(ids), dtype=np.float32)
    for idx, rid in enumerate(ids):
        nt = nt_map.get(rid, "")
        if nt == "ACH":
            pre_signs[idx] = 1.0
        elif nt == "GLUT":
            pre_signs[idx] = -1.0
    return pre_signs


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _effective_rank(mat: np.ndarray) -> float:
    centered = mat - mat.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    p = s ** 2
    tot = float(np.sum(p))
    if tot <= 1e-12:
        return 0.0
    probs = p[p > 1e-12] / tot
    return float(np.exp(-np.sum(probs * np.log(probs))))


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, p_const: float, n_bins: int = 5) -> dict:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    n = int(len(y))
    pos = int(np.sum(y > 0.5))
    bs = float(np.mean((p - y) ** 2))
    nll = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    p_c = float(np.clip(p_const, 1e-7, 1.0 - 1e-7))
    bs_const = float(np.mean((p_c - y) ** 2))
    nll_const = float(-np.mean(y * np.log(p_c) + (1.0 - y) * np.log(1.0 - p_c)))
    bs_zero = float(np.mean((1e-7 - y) ** 2))
    bss = float(1.0 - bs / bs_const) if bs_const > 1e-12 else 0.0

    if pos == 0:
        pr_auc = recall_50 = recall_20 = None
        danger_fn = 0
    else:
        order = np.argsort(-p)
        y_s = y[order] > 0.5
        tp_cum = np.cumsum(y_s).astype(np.float64)
        precisions = tp_cum / np.arange(1, n + 1, dtype=np.float64)
        pr_auc = float(np.sum(precisions * y_s) / float(pos))
        recall_50 = float(np.sum((p >= 0.50) & (y > 0.5)) / float(pos))
        recall_20 = float(np.sum((p >= 0.20) & (y > 0.5)) / float(pos))
        danger_fn = int(np.sum((p < 0.20) & (y > 0.5)))

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b_idx in range(n_bins):
        lo, hi = edges[b_idx], edges[b_idx + 1]
        mask = (p >= lo) & (p <= hi if b_idx == n_bins - 1 else p < hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            ece += (cnt / max(n, 1)) * abs(float(np.mean(p[mask])) - float(np.mean(y[mask])))

    return {
        "n": n,
        "positives": pos,
        "negatives": n - pos,
        "prevalence": float(pos / max(n, 1)),
        "brier": bs,
        "brier_const_baseline": bs_const,
        "brier_always_zero_baseline": bs_zero,
        "brier_skill_score": bss,
        "nll": nll,
        "nll_const_baseline": nll_const,
        "pr_auc": pr_auc,
        "recall_at_0_50": recall_50,
        "recall_at_0_20": recall_20,
        "dangerous_fn_lt_0_20": danger_fn,
        "ece": float(ece),
    }


def _compute_cum_risks(out_np: np.ndarray) -> np.ndarray:
    q = 1.0 / (1.0 + np.exp(-np.clip(out_np[:, :5], -30.0, 30.0)))
    return 1.0 - np.cumprod(1.0 - q, axis=1)


class JointBioNet(nn.Module):
    def __init__(self, adj: np.ndarray, sign: np.ndarray, in_dim: int, core_type: str = "bio", seed: int = 7):
        super().__init__()
        self.core_type = core_type
        self.register_buffer("adj", torch.tensor(adj, dtype=torch.float32))
        self.register_buffer("sign", torch.tensor(sign, dtype=torch.float32))
        n = adj.shape[0]
        self.n_nodes = n
        rng = np.random.default_rng(seed)
        self.theta = nn.Parameter(torch.full((n, n), -1.0, dtype=torch.float32))
        self.enc_w = nn.Parameter(torch.tensor(rng.normal(0.0, 1.0 / math.sqrt(in_dim), (n, in_dim)), dtype=torch.float32))
        self.enc_b = nn.Parameter(torch.zeros(n, dtype=torch.float32))
        self.act_emb = nn.Parameter(torch.tensor(rng.normal(0.0, 0.25, (5, 16)), dtype=torch.float32))
        self.h0_w = nn.Parameter(torch.tensor(rng.normal(0.0, math.sqrt(2.0 / (n + 16)), (64, n + 16)), dtype=torch.float32))
        self.h0_b = nn.Parameter(torch.zeros(64, dtype=torch.float32))
        self.h2_w = nn.Parameter(torch.tensor(rng.normal(0.0, math.sqrt(1.0 / 64.0), (8, 64)), dtype=torch.float32))
        self.h2_b = nn.Parameter(torch.zeros(8, dtype=torch.float32))

    def encode(self, x_seq: torch.Tensor) -> torch.Tensor:
        w_eff = self.adj * self.sign * F.softplus(self.theta)
        z = torch.zeros(x_seq.shape[0], self.n_nodes, device=x_seq.device)
        drives = F.linear(x_seq, self.enc_w, self.enc_b)
        wt = w_eff.T
        for t in range(x_seq.shape[1]):
            z = z + 0.2 * (torch.tanh(drives[:, t, :] + z @ wt) - z)
        return z

    def forward(self, x_seq: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x_seq)
        h_in = torch.cat([z, self.act_emb[act]], dim=-1)
        h0 = F.relu(F.linear(h_in, self.h0_w, self.h0_b))
        out = F.linear(h0, self.h2_w, self.h2_b)
        return out, z


class JointGRUNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, seed: int = 7):
        super().__init__()
        self.core_type = "gru"
        self.hidden_dim = hidden_dim
        rng = np.random.default_rng(seed)
        d = hidden_dim
        self.x_w = nn.Parameter(torch.tensor(rng.normal(0.0, 1.0 / math.sqrt(in_dim), (3 * d, in_dim)), dtype=torch.float32))
        self.h_w = nn.Parameter(torch.tensor(rng.normal(0.0, 1.0 / math.sqrt(d), (3 * d, d)), dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(3 * d, dtype=torch.float32))
        self.act_emb = nn.Parameter(torch.tensor(rng.normal(0.0, 0.25, (5, 16)), dtype=torch.float32))
        self.h0_w = nn.Parameter(torch.tensor(rng.normal(0.0, math.sqrt(2.0 / (d + 16)), (64, d + 16)), dtype=torch.float32))
        self.h0_b = nn.Parameter(torch.zeros(64, dtype=torch.float32))
        self.h2_w = nn.Parameter(torch.tensor(rng.normal(0.0, math.sqrt(1.0 / 64.0), (8, 64)), dtype=torch.float32))
        self.h2_b = nn.Parameter(torch.zeros(8, dtype=torch.float32))

    def encode(self, x_seq: torch.Tensor) -> torch.Tensor:
        d = self.hidden_dim
        wx_r, wx_z, wx_n = self.x_w[:d], self.x_w[d : 2 * d], self.x_w[2 * d :]
        wh_r, wh_z, wh_n = self.h_w[:d], self.h_w[d : 2 * d], self.h_w[2 * d :]
        br, bz, bn = self.b[:d], self.b[d : 2 * d], self.b[2 * d :]
        h = torch.zeros(x_seq.shape[0], d, device=x_seq.device)
        dx_r = F.linear(x_seq, wx_r, br)
        dx_z = F.linear(x_seq, wx_z, bz)
        dx_n = F.linear(x_seq, wx_n, bn)
        for t in range(x_seq.shape[1]):
            r = torch.sigmoid(dx_r[:, t, :] + h @ wh_r.T)
            z = torch.sigmoid(dx_z[:, t, :] + h @ wh_z.T)
            nt = torch.tanh(dx_n[:, t, :] + r * (h @ wh_n.T))
            h = (1.0 - z) * h + z * nt
        return h

    def forward(self, x_seq: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x_seq)
        h_in = torch.cat([z, self.act_emb[act]], dim=-1)
        h0 = F.relu(F.linear(h_in, self.h0_w, self.h0_b))
        out = F.linear(h0, self.h2_w, self.h2_b)
        return out, z


class ModularBioNet(nn.Module):
    def __init__(self, adj: np.ndarray, sign: np.ndarray, in_dim: int, freeze_core: bool = False, belief_skip_dim: int = 24, seed: int = 7):
        super().__init__()
        self.core_type = "bio_modular_frozen" if freeze_core else "bio_modular"
        self.freeze_core = freeze_core
        self.belief_skip_dim = belief_skip_dim
        self.register_buffer("adj", torch.tensor(adj, dtype=torch.float32))
        self.register_buffer("sign", torch.tensor(sign, dtype=torch.float32))
        n = adj.shape[0]
        self.n_nodes = n
        ro_in = n + belief_skip_dim + 16
        rng = np.random.default_rng(seed)

        self.theta = nn.Parameter(torch.full((n, n), -1.0, dtype=torch.float32), requires_grad=(not freeze_core))
        self.tau_logit = nn.Parameter(torch.full((n,), -1.54, dtype=torch.float32), requires_grad=(not freeze_core))
        self.enc_w = nn.Parameter(torch.tensor(rng.normal(0.0, 1.0 / math.sqrt(in_dim), (n, in_dim)), dtype=torch.float32), requires_grad=(not freeze_core))
        self.enc_b = nn.Parameter(torch.zeros(n, dtype=torch.float32), requires_grad=(not freeze_core))
        self.act_emb = nn.Parameter(torch.tensor(rng.normal(0.0, 0.25, (5, 16)), dtype=torch.float32))
        self.hr0_w = nn.Parameter(torch.tensor(rng.normal(0.0, math.sqrt(2.0 / ro_in), (48, ro_in)), dtype=torch.float32))
        self.hr0_b = nn.Parameter(torch.zeros(48, dtype=torch.float32))
        self.hr2_w = nn.Parameter(torch.tensor(rng.normal(0.0, math.sqrt(1.0 / 48.0), (6, 48)), dtype=torch.float32))
        self.hr2_b = nn.Parameter(torch.zeros(6, dtype=torch.float32))
        self.hu0_w = nn.Parameter(torch.tensor(rng.normal(0.0, math.sqrt(2.0 / ro_in), (48, ro_in)), dtype=torch.float32))
        self.hu0_b = nn.Parameter(torch.zeros(48, dtype=torch.float32))
        self.hu2_w = nn.Parameter(torch.tensor(rng.normal(0.0, math.sqrt(1.0 / 48.0), (2, 48)), dtype=torch.float32))
        self.hu2_b = nn.Parameter(torch.zeros(2, dtype=torch.float32))

    def encode(self, x_seq: torch.Tensor) -> torch.Tensor:
        w_eff = self.adj * self.sign * F.softplus(self.theta)
        alpha = 0.05 + 0.85 * torch.sigmoid(self.tau_logit)
        one_m = 1.0 - alpha
        z = torch.zeros(x_seq.shape[0], self.n_nodes, device=x_seq.device)
        drives = F.linear(x_seq, self.enc_w, self.enc_b)
        wt = w_eff.T
        for t in range(x_seq.shape[1]):
            z = one_m * z + alpha * torch.tanh(drives[:, t, :] + z @ wt)
        return z

    def forward(self, x_seq: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x_seq)
        b_t = x_seq[:, -1, : self.belief_skip_dim]
        ro_in = torch.cat([z, b_t, self.act_emb[act]], dim=-1)
        out_r = F.linear(F.relu(F.linear(ro_in, self.hr0_w, self.hr0_b)), self.hr2_w, self.hr2_b)
        out_u = F.linear(F.relu(F.linear(ro_in, self.hu0_w, self.hu0_b)), self.hu2_w, self.hu2_b)
        return torch.cat([out_r, out_u], dim=-1), z


class ModularGRUNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, belief_skip_dim: int = 24, seed: int = 7):
        super().__init__()
        self.core_type = "gru_modular"
        self.hidden_dim = hidden_dim
        self.belief_skip_dim = belief_skip_dim
        ro_in = hidden_dim + belief_skip_dim + 16
        d = hidden_dim
        rng = np.random.default_rng(seed)
        self.x_w = nn.Parameter(torch.tensor(rng.normal(0.0, 1.0 / math.sqrt(in_dim), (3 * d, in_dim)), dtype=torch.float32))
        self.h_w = nn.Parameter(torch.tensor(rng.normal(0.0, 1.0 / math.sqrt(d), (3 * d, d)), dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(3 * d, dtype=torch.float32))
        self.act_emb = nn.Parameter(torch.tensor(rng.normal(0.0, 0.25, (5, 16)), dtype=torch.float32))
        self.hr0_w = nn.Parameter(torch.tensor(rng.normal(0.0, math.sqrt(2.0 / ro_in), (48, ro_in)), dtype=torch.float32))
        self.hr0_b = nn.Parameter(torch.zeros(48, dtype=torch.float32))
        self.hr2_w = nn.Parameter(torch.tensor(rng.normal(0.0, math.sqrt(1.0 / 48.0), (6, 48)), dtype=torch.float32))
        self.hr2_b = nn.Parameter(torch.zeros(6, dtype=torch.float32))
        self.hu0_w = nn.Parameter(torch.tensor(rng.normal(0.0, math.sqrt(2.0 / ro_in), (48, ro_in)), dtype=torch.float32))
        self.hu0_b = nn.Parameter(torch.zeros(48, dtype=torch.float32))
        self.hu2_w = nn.Parameter(torch.tensor(rng.normal(0.0, math.sqrt(1.0 / 48.0), (2, 48)), dtype=torch.float32))
        self.hu2_b = nn.Parameter(torch.zeros(2, dtype=torch.float32))

    def encode(self, x_seq: torch.Tensor) -> torch.Tensor:
        d = self.hidden_dim
        wx_r, wx_z, wx_n = self.x_w[:d], self.x_w[d : 2 * d], self.x_w[2 * d :]
        wh_r, wh_z, wh_n = self.h_w[:d], self.h_w[d : 2 * d], self.h_w[2 * d :]
        br, bz, bn = self.b[:d], self.b[d : 2 * d], self.b[2 * d :]
        h = torch.zeros(x_seq.shape[0], d, device=x_seq.device)
        dx_r = F.linear(x_seq, wx_r, br)
        dx_z = F.linear(x_seq, wx_z, bz)
        dx_n = F.linear(x_seq, wx_n, bn)
        for t in range(x_seq.shape[1]):
            r = torch.sigmoid(dx_r[:, t, :] + h @ wh_r.T)
            z = torch.sigmoid(dx_z[:, t, :] + h @ wh_z.T)
            nt = torch.tanh(dx_n[:, t, :] + r * (h @ wh_n.T))
            h = (1.0 - z) * h + z * nt
        return h

    def forward(self, x_seq: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x_seq)
        b_t = x_seq[:, -1, : self.belief_skip_dim]
        ro_in = torch.cat([z, b_t, self.act_emb[act]], dim=-1)
        out_r = F.linear(F.relu(F.linear(ro_in, self.hr0_w, self.hr0_b)), self.hr2_w, self.hr2_b)
        out_u = F.linear(F.relu(F.linear(ro_in, self.hu0_w, self.hu0_b)), self.hu2_w, self.hu2_b)
        return torch.cat([out_r, out_u], dim=-1), z


def compute_loss(out: torch.Tensor, batch: dict, variant: str, pos_wt_prot: float = 3.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.sigmoid(out[:, :5])
    if variant == "single_24s":
        y24 = batch["y_risk_horizons"][:, 4]
        loss_r = F.binary_cross_entropy(torch.clamp(q[:, 4], 1e-7, 1.0 - 1e-7), y24)
        return loss_r, loss_r, out.new_zeros(())

    j_idx = torch.arange(5, device=out.device)[None, :]
    haz = batch["y_hazard_bin"][:, None]
    at_risk = (j_idx <= torch.clamp(haz, max=4)).float()
    hit_here = (j_idx == haz).float()
    loss_surv = (-at_risk * (hit_here * torch.log(q + 1e-7) + (1.0 - hit_here) * torch.log(1.0 - q + 1e-7))).sum() / out.shape[0]

    pr = torch.sigmoid(out[:, 5])
    yp = batch["y_risk_protected"]
    w = torch.where(yp > 0.5, pos_wt_prot, 1.0)
    loss_prot = (w * (-(yp * torch.log(pr + 1e-7) + (1.0 - yp) * torch.log(1.0 - pr + 1e-7)))).mean()
    loss_risk = loss_surv + loss_prot

    loss_prog = 0.25 * F.mse_loss(out[:, 6], batch["y_progress_protected"])
    loss_du = 0.50 * F.mse_loss(out[:, 7], batch["y_signed_delta_u"]) if variant == "full_v3_neurocover" else out.new_zeros(())
    loss_util = loss_prog + loss_du
    return loss_risk + loss_util, loss_risk, loss_util


def train_torch_model(model: nn.Module, tr: dict, variant: str, epochs: int = 80, lr: float = 5e-3, seed: int = 7, batch_size: int = 64):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    rng = np.random.default_rng(seed)
    n = len(tr["X_norm"])
    model.train()
    for _ in range(epochs):
        perm = rng.permutation(n)
        for st in range(0, n, batch_size):
            idx = perm[st : st + batch_size]
            sub = {k: v[idx] for k, v in tr.items()}
            opt.zero_grad()
            out, _ = model(sub["X_norm"], sub["action"])
            loss, _, _ = compute_loss(out, sub, variant)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()


def export_model_weights(model: nn.Module, key_p: str) -> dict:
    d = {f"{key_p}__core_type": np.array([model.core_type])}
    if hasattr(model, "adj"):
        d[f"{key_p}__adj"] = model.adj.detach().cpu().numpy().astype(np.float32)
        d[f"{key_p}__sign"] = model.sign.detach().cpu().numpy().astype(np.float32)
    if hasattr(model, "belief_skip_dim"):
        d[f"{key_p}__belief_skip_dim"] = np.array([model.belief_skip_dim], dtype=np.int64)
    for k, p in model.named_parameters():
        d[f"{key_p}__{k}"] = p.detach().cpu().numpy().astype(np.float32)
    return d


def count_trainable_params(model: nn.Module) -> int:
    total = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name == "theta" and hasattr(model, "adj"):
            total += int((model.adj != 0).sum().item())
        else:
            total += int(p.numel())
    return total


def main() -> None:
    clone_dir = Path("/kaggle/temp/vigil-repo") if Path("/kaggle").exists() else Path(".")
    if Path("/kaggle").exists() and not clone_dir.exists():
        clone_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(clone_dir)], check=True)
    commit = subprocess.check_output(["git", "-C", str(clone_dir), "rev-parse", "HEAD"], text=True).strip() if (clone_dir / ".git").exists() else "local"

    data_dir = clone_dir / "runs/pilot-v3-p69"
    bio_dir = clone_dir / "runs/graph-controls/escape"
    ann_csv = clone_dir / "runs/connectome/escape/annotations.csv"
    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("runs/p69-kaggle-output")
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device={dev} | repo_commit={commit}")

    def _load(sp: str):
        with np.load(data_dir / f"{sp}.npz") as d:
            return {
                "X": d["X"].astype(np.float32),
                "action": d["action"].astype(np.int64),
                "y_hazard_bin": d["y_hazard_bin"].astype(np.int64),
                "y_risk_horizons": d["y_risk_horizons"].astype(np.float32),
                "y_risk_protected": d["y_risk_protected"].astype(np.float32),
                "y_progress_protected": d["y_progress_protected"].astype(np.float32),
                "y_signed_delta_u": d["y_signed_delta_u"].astype(np.float32),
            }

    tr_np, va_np, te_np = _load("train"), _load("val"), _load("test")
    mu = tr_np["X"].mean(axis=(0, 1), keepdims=True)
    sd = np.maximum(tr_np["X"].std(axis=(0, 1), keepdims=True), 1e-4)
    for d in (tr_np, va_np, te_np):
        d["X_norm"] = ((d["X"] - mu) / sd).astype(np.float32)

    def _to_torch(d: dict) -> dict:
        return {
            "X_norm": torch.tensor(d["X_norm"], device=dev),
            "action": torch.tensor(d["action"], dtype=torch.long, device=dev),
            "y_hazard_bin": torch.tensor(d["y_hazard_bin"], dtype=torch.long, device=dev),
            "y_risk_horizons": torch.tensor(d["y_risk_horizons"], device=dev),
            "y_risk_protected": torch.tensor(d["y_risk_protected"], device=dev),
            "y_progress_protected": torch.tensor(d["y_progress_protected"], device=dev),
            "y_signed_delta_u": torch.tensor(d["y_signed_delta_u"], device=dev),
        }

    tr_t, va_t, te_t = _to_torch(tr_np), _to_torch(va_np), _to_torch(te_np)
    in_dim = tr_np["X"].shape[-1]
    pre_signs = _load_nt_signs(ann_csv, bio_dir / "nodes.txt")

    def _masks(p: Path):
        mat = load_npz(p).toarray()
        return (mat != 0).astype(np.float32), np.broadcast_to(pre_signs[None, :], mat.shape).copy()

    adj_bio, sgn_bio = _masks(bio_dir / "biological.npz")
    adj_r7, sgn_r7 = _masks(bio_dir / "rewired-7.npz")
    adj_r19, sgn_r19 = _masks(bio_dir / "rewired-19.npz")
    adj_r41, sgn_r41 = _masks(bio_dir / "rewired-41.npz")
    n_bio, edge_count = adj_bio.shape[0], int((adj_bio != 0).sum())

    def _rand_sparse(s: int):
        rng_r = np.random.default_rng(1000 + s)
        ar = np.zeros_like(adj_bio)
        ar.ravel()[rng_r.choice(n_bio * n_bio, size=edge_count, replace=False)] = 1.0
        sr = rng_r.choice([-1.0, 1.0], size=(n_bio, n_bio)).astype(np.float32)
        return ar, sr

    seeds = (7, 17, 27)
    epochs = 80
    p_const_prot = float(np.mean(tr_np["y_risk_protected"] > 0.5))
    p_const_03 = float(np.mean(tr_np["y_risk_horizons"][:, 1] > 0.5))
    p_const_24 = float(np.mean(tr_np["y_risk_horizons"][:, 4] > 0.5))

    # --- PART 1: P6.9.5 Architecture & Gradient Diagnostics ---
    diag_arches = [
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
    diag_summary = {"repo_commit": commit, "device": str(dev), "epochs": epochs, "seeds": list(seeds), "bio_nodes": n_bio, "bio_edges": edge_count, "models": {}}

    for arch in diag_arches:
        per_seed = []
        p_cnt = 0
        for s in seeds:
            if arch == "Bio_Joint_P6":
                mod = JointBioNet(adj_bio, sgn_bio, in_dim, "bio", s).to(dev)
            elif arch == "Bio_Frozen_Modular":
                mod = ModularBioNet(adj_bio, sgn_bio, in_dim, freeze_core=True, seed=s).to(dev)
            elif arch == "Bio_Adaptive_Modular":
                mod = ModularBioNet(adj_bio, sgn_bio, in_dim, freeze_core=False, seed=s).to(dev)
            elif arch == "Rewired7_Adaptive_Mod":
                mod = ModularBioNet(adj_r7, sgn_r7, in_dim, freeze_core=False, seed=s).to(dev)
            elif arch == "Rewired19_Adaptive_Mod":
                mod = ModularBioNet(adj_r19, sgn_r19, in_dim, freeze_core=False, seed=s).to(dev)
            elif arch == "Rewired41_Adaptive_Mod":
                mod = ModularBioNet(adj_r41, sgn_r41, in_dim, freeze_core=False, seed=s).to(dev)
            elif arch == "RandomSparse_Adaptive_Mod":
                ar, sr = _rand_sparse(s)
                mod = ModularBioNet(ar, sr, in_dim, freeze_core=False, seed=s).to(dev)
            elif arch == "GRU_Joint_P6":
                mod = JointGRUNet(in_dim, 64, s).to(dev)
            elif arch == "GRU_Modular":
                mod = ModularGRUNet(in_dim, 64, 24, s).to(dev)

            train_torch_model(mod, tr_t, "full_v3_neurocover", epochs=epochs, lr=5e-3, seed=s, batch_size=64)
            p_cnt = count_trainable_params(mod)
            mod.eval()
            with torch.no_grad():
                te_out, z_te = mod(te_t["X_norm"], te_t["action"])
                te_np_out = te_out.cpu().numpy()
                z_np = z_te.cpu().numpy()

            rc = _compute_cum_risks(te_np_out)
            pr = 1.0 / (1.0 + np.exp(-np.clip(te_np_out[:, 5], -30.0, 30.0)))
            m_prot = binary_metrics(te_np["y_risk_protected"], pr, p_const_prot)
            m_24 = binary_metrics(te_np["y_risk_horizons"][:, 4], rc[:, 4], p_const_24)

            # Gradient cosine between risk and utility in shared recurrent encoder
            cos_ru = 0.0
            shared_p = getattr(mod, "enc_w", None) if hasattr(mod, "enc_w") else getattr(mod, "x_w", None)
            if shared_p is not None and shared_p.requires_grad:
                sub = {k: v[:64] for k, v in tr_t.items()}
                mod.zero_grad()
                o_s, _ = mod(sub["X_norm"], sub["action"])
                _, l_r, l_u = compute_loss(o_s, sub, "full_v3_neurocover")
                g_r = torch.autograd.grad(l_r, shared_p, retain_graph=True)[0].detach().cpu().numpy().ravel()
                g_u = torch.autograd.grad(l_u, shared_p)[0].detach().cpu().numpy().ravel()
                cos_ru = _cosine(g_r, g_u)

            per_seed.append({
                "protected_brier": m_prot["brier"],
                "protected_bss": m_prot["brier_skill_score"],
                "protected_pr_auc": m_prot["pr_auc"],
                "protected_fn_lt_0_20": m_prot["dangerous_fn_lt_0_20"],
                "risk_24s_brier": m_24["brier"],
                "risk_24s_bss": m_24["brier_skill_score"],
                "risk_24s_pr_auc": m_24["pr_auc"],
                "progress_mae": float(np.mean(np.abs(te_np_out[:, 6] - te_np["y_progress_protected"]))),
                "delta_u_mae": float(np.mean(np.abs(te_np_out[:, 7] - te_np["y_signed_delta_u"]))),
                "hidden_saturation_frac": float(np.mean(np.abs(z_np) > 0.90)),
                "hidden_effective_rank": _effective_rank(z_np),
                "grad_cos_risk_vs_util": cos_ru,
            })

        agg = {"trainable_params": p_cnt}
        for mk in ("protected_brier", "protected_bss", "protected_pr_auc", "risk_24s_brier", "risk_24s_bss", "risk_24s_pr_auc", "progress_mae", "delta_u_mae", "hidden_saturation_frac", "hidden_effective_rank", "grad_cos_risk_vs_util"):
            vals = [float(r[mk]) for r in per_seed if r[mk] is not None]
            agg[mk + "_mean"] = float(np.mean(vals))
            agg[mk + "_std"] = float(np.std(vals))
        agg["per_seed"] = per_seed
        diag_summary["models"][arch] = agg
        print(f"[P6.9.5] {arch:26s} | prot_BSS={agg['protected_bss_mean']:+.3f} PRAUC={agg['protected_pr_auc_mean']:.3f} | duMAE={agg['delta_u_mae_mean']:.4f} | cos={agg['grad_cos_risk_vs_util_mean']:+.3f}")

    (out_dir / "bio_diagnostics.json").write_text(json.dumps(diag_summary, indent=2))

    # --- PART 2: P6.9.6 Matched Pilot Ensemble Weights ---
    eval_specs = [
        ("Single24s_Bio", "single_24s"),
        ("MultiProt_NoRecovTrain_Bio", "multi_horizon_prot"),
        ("Full_Bio_Joint", "full_v3_neurocover"),
        ("Full_Bio_Modular", "full_v3_neurocover"),
        ("Full_Rewired7_Modular", "full_v3_neurocover"),
        ("Full_Rewired19_Modular", "full_v3_neurocover"),
        ("Full_Rewired41_Modular", "full_v3_neurocover"),
        ("Full_GRU_Modular", "full_v3_neurocover"),
    ]
    exported = {"feat_mean": mu.squeeze().astype(np.float32), "feat_std": sd.squeeze().astype(np.float32)}
    off_report = {"repo_commit": commit, "device": str(dev), "epochs": epochs, "seeds": list(seeds), "models": {}}

    for arch, variant in eval_specs:
        seed_metrics = {"val": [], "test": []}
        for s in seeds:
            if arch in ("Single24s_Bio", "Full_Bio_Joint"):
                mod = JointBioNet(adj_bio, sgn_bio, in_dim, "bio", s).to(dev)
            elif arch in ("MultiProt_NoRecovTrain_Bio", "Full_Bio_Modular"):
                mod = ModularBioNet(adj_bio, sgn_bio, in_dim, freeze_core=False, seed=s).to(dev)
            elif arch == "Full_Rewired7_Modular":
                mod = ModularBioNet(adj_r7, sgn_r7, in_dim, freeze_core=False, seed=s).to(dev)
            elif arch == "Full_Rewired19_Modular":
                mod = ModularBioNet(adj_r19, sgn_r19, in_dim, freeze_core=False, seed=s).to(dev)
            elif arch == "Full_Rewired41_Modular":
                mod = ModularBioNet(adj_r41, sgn_r41, in_dim, freeze_core=False, seed=s).to(dev)
            elif arch == "Full_GRU_Modular":
                mod = ModularGRUNet(in_dim, 64, 24, s).to(dev)

            train_torch_model(mod, tr_t, variant, epochs=epochs, lr=5e-3, seed=s, batch_size=64)
            exported.update(export_model_weights(mod, f"{arch}__s{s}"))
            mod.eval()
            for sp_name, ds_t, ds_np in (("val", va_t, va_np), ("test", te_t, te_np)):
                with torch.no_grad():
                    out_sp = mod(ds_t["X_norm"], ds_t["action"])[0].cpu().numpy()
                if variant == "single_24s":
                    r24 = 1.0 / (1.0 + np.exp(-np.clip(out_sp[:, 4], -30.0, 30.0)))
                    seed_metrics[sp_name].append({
                        "risk_24s": binary_metrics(ds_np["y_risk_horizons"][:, 4], r24, p_const_24),
                        "cross_protected": binary_metrics(ds_np["y_risk_protected"], r24, p_const_prot),
                    })
                else:
                    rc = _compute_cum_risks(out_sp)
                    pr = 1.0 / (1.0 + np.exp(-np.clip(out_sp[:, 5], -30.0, 30.0)))
                    seed_metrics[sp_name].append({
                        "protected_0_30s": binary_metrics(ds_np["y_risk_protected"], pr, p_const_prot),
                        "cum_0_30s": binary_metrics(ds_np["y_risk_horizons"][:, 1], rc[:, 1], p_const_03),
                        "cum_2_40s": binary_metrics(ds_np["y_risk_horizons"][:, 4], rc[:, 4], p_const_24),
                        "progress_mae": float(np.mean(np.abs(out_sp[:, 6] - ds_np["y_progress_protected"]))),
                        "delta_u_mae": float(np.mean(np.abs(out_sp[:, 7] - ds_np["y_signed_delta_u"]))),
                        "survival_monotonic": bool(np.all(np.diff(rc, axis=1) >= -1e-9)),
                    })

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
        off_report["models"][arch] = agg

    np.savez_compressed(out_dir / "p69_weights.npz", **exported)
    (out_dir / "p69_offline_metrics.json").write_text(json.dumps(off_report, indent=2))
    print("Saved bio_diagnostics.json, p69_weights.npz, p69_offline_metrics.json to", out_dir)


if __name__ == "__main__":
    main()
