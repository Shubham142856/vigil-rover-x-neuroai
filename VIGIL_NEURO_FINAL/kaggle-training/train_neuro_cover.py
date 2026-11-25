"""G07 Kaggle GPU trainer: M3 GRU, M4 Biological Masked, M5 Rewired, and M8 Neuro-COVER-L."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MaskedCircuitCore(nn.Module):
    """Differentiable rate core: W_eff = M * S * softplus(theta)."""

    def __init__(self, adj_mask: torch.Tensor, sign_mask: torch.Tensor, in_dim: int):
        super().__init__()
        self.register_buffer("adj_mask", adj_mask)
        self.register_buffer("sign_mask", sign_mask)
        n = adj_mask.shape[0]
        self.theta = nn.Parameter(torch.full((n, n), -1.0))
        self.enc = nn.Linear(in_dim, n)

    def forward(self, x_seq):
        # x_seq: (B, T, F)
        w_eff = self.adj_mask * self.sign_mask * torch.nn.functional.softplus(self.theta)
        state = torch.zeros(x_seq.shape[0], self.adj_mask.shape[0], device=x_seq.device)
        for t in range(x_seq.shape[1]):
            drive = self.enc(x_seq[:, t, :])
            state = state + 0.2 * (torch.tanh(drive + state @ w_eff.T) - state)
        return state


class ActionHeadPolicy(nn.Module):
    """Shared action-conditioned risk, progress, and recoverability heads."""

    def __init__(self, core_type, core_dim, in_dim, n_actions=5, adj_mask=None, sign_mask=None):
        super().__init__()
        self.core_type = core_type
        if core_type == "gru":
            self.core = nn.GRU(in_dim, core_dim, batch_first=True)
            feat_dim = core_dim
        else:
            self.core = MaskedCircuitCore(adj_mask, sign_mask, in_dim)
            feat_dim = adj_mask.shape[0]
        self.act_emb = nn.Embedding(n_actions, 16)
        self.head = nn.Sequential(
            nn.Linear(feat_dim + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # [contact_logit, progress, recover_logit]
        )

    def forward(self, x_seq, action):
        if self.core_type == "gru":
            _, h = self.core(x_seq)
            z = h.squeeze(0)
        else:
            z = self.core(x_seq)
        out = self.head(torch.cat([z, self.act_emb(action)], dim=-1))
        return out[:, 0], out[:, 1], out[:, 2]


def calibrate_zero_fp_threshold(probs, y_true):
    """Freeze lowest validation threshold yielding zero false-safe misses (high recall, bounded FP)."""
    order = np.argsort(probs)
    best_t, best_score = 0.5, (-1.0, -1.0)
    for t in np.unique(probs[order]):
        pred = probs >= t
        fp = int(np.sum(pred & (y_true == 0)))
        tp = int(np.sum(pred & (y_true == 1)))
        tpr = tp / max(1, int(np.sum(y_true == 1)))
        tnr = 1.0 - fp / max(1, int(np.sum(y_true == 0)))
        # Prioritize zero false positives on safe actions while maximizing collision recall
        cand = (1.0 if fp == 0 else tnr, tpr)
        if cand > best_score:
            best_score = cand
            best_t = float(t)
    return best_t


def train_and_eval(data_dir: Path, bio_npz: Path, rewired_npz: Path, out_dir: Path, epochs=25, device="cuda"):
    from scipy.sparse import load_npz

    torch.manual_seed(7)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    def load_split(name):
        with np.load(data_dir / f"{name}.npz") as d:
            return (
                torch.tensor(d["X"], dtype=torch.float32),
                torch.tensor(d["action"], dtype=torch.long),
                torch.tensor(d["y_contact"], dtype=torch.float32),
                torch.tensor(d["y_progress"], dtype=torch.float32),
                torch.tensor(d["y_recover"], dtype=torch.float32),
            )

    tr = load_split("train")
    va = load_split("val")
    te = load_split("test")

    def make_masks(npz_path):
        mat = load_npz(npz_path).toarray()
        adj = torch.tensor((mat != 0).astype(np.float32), device=dev)
        sign = torch.ones_like(adj)
        return adj, sign

    adj_bio, sgn_bio = make_masks(bio_npz)
    adj_rew, sgn_rew = make_masks(rewired_npz)
    in_dim = tr[0].shape[-1]

    configs = {
        "M3_GRU": ActionHeadPolicy("gru", 64, in_dim).to(dev),
        "M4_BioMasked": ActionHeadPolicy("bio", 0, in_dim, adj_mask=adj_bio, sign_mask=sgn_bio).to(dev),
        "M5_Rewired": ActionHeadPolicy("rewired", 0, in_dim, adj_mask=adj_rew, sign_mask=sgn_rew).to(dev),
        "M8_NeuroCOVER_L": ActionHeadPolicy("bio", 0, in_dim, adj_mask=adj_bio, sign_mask=sgn_bio).to(dev),
    }

    loader = DataLoader(TensorDataset(*tr), batch_size=32, shuffle=True)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    metrics = {}
    checkpoints = {}
    for name, model in configs.items():
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        for _ in range(epochs):
            model.train()
            for xb, ab, yc, yp, yr in loader:
                xb, ab, yc, yp, yr = xb.to(dev), ab.to(dev), yc.to(dev), yp.to(dev), yr.to(dev)
                opt.zero_grad()
                lc, pr, lr_ = model(xb, ab)
                loss = bce(lc, yc) + 0.5 * mse(pr, yp) + (0.5 * bce(lr_, yr) if name == "M8_NeuroCOVER_L" else 0.0)
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            vlc, _, _ = model(va[0].to(dev), va[1].to(dev))
            vprob = torch.sigmoid(vlc).cpu().numpy()
            thresh = calibrate_zero_fp_threshold(vprob, va[2].numpy())

            tlc, tpr_, tlr = model(te[0].to(dev), te[1].to(dev))
            tprob = torch.sigmoid(tlc).cpu().numpy()
            ty = te[2].numpy()
            pred = tprob >= thresh
            metrics[name] = {
                "frozen_val_threshold": thresh,
                "test_brier": float(np.mean((tprob - ty) ** 2)),
                "test_false_positive_rate": float(np.mean(pred[ty == 0])) if np.any(ty == 0) else 0.0,
                "test_recall": float(np.mean(pred[ty == 1])) if np.any(ty == 1) else 0.0,
                "test_progress_mae": float(np.mean(np.abs(tpr_.cpu().numpy() - te[3].numpy()))),
            }
        checkpoints[name] = {k: v.cpu() for k, v in model.state_dict().items()}

    torch.save(checkpoints, out_dir / "checkpoints.pt")
    report = {
        "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU",
        "epochs": epochs,
        "models": metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--bio", type=Path, required=True)
    p.add_argument("--rewired", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("/kaggle/working"))
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    print(json.dumps(train_and_eval(a.data, a.bio, a.rewired, a.out, a.epochs, a.device), indent=2))
