from __future__ import annotations

"""G07 Kaggle GPU trainer: M3 GRU, M4 Biological Masked, M5 Rewired, and M8 Neuro-COVER-L."""
import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
    _ModuleBase = nn.Module
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    _TORCH_AVAILABLE = False
    _ModuleBase = object

REPO_URL = "https://github.com/Shubham142856/vigil-rover-x-neuroai.git"


class MaskedCircuitCore(_ModuleBase):
    """Differentiable rate core: W_eff = M * S * softplus(theta)."""

    def __init__(self, adj_mask: torch.Tensor, sign_mask: torch.Tensor, in_dim: int):
        super().__init__()
        self.register_buffer("adj_mask", adj_mask)
        self.register_buffer("sign_mask", sign_mask)
        n = adj_mask.shape[0]
        self.theta = nn.Parameter(torch.full((n, n), -1.0))
        self.enc = nn.Linear(in_dim, n)

    def forward(self, x_seq):
        w_eff = self.adj_mask * self.sign_mask * torch.nn.functional.softplus(self.theta)
        state = torch.zeros(x_seq.shape[0], self.adj_mask.shape[0], device=x_seq.device)
        for t in range(x_seq.shape[1]):
            drive = self.enc(x_seq[:, t, :])
            state = state + 0.2 * (torch.tanh(drive + state @ w_eff.T) - state)
        return state


class ActionHeadPolicy(_ModuleBase):
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


def calibrate_bounded_fp_threshold(probs, y_true, max_fpr=0.02):
    """Freeze lowest validation threshold keeping validation FPR <= max_fpr and maximizing TPR."""
    order = np.argsort(probs)
    best_t, best_score = 0.5, (-1.0, -1.0, -1.0)
    n_neg = max(1, int(np.sum(y_true == 0)))
    n_pos = max(1, int(np.sum(y_true == 1)))
    for t in np.unique(probs[order]):
        pred = probs >= t
        fp = int(np.sum(pred & (y_true == 0)))
        tp = int(np.sum(pred & (y_true == 1)))
        fpr = fp / n_neg
        tpr = tp / n_pos
        within_budget = 1.0 if fpr <= max_fpr else (1.0 - fpr)
        cand = (within_budget, tpr - fpr, tpr, -float(t))
        if cand > best_score:
            best_score = cand
            best_t = float(t)
    return best_t


def _binary_f1(pred: np.ndarray, y_true: np.ndarray) -> float:
    tp = int(np.sum(pred & (y_true == 1)))
    fp = int(np.sum(pred & (y_true == 0)))
    fn = int(np.sum((~pred) & (y_true == 1)))
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0


def resolve_output_dir(explicit_out: Path | None) -> Path:
    """Resolve output directory: explicit path, /kaggle/working, /content/working, or local fallback."""
    if explicit_out is not None:
        p = Path(explicit_out)
    elif Path("/kaggle/working").exists():
        p = Path("/kaggle/working")
    elif Path("/content").exists():
        p = Path("/content/working")
    else:
        p = Path("runs/kaggle-training-output")
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_inputs(data_dir, bio_path, rewired_path, annotations_path):
    """Resolve inputs from CLI args, local repo, /kaggle/input, /content, or pinned public GitHub clone."""
    if all(p is not None and Path(p).exists() for p in (data_dir, bio_path, rewired_path, annotations_path)):
        return Path(data_dir), Path(bio_path), Path(rewired_path), Path(annotations_path), "explicit"

    candidates = [Path("."), Path("/kaggle/input"), Path("/content")]
    for root in candidates:
        for ds_name in ("scaled-dataset-combined", "scaled-dataset"):
            if (root / f"runs/{ds_name}/train.npz").exists():
                return (
                    root / f"runs/{ds_name}",
                    root / "runs/graph-controls/escape/biological.npz",
                    root / "runs/graph-controls/escape/rewired-7.npz",
                    root / "runs/connectome/escape/annotations.csv",
                    str(root),
                )
        if root.exists():
            found = list(root.rglob("train.npz"))
            if found:
                d = found[0].parent
                b_list = list(root.rglob("biological.npz"))
                r_list = list(root.rglob("rewired-7.npz"))
                a_list = list(root.rglob("annotations.csv"))
                if b_list and r_list and a_list:
                    return d, b_list[0], r_list[0], a_list[0], str(root)

    clone_dir = Path("/kaggle/temp/vigil-repo") if Path("/kaggle").exists() else Path("/tmp/vigil-repo")
    if not clone_dir.exists():
        clone_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(clone_dir)], check=True)
    commit = subprocess.check_output(["git", "-C", str(clone_dir), "rev-parse", "HEAD"], text=True).strip()
    ds_dir = clone_dir / "runs/scaled-dataset-combined"
    if not (ds_dir / "train.npz").exists():
        ds_dir = clone_dir / "runs/scaled-dataset"
    return (
        ds_dir,
        clone_dir / "runs/graph-controls/escape/biological.npz",
        clone_dir / "runs/graph-controls/escape/rewired-7.npz",
        clone_dir / "runs/connectome/escape/annotations.csv",
        f"github:{commit}",
    )


def load_nt_signs(annotations_path: Path, nodes_path: Path):
    """Map FAFB predicted_nt to explicit postsynaptic sign mask (ACH=+1, GLUT=-1, UNKNOWN=0)."""
    ids = nodes_path.read_text().splitlines()
    with annotations_path.open() as f:
        nt_map = {r["root_id"]: r["predicted_nt"] for r in csv.DictReader(f)}
    pre_signs = np.zeros(len(ids), dtype=np.float32)
    for idx, rid in enumerate(ids):
        nt = nt_map.get(rid, "")
        if nt == "ACH":
            pre_signs[idx] = 1.0
        elif nt == "GLUT":
            pre_signs[idx] = -1.0
    return pre_signs


def export_portable_npz(checkpoints, thresholds, out_path: Path):
    """Save NumPy-only weights and thresholds so CPU rover controllers load without PyTorch."""
    flat = {}
    for model_name, sd in checkpoints.items():
        for k, tensor in sd.items():
            flat[f"{model_name}__{k}"] = tensor.numpy()
        flat[f"{model_name}__threshold"] = np.array([thresholds[model_name]], dtype=np.float32)
    np.savez_compressed(out_path, **flat)


def train_and_eval(data_dir: Path, bio_npz: Path, rewired_npz: Path, annotations_csv: Path, out_dir: Path, epochs=35, device="cuda"):
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed in this environment. Please run training on Kaggle GPU or Google Colab.")
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
    nodes_path = bio_npz.parent / "nodes.txt"
    pre_signs = load_nt_signs(annotations_csv, nodes_path)

    def make_masks(npz_path):
        mat = load_npz(npz_path).toarray()
        adj = torch.tensor((mat != 0).astype(np.float32), device=dev)
        # Apply presynaptic NT sign across columns; unknown-NT edges stay 0 in strict signed mask
        sign = torch.tensor(np.broadcast_to(pre_signs[None, :], mat.shape).copy(), device=dev)
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
    pos_count = float(tr[2].sum())
    neg_count = float(len(tr[2]) - pos_count)
    # Cap pos_weight at 5.0 to prevent extreme probability saturation on noisy validation frames
    pos_weight = torch.tensor([min(5.0, max(1.0, neg_count / max(pos_count, 1.0)))], device=dev)
    bce_contact = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bce_recov = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    metrics = {}
    checkpoints = {}
    thresholds = {}
    for name, model in configs.items():
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        for _ in range(epochs):
            model.train()
            for xb, ab, yc, yp, yr in loader:
                xb, ab, yc, yp, yr = xb.to(dev), ab.to(dev), yc.to(dev), yp.to(dev), yr.to(dev)
                opt.zero_grad()
                lc, pr, lr_ = model(xb, ab)
                loss = bce_contact(lc, yc) + 0.5 * mse(pr, yp) + (0.5 * bce_recov(lr_, yr) if name == "M8_NeuroCOVER_L" else 0.0)
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            vlc, _, _ = model(va[0].to(dev), va[1].to(dev))
            vprob = torch.sigmoid(vlc).cpu().numpy()
            thresh = calibrate_bounded_fp_threshold(vprob, va[2].numpy(), max_fpr=0.02)
            zero_fp_thresh = calibrate_bounded_fp_threshold(vprob, va[2].numpy(), max_fpr=0.0)
            thresholds[name] = thresh

            tlc, tpr_, tlr = model(te[0].to(dev), te[1].to(dev))
            tprob = torch.sigmoid(tlc).cpu().numpy()
            ty = te[2].numpy()
            pred = tprob >= thresh
            metrics[name] = {
                "frozen_val_threshold": thresh,
                "strict_zero_fp_val_threshold": zero_fp_thresh,
                "test_brier": float(np.mean((tprob - ty) ** 2)),
                "test_false_positive_rate": float(np.mean(pred[ty == 0])) if np.any(ty == 0) else 0.0,
                "test_recall": float(np.mean(pred[ty == 1])) if np.any(ty == 1) else 0.0,
                "test_f1": _binary_f1(pred, ty),
                "test_progress_mae": float(np.mean(np.abs(tpr_.cpu().numpy() - te[3].numpy()))),
                "test_recover_bce": float(bce_recov(tlr.cpu(), te[4]).item()),
            }
        checkpoints[name] = {k: v.cpu() for k, v in model.state_dict().items()}

    torch.save(checkpoints, out_dir / "checkpoints.pt")
    export_portable_npz(checkpoints, thresholds, out_dir / "portable_weights.npz")
    report = {
        "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU",
        "epochs": epochs,
        "unknown_sign_nodes": int(np.sum(pre_signs == 0)),
        "models": metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--bio", type=Path, default=None)
    p.add_argument("--rewired", type=Path, default=None)
    p.add_argument("--annotations", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--device", default="cuda")
    # parse_known_args safely ignores extra flags like -f injected by Jupyter / Google Colab kernels
    a, _ = p.parse_known_args(argv)
    out_dir = resolve_output_dir(a.out)
    data_d, bio_p, rew_p, ann_p, source = resolve_inputs(a.data, a.bio, a.rewired, a.annotations)
    rep = train_and_eval(data_d, bio_p, rew_p, ann_p, out_dir, a.epochs, a.device)
    rep["source"] = source
    (out_dir / "metrics.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return rep


if __name__ == "__main__":
    main()
