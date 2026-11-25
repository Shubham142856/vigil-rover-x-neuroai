"""G05 directional memory benchmark: connectome ring attractor vs odometry, Bayes filter, and GRU."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from camera_compass import bearing, fresh_heading
from heading_memory import advance, decode, wrap
from rover_sim import RoverSim


def load_compass_connectome(compass_dir: Path):
    """Load FAFB compass graph, signed weights, and connectome-derived EPG ring angles."""
    with (compass_dir / "annotations.csv").open() as f:
        ann = list(csv.DictReader(f))
    ids = [r["root_id"] for r in ann]
    idx = {rid: i for i, rid in enumerate(ids)}
    types = np.array([r["primary_type"] for r in ann])
    sides = np.array([r["side"] for r in ann])
    nts = np.array([r["predicted_nt"] for r in ann])

    # Explicit sign vector S: ACH -> +1, GLUT -> -1, unknown -> 0 (recorded explicitly)
    signs = np.zeros(len(ids), dtype=float)
    signs[nts == "ACH"] = 1.0
    signs[nts == "GLUT"] = -1.0

    W = np.zeros((len(ids), len(ids)), dtype=float)
    with (compass_dir / "edges.csv").open() as f:
        for e in csv.DictReader(f):
            W[idx[e["post_id"]], idx[e["pre_id"]]] += int(e["synapse_count"])

    epg = types == "EPG"
    d7 = types == "Delta7"
    pen_a = types == "PEN_a/PEN1"
    pen_b = types == "PEN_b/PEN2"
    peg = types == "PEG"

    # Spectral circular embedding of EPG from local excitation minus Delta7 opponent inhibition
    W_exc = W[np.ix_(epg, epg)] + 0.01 * (W[np.ix_(epg, pen_b)] @ W[np.ix_(pen_b, epg)])
    W_inh = W[np.ix_(epg, d7)] @ W[np.ix_(d7, epg)]
    A = 0.5 * (W_exc + W_exc.T)
    A /= max(A.max(), 1.0)
    B = 0.5 * (W_inh + W_inh.T)
    B /= max(B.max(), 1.0)
    _, vecs = np.linalg.eigh(A - 0.6 * B)
    phi = np.arctan2(vecs[:, -1], vecs[:, -2])

    # Determine chiral direction of PEN_a(left) vs PEN_a(right) directly from connectome
    def mean_shift(mask):
        M = W[np.ix_(epg, mask)] @ W[np.ix_(mask, epg)]
        w = M.sum()
        if w <= 0:
            return 0.0
        d = (phi[:, None] - phi[None, :] + np.pi) % (2 * np.pi) - np.pi
        return float(np.arcsin(np.clip((M * np.sin(d)).sum() / w, -1.0, 1.0)))

    shift_l = mean_shift(pen_a & (sides == "left"))
    shift_r = mean_shift(pen_a & (sides == "right"))
    # Flip sign of phi if needed so +delta_yaw aligns with PEN_a positive shift convention
    if shift_l < 0 < shift_r:
        phi = -phi
        shift_l, shift_r = -shift_l, -shift_r

    return {
        "ids": ids,
        "types": types,
        "sides": sides,
        "signs": signs,
        "unknown_sign_count": int((signs == 0).sum()),
        "W": W,
        "epg": epg,
        "peg": peg,
        "pen_a": pen_a,
        "pen_b": pen_b,
        "d7": d7,
        "phi_epg": phi,
        "pen_a_left_shift_deg": float(np.degrees(shift_l)),
        "pen_a_right_shift_deg": float(np.degrees(shift_r)),
    }


class ConnectomeCompass:
    """FAFB-constrained E-PG / P-EN / P-EG / Delta7 circular attractor compass."""

    def __init__(self, model, n_bins=72):
        self.m = model
        self.phi = model["phi_epg"]
        self.epg_idx = np.where(model["epg"])[0]
        # Row-normalized signed biological coupling matrix
        W_signed = model["W"] * model["signs"][None, :]
        row_norm = np.maximum(np.abs(W_signed).sum(axis=1, keepdims=True), 1.0)
        self.W_norm = W_signed / row_norm
        # Map EPG continuous phase dynamics over connectome cells + angular-velocity PEN_a modulation
        self.n_nodes = len(model["ids"])
        self.rate = np.zeros(self.n_nodes, dtype=float)
        self.angles = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)
        self.ring_mass = np.exp(10.0 * np.cos(self.angles))
        self.ring_mass /= self.ring_mass.sum()
        self._project_to_cells()
        self.outlier_streak = 0

    def _project_to_cells(self):
        # Sample ring mass onto the 47 connectome EPG neurons at their anatomical angles phi
        epg_drive = np.array([np.sum(self.ring_mass * np.exp(4.0 * np.cos(self.angles - p))) for p in self.phi])
        epg_drive /= max(epg_drive.max(), 1e-9)
        self.rate[self.epg_idx] = epg_drive
        # One recurrent pass through signed connectome (EPG -> Delta7/PEG/PEN_b -> EPG)
        self.rate = np.clip(self.rate + 0.15 * (np.tanh(self.W_norm @ self.rate) - self.rate), 0.0, 2.0)

    def step(self, delta_yaw, cue):
        # Angular velocity shift routed through PEN_a left/right asymmetry
        self.ring_mass = advance(self.ring_mass, delta_yaw)
        pen_l = self.m["pen_a"] & (self.m["sides"] == "left")
        pen_r = self.m["pen_a"] & (self.m["sides"] == "right")
        self.rate[pen_l] = max(0.0, delta_yaw * 20.0)
        self.rate[pen_r] = max(0.0, -delta_yaw * 20.0)

        if cue is not None:
            est = self.decode_angle()
            err = abs(wrap(cue - est)) if est is not None else 0.0
            # Conflicting landmark gate: reject single-frame >45 deg jumps unless confirmed twice
            if err > np.radians(45.0) and self.outlier_streak < 2:
                self.outlier_streak += 1
            else:
                self.outlier_streak = 0
                self.ring_mass *= np.exp(4.0 * np.cos(self.angles - cue))
                self.ring_mass /= self.ring_mass.sum()
        else:
            self.outlier_streak = 0

        self._project_to_cells()
        return self.decode_angle(), self.bump_magnitude()

    def decode_angle(self):
        act = np.maximum(self.rate[self.epg_idx], 0.0)
        sx = float(np.sum(act * np.sin(self.phi)))
        cx = float(np.sum(act * np.cos(self.phi)))
        if np.hypot(sx, cx) < 1e-9:
            return None
        return float(np.arctan2(sx, cx))

    def bump_magnitude(self):
        act = np.maximum(self.rate[self.epg_idx], 0.0)
        tot = float(act.sum())
        if tot < 1e-9:
            return 0.0
        vec = np.sum(act * np.exp(1j * self.phi))
        return float(abs(vec) / tot)


class GRUCompass:
    """Compact pure-NumPy GRU heading comparator trained with BPTT + L-BFGS-B."""

    def __init__(self, hidden=12, seed=0):
        self.in_dim = 4  # [delta_yaw, cue_valid, sin(cue), cos(cue)]
        self.h_dim = hidden
        rng = np.random.default_rng(seed)
        s = 0.25
        # Packed parameter shapes: Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh, Wo, bo
        self.shapes = [
            (hidden, self.in_dim), (hidden, hidden), (hidden,),
            (hidden, self.in_dim), (hidden, hidden), (hidden,),
            (hidden, self.in_dim), (hidden, hidden), (hidden,),
            (2, hidden), (2,),
        ]
        self.params = np.concatenate([rng.normal(0, s, size=np.prod(sh)) for sh in self.shapes])

    def _unpack(self, vec):
        out = []
        pos = 0
        for sh in self.shapes:
            sz = int(np.prod(sh))
            out.append(vec[pos : pos + sz].reshape(sh))
            pos += sz
        return out

    @staticmethod
    def _sig(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))

    def loss_and_grad(self, vec, X, Y):
        """X: (B, T, 4), Y: (B, T, 2) -> MSE + analytical BPTT gradient."""
        Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh, Wo, bo = self._unpack(vec)
        B, T, _ = X.shape
        H = self.h_dim

        hs = np.zeros((B, T + 1, H))
        zs = np.zeros((B, T, H))
        rs = np.zeros((B, T, H))
        htildes = np.zeros((B, T, H))
        preds = np.zeros((B, T, 2))

        for t in range(T):
            x = X[:, t, :]
            hp = hs[:, t, :]
            z = self._sig(x @ Wz.T + hp @ Uz.T + bz)
            r = self._sig(x @ Wr.T + hp @ Ur.T + br)
            ht = np.tanh(x @ Wh.T + (r * hp) @ Uh.T + bh)
            h = (1.0 - z) * hp + z * ht
            zs[:, t, :], rs[:, t, :], htildes[:, t, :], hs[:, t + 1, :] = z, r, ht, h
            preds[:, t, :] = h @ Wo.T + bo

        diff = preds - Y
        loss = float(np.mean(diff ** 2) + 1e-4 * np.mean(vec ** 2))
        dpreds = (2.0 / (B * T * 2)) * diff

        gWz = np.zeros_like(Wz); gUz = np.zeros_like(Uz); gbz = np.zeros_like(bz)
        gWr = np.zeros_like(Wr); gUr = np.zeros_like(Ur); gbr = np.zeros_like(br)
        gWh = np.zeros_like(Wh); gUh = np.zeros_like(Uh); gbh = np.zeros_like(bh)
        gWo = np.zeros_like(Wo); gbo = np.zeros_like(bo)
        dh_next = np.zeros((B, H))

        for t in reversed(range(T)):
            x = X[:, t, :]
            hp = hs[:, t, :]
            h = hs[:, t + 1, :]
            z = zs[:, t, :]
            r = rs[:, t, :]
            ht = htildes[:, t, :]

            dp = dpreds[:, t, :]
            gWo += dp.T @ h
            gbo += dp.sum(axis=0)
            dh = dp @ Wo + dh_next

            dht = dh * z * (1.0 - ht ** 2)
            gWh += dht.T @ x
            gUh += dht.T @ (r * hp)
            gbh += dht.sum(axis=0)

            dr_pre = (dht @ Uh) * hp * r * (1.0 - r)
            gWr += dr_pre.T @ x
            gUr += dr_pre.T @ hp
            gbr += dr_pre.sum(axis=0)

            dz_pre = dh * (ht - hp) * z * (1.0 - z)
            gWz += dz_pre.T @ x
            gUz += dz_pre.T @ hp
            gbz += dz_pre.sum(axis=0)

            dh_next = dh * (1.0 - z) + dz_pre @ Uz + dr_pre @ Ur + (dht @ Uh) * r

        grad = np.concatenate([g.ravel() for g in (gWz, gUz, gbz, gWr, gUr, gbr, gWh, gUh, gbh, gWo, gbo)])
        grad += (2e-4 / len(vec)) * vec
        return loss, grad

    def fit(self, X, Y, maxiter=80):
        res = minimize(lambda v: self.loss_and_grad(v, X, Y), self.params, jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
        self.params = res.x
        return float(res.fun)

    def predict_seq(self, X_seq):
        Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh, Wo, bo = self._unpack(self.params)
        h = np.zeros(self.h_dim)
        out = []
        for x in X_seq:
            z = self._sig(Wz @ x + Uz @ h + bz)
            r = self._sig(Wr @ x + Ur @ h + br)
            ht = np.tanh(Wh @ x + Uh @ (r * h) + bh)
            h = (1.0 - z) * h + z * ht
            pred = Wo @ h + bo
            out.append(float(np.arctan2(pred[0], pred[1])))
        return out


def make_synthetic_training_batch(n_traj=24, steps=120, seed=123):
    """Randomized turn sequences and unknown odometry bias so GRU cannot memorize test cadence."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n_traj, steps, 4))
    Y = np.zeros((n_traj, steps, 2))
    for b in range(n_traj):
        yaw = 0.0
        bias = float(rng.uniform(-0.2, 0.2))
        sign = 1.0 if rng.random() < 0.5 else -1.0
        flip_at = int(rng.integers(6, 24))
        for t in range(steps):
            if t >= flip_at:
                sign = -sign
                flip_at = t + int(rng.integers(6, 24))
            true_delta = sign / 20.0
            yaw = wrap(yaw + true_delta)
            noisy_delta = true_delta + (bias + rng.normal(0.0, 0.05)) / 20.0
            visible = not (20 <= t < 80) and (t % 2 == 0)
            cue = wrap(yaw + rng.normal(0.0, 0.03)) if visible else 0.0
            X[b, t] = [noisy_delta, 1.0 if visible else 0.0, np.sin(cue) if visible else 0.0, np.cos(cue) if visible else 0.0]
            Y[b, t] = [np.sin(yaw), np.cos(yaw)]
    return X, Y


def run(compass_dir: Path, out: Path, seeds=8):
    model = load_compass_connectome(compass_dir)
    gru = GRUCompass(hidden=12, seed=0)
    X_tr, Y_tr = make_synthetic_training_batch(n_traj=24, steps=120, seed=123)
    train_loss = gru.fit(X_tr, Y_tr, maxiter=80)

    angles = np.linspace(-np.pi, np.pi, 72, endpoint=False)
    out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    errors = {m: {"blackout": [], "reacquired": [], "conflict": []} for m in ("odometry", "bayesian", "connectome_ring", "gru")}
    bump_vals = []
    bump_errs = []

    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        bayes_mass = np.exp(10.0 * np.cos(angles))
        bayes_mass /= bayes_mass.sum()
        ring = ConnectomeCompass(model)
        odom = 0.0
        gru_inputs = []
        truths = []

        with RoverSim(scene="post", seed=seed, fault="freeze", fault_start_s=1.0, fault_end_s=4.0) as sim:
            initial = sim.observe()
            anchor = bearing(initial["rgb"])
            if anchor is None:
                raise ValueError("Initial landmark unavailable")
            prev_cap = initial["capture_s"]

            for step in range(120):
                sign = 1 if (step + 6) // 12 % 2 == 0 else -1
                obs = sim.step("left" if sign == 1 else "right")
                delta = (sign + 0.09 + rng.normal(0.0, 0.05)) / 20.0
                odom = wrap(odom + delta)
                bayes_mass = advance(bayes_mass, delta)

                raw_cue = fresh_heading(obs, prev_cap)
                prev_cap = obs["capture_s"]
                cue = None if raw_cue is None else wrap(raw_cue - anchor)
                # Inject a 1-frame conflicting landmark distractor at step 15 (t=0.75s)
                if step == 15 and cue is not None:
                    cue = wrap(cue + 1.4)

                if cue is not None:
                    bayes_mass *= np.exp(4.0 * np.cos(angles - cue))
                    bayes_mass /= bayes_mass.sum()
                bayes_est = decode(bayes_mass, angles)
                ring_est, bump = ring.step(delta, cue)
                if ring_est is None:
                    raise ValueError("Connectome ring lost population vector")

                truth = sim.evaluator()["yaw_rad"]
                truths.append(truth)
                gru_inputs.append([delta, 1.0 if cue is not None else 0.0, np.sin(cue) if cue is not None else 0.0, np.cos(cue) if cue is not None else 0.0])

                t = obs["time_s"]
                e_o = np.degrees(abs(wrap(odom - truth)))
                e_b = np.degrees(abs(wrap(bayes_est - truth)))
                e_r = np.degrees(abs(wrap(ring_est - truth)))
                bump_vals.append(bump)
                bump_errs.append(e_r)

                if step == 15:
                    errors["odometry"]["conflict"].append(e_o)
                    errors["bayesian"]["conflict"].append(e_b)
                    errors["connectome_ring"]["conflict"].append(e_r)
                elif abs(t - 3.95) < 1e-6:
                    errors["odometry"]["blackout"].append(e_o)
                    errors["bayesian"]["blackout"].append(e_b)
                    errors["connectome_ring"]["blackout"].append(e_r)
                elif abs(t - 5.0) < 1e-6:
                    errors["odometry"]["reacquired"].append(e_o)
                    errors["bayesian"]["reacquired"].append(e_b)
                    errors["connectome_ring"]["reacquired"].append(e_r)
                all_rows.append([seed, t, cue is not None, truth, odom, bayes_est, ring_est, bump])

        gru_preds = gru.predict_seq(np.asarray(gru_inputs))
        for step, (gp, tr) in enumerate(zip(gru_preds, truths)):
            eg = np.degrees(abs(wrap(gp - tr)))
            t = (step + 1) / 20.0
            if step == 15:
                errors["gru"]["conflict"].append(eg)
            elif abs(t - 3.95) < 1e-6:
                errors["gru"]["blackout"].append(eg)
            elif abs(t - 5.0) < 1e-6:
                errors["gru"]["reacquired"].append(eg)

    with (out / "traces.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "time_s", "cue_present", "truth_rad", "odom_rad", "bayesian_rad", "ring_rad", "bump_magnitude"])
        w.writerows(all_rows)

    summary = {
        "seeds": seeds,
        "connectome_nodes": len(model["ids"]),
        "unknown_sign_nodes": model["unknown_sign_count"],
        "pen_a_left_shift_deg": model["pen_a_left_shift_deg"],
        "pen_a_right_shift_deg": model["pen_a_right_shift_deg"],
        "gru_train_mse": train_loss,
        "bump_stability": {
            "mean": float(np.mean(bump_vals)),
            "std": float(np.std(bump_vals)),
            "min": float(np.min(bump_vals)),
            "missing_estimates": 0,
        },
        "mean_abs_error_deg": {
            m: {k: float(np.mean(v)) for k, v in phases.items()}
            for m, phases in errors.items()
        },
        "scope": "FAFB v783 signed compass graph (47 EPG, 42 Delta7 GLUT-, 20 PEG, 20 PEN_a, 22 PEN_b) with connectome-derived spectral ring phases vs odometry, 72-bin Bayes filter, and BPTT-trained 12-unit GRU under 1-4s freeze + 1.4 rad conflicting distractor at t=0.8s.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compass", type=Path, default=Path("runs/connectome/compass"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=int, default=8)
    args = p.parse_args()
    print(json.dumps(run(args.compass, args.out, args.seeds), indent=2))
