"""Matched open-loop warnings and closed-loop braking from rover RGB and FAFB topology."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.sparse import diags, load_npz

from rover_sim import RoverSim


def _post_angles(rgb):
    """Extract horizontal angular bounds (theta_left, theta_right) of red target pixels."""
    r, g, b = rgb.astype(float).transpose(2, 0, 1)
    _, x = np.nonzero((r > 60) & (r > 1.7 * g) & (r > 1.7 * b))
    if len(x) < 6:
        return None
    focal = rgb.shape[0] / (2 * np.tan(np.radians(66) / 2))
    center = (rgb.shape[1] - 1) / 2
    theta_l = float(np.arctan((x.min() - 0.5 - center) / focal))
    theta_r = float(np.arctan((x.max() + 0.5 - center) / focal))
    return theta_l, theta_r


def visual_drive(rgb, previous_size, dt):
    """Apparent target width (rad) and positive width change (rad/s) in a red-post pilot."""
    angles = _post_angles(rgb)
    if angles is None:
        return 0.0, 0.0
    theta_l, theta_r = angles
    size = float(theta_r - theta_l)
    growth = 0.0 if previous_size is None else max(0.0, (size - previous_size) / dt)
    return size, float(growth)


def visual_looming(rgb, previous_state, dt):
    """Collision-course LPLC2 (centered size) and LC4 (drift-suppressed expansion) drives."""
    angles = _post_angles(rgb)
    if angles is None:
        return 0.0, 0.0, None
    theta_l, theta_r = angles
    size = float(theta_r - theta_l)
    bearing = 0.5 * (theta_l + theta_r)
    # Suppress targets off the collision cone (near misses shift laterally).
    center_weight = float(np.exp(-((bearing / 0.14) ** 2)))
    if previous_state is None:
        return size * center_weight, 0.0, (theta_l, theta_r, size, bearing)
    prev_l, prev_r, prev_size, prev_bearing = previous_state
    growth = max(0.0, (size - prev_size) / dt)
    bearing_rate = abs(bearing - prev_bearing) / dt
    # Require expansion to dominate lateral bearing drift (LPLC2 radial opponency).
    radial_gate = max(0.0, 1.0 - bearing_rate / max(growth, 0.05))
    return size * center_weight, float(growth * center_weight * radial_gate), (theta_l, theta_r, size, bearing)


class Network:
    def __init__(self, path, labels):
        matrix = load_npz(path)
        totals = np.asarray(matrix.sum(axis=1)).ravel()
        self.weights = diags(1 / np.maximum(1, totals)) @ matrix
        self.size_cells = labels == "LPLC2"
        self.speed_cells = labels == "LC4"
        self.gf = labels == "DNp01"
        if not all(np.any(x) for x in (self.size_cells, self.speed_cells, self.gf)):
            raise ValueError("Incomplete escape circuit")
        self.state = np.zeros(len(labels))

    def update(self, size, growth):
        drive = self.size_cells * size + self.speed_cells * growth
        self.state += 0.2 * (np.tanh(drive + 0.8 * (self.weights @ self.state)) - self.state)
        return float(self.state[self.gf].mean())


def choose_threshold(samples):
    """Select the earliest threshold with zero calibration false positives (with margin)."""
    scores = np.asarray([s for s, _ in samples], dtype=float)
    labels = np.asarray([y for _, y in samples], dtype=bool)
    if labels.all() or not labels.any():
        raise ValueError("Calibration requires contacts and near misses")
    neg_max = float(scores[~labels].max())
    pos_scores = np.sort(scores[labels])
    above = pos_scores[pos_scores > neg_max]
    if len(above) > 0:
        # Earliest positive above all negatives, retaining a safety margin above neg_max.
        earliest_pos = float(above[0])
        return float(min(earliest_pos, neg_max + 0.5 * (earliest_pos - neg_max)))
    # Fallback only if classes overlap completely: minimize false positives first.
    thresholds = np.unique(np.r_[scores, np.nextafter(scores.max(), np.inf)])

    def objective(t):
        return (float((scores[~labels] < t).mean()), float((scores[labels] >= t).mean()), -t)

    return float(max(thresholds, key=objective))


def episode(seed, models, threshold=None):
    rng = np.random.default_rng(seed)
    x = float(rng.uniform(0.52, 0.85))
    side = float(rng.choice([-1, 1]))
    y = side * float(rng.uniform(0.13, 0.24) if seed % 2 else rng.uniform(0, 0.009))
    with RoverSim(scene="open", seed=seed) as sim:
        sim.solids.append(sim.box([x, y, 0.12], [0.03, 0.018, 0.12], 0, [0.86, 0.17, 0.09, 1]))
        sim.update_camera()
        obs = sim.observe()
        capture = None
        vstate = None
        latched = False
        peaks = {k: 0.0 for k in models}
        first = {k: None for k in models}
        contact = None
        history = []
        for _ in range(80):
            if obs["capture_s"] != capture:
                dt = 0.1 if capture is None else obs["capture_s"] - capture
                lplc2_drive, lc4_drive, vstate = visual_looming(obs["rgb"], vstate, dt)
                capture = obs["capture_s"]
                scores = {}
                for k, model in models.items():
                    score = model.update(lplc2_drive, lc4_drive)
                    scores[k] = score
                    peaks[k] = max(peaks[k], score)
                    if threshold is not None and score >= threshold[k] and first[k] is None:
                        first[k] = obs["time_s"]
                        latched = True
                history.append((obs["time_s"], scores))
            action = "brake" if len(models) == 1 and threshold is not None and latched else "forward"
            obs = sim.step(action)
            if sim.evaluator()["body_contact"]:
                contact = obs["time_s"]
                break
        state = sim.evaluator()
        # Upper bound on stop time at full speed plus one camera period.
        deadline = float("inf") if contact is None else contact - (0.35 / 0.8 + 0.1)
        timely = {k: max((scores[k] for t, scores in history if t <= deadline), default=0.0) for k in models}
    return dict(seed=seed, post_x=x, post_y=y, contact=contact, progress=state["x_m"], peaks=peaks, timely=timely, first=first)


def run(graph, annotations, out, pairs):
    if pairs < 1:
        raise ValueError("At least one pair required")
    ids = (graph / "nodes.txt").read_text().splitlines()
    with annotations.open() as f:
        types = {r["root_id"]: r["primary_type"] for r in csv.DictReader(f)}
    labels = np.array([types[i] for i in ids])
    paths = {f.stem: f for f in sorted(graph.glob("*.npz"))}
    if len(paths) < 2 or "biological" not in paths:
        raise ValueError("Missing matched networks")

    def networks(keys):
        return {k: Network(paths[k], labels) for k in keys}

    calibration = [episode(s, networks(paths)) for s in range(2 * pairs)]
    thresholds = {k: choose_threshold([(e["timely"][k], e["contact"] is not None) for e in calibration]) for k in paths}
    rows = []
    for s in range(20000, 20000 + 2 * pairs):
        replay = episode(s, networks(paths), thresholds)
        for k in paths:
            closed = episode(s, networks([k]), thresholds)
            contact, warning = replay["contact"], replay["first"][k]
            rows.append(
                dict(
                    topology=k,
                    seed=s,
                    x=replay["post_x"],
                    y=replay["post_y"],
                    reference_contact_s=contact,
                    warning_s=warning,
                    lead_s=None if warning is None or contact is None else contact - warning,
                    closed_contact=closed["contact"] is not None,
                    forward_progress_m=closed["progress"],
                    unobstructed_pass=closed["progress"] > replay["post_x"] + 0.15 and closed["contact"] is None,
                )
            )
    out.mkdir(parents=True, exist_ok=True)
    with (out / "episodes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    metrics = {}
    for k in paths:
        subset = [r for r in rows if r["topology"] == k]
        hazard = [r for r in subset if r["reference_contact_s"] is not None]
        misses = [r for r in subset if r["reference_contact_s"] is None]
        leads = [r["lead_s"] for r in hazard if r["lead_s"] is not None]
        metrics[k] = dict(
            threshold=thresholds[k],
            collision_courses=len(hazard),
            near_misses=len(misses),
            early_warnings=len(leads),
            median_lead_s=float(np.median(leads)) if leads else None,
            false_warnings=sum(r["warning_s"] is not None for r in misses),
            closed_collisions=sum(r["closed_contact"] for r in subset),
            near_misses_passed=sum(r["unobstructed_pass"] for r in misses),
        )
    (out / "summary.json").write_text(
        json.dumps(
            dict(
                calibration_seeds=[0, 2 * pairs - 1],
                test_seeds=[20000, 20000 + 2 * pairs - 1],
                metrics=metrics,
                scope="One visible red post, fixed camera and shape family. Bearing-gated LPLC2 size and drift-suppressed LC4 expansion drives, unsigned rates, zero-FP per-topology calibrated threshold, latched brake. No active recovery.",
            ),
            indent=2,
        )
    )
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("graph", "annotations", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--pairs", type=int, default=4)
    print(run(**vars(p.parse_args())))
