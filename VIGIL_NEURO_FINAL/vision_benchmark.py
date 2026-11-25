"""G04 biological vision benchmark: published EMD-LPLC2 model on RoverSim RGB regimes."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pybullet as pb

from rover_sim import RoverSim


def emd_lplc2_traces(frames, half=40, strip=16, threshold=2.0):
    """Run published EMD + four-arm LPLC2 opponency on luminance [0, 255] frames.

    Uses exact temporal filter constants from Zhao et al. (tau_H=0.25s, tau_L=0.05s,
    dt=0.01s -> 25/26 high-pass, 5/6 low-pass, OffRect threshold 0.05) over a
    symmetric foveal receptive field of side 2*half with arm strip width 2*strip.
    """
    if frames.ndim != 3 or len(frames) < 2:
        raise ValueError("Expected (T>=2, H, W) luminance frames")
    h, w = frames.shape[1:]
    cy, cx = (h - 1) // 2, (w - 1) // 2
    if cy - half < 0 or cy + half > h - 1 or cx - half < 0 or cx + half > w - 1:
        raise ValueError(f"Receptive field half={half} exceeds EMD grid ({h-1}, {w-1})")

    movie = frames.astype(np.float64)
    high = np.zeros_like(movie[0])
    delay = np.zeros_like(high)
    records = []
    for i in range(1, len(movie)):
        high = (25.0 / 26.0) * (high + movie[i] - movie[i - 1])
        off = np.maximum(0.05 - high, 0.0)
        lag = delay * (5.0 / 6.0) + off / 6.0
        horizontal = lag[:-1, :-1] * off[:-1, 1:] - off[:-1, :-1] * lag[:-1, 1:]
        vertical = lag[:-1, :-1] * off[1:, :-1] - off[:-1, :-1] * lag[1:, :-1]
        arms = np.array(
            [
                -horizontal[cy - strip : cy + strip, cx - half : cx].sum(),
                horizontal[cy - strip : cy + strip, cx : cx + half].sum(),
                -vertical[cy - half : cy, cx - strip : cx + strip].sum(),
                vertical[cy : cy + half, cx - strip : cx + strip].sum(),
            ],
            dtype=np.float64,
        )
        active = np.where(arms > threshold, arms, 0.0)
        records.append((float(active.sum()), float(active.prod())))
        delay = lag
    return np.asarray(records, dtype=np.float64)


def to_luminance(rgb):
    r, g, b = rgb.astype(np.float64).transpose(2, 0, 1)
    return 0.299 * r + 0.587 * g + 0.114 * b


def capture_regime(regime, seed=7):
    """Capture 10 Hz RoverSim RGB frames (96x128, 66 deg vertical FOV) for each G04 regime."""
    frames = []
    with RoverSim(scene="open", seed=seed) as sim:
        post = None
        if regime == "approach":
            post = sim.box([0.62, 0.0, 0.12], [0.03, 0.035, 0.12], 0, [0.86, 0.17, 0.09, 1])
        elif regime == "recede":
            post = sim.box([0.28, 0.0, 0.12], [0.03, 0.035, 0.12], 0, [0.86, 0.17, 0.09, 1])
        elif regime == "translate":
            post = sim.box([0.55, -0.32, 0.12], [0.03, 0.035, 0.12], 0, [0.86, 0.17, 0.09, 1])
        elif regime == "yaw":
            post = sim.box([0.55, 0.0, 0.12], [0.03, 0.035, 0.12], 0, [0.86, 0.17, 0.09, 1])
        elif regime == "near_miss":
            post = sim.box([0.62, 0.18, 0.12], [0.03, 0.035, 0.12], 0, [0.86, 0.17, 0.09, 1])
        elif regime == "empty":
            pass
        else:
            raise ValueError(f"Unknown regime: {regime}")
        if post is not None:
            sim.solids.append(post)
        sim.packet = None
        sim.queue.clear()
        sim.update_camera()
        last_cap = None
        for step in range(36):
            obs = sim.observe()
            if obs["frame_valid"] and obs["capture_s"] != last_cap:
                frames.append(to_luminance(obs["rgb"]))
                last_cap = obs["capture_s"]
            if regime in ("approach", "near_miss", "empty"):
                sim.step("forward")
            elif regime == "yaw":
                sim.step("left")
            elif regime == "recede":
                x = 0.28 + 0.018 * (step + 1)
                pb.resetBasePositionAndOrientation(post, [x, 0.0, 0.12], [0, 0, 0, 1], physicsClientId=sim.client)
                sim.step("brake")
            elif regime == "translate":
                y = -0.32 + 0.018 * (step + 1)
                pb.resetBasePositionAndOrientation(post, [0.55, y, 0.12], [0, 0, 0, 1], physicsClientId=sim.client)
                sim.step("brake")
    return np.stack(frames)


def run(out, seed=7):
    regimes = ("approach", "recede", "translate", "yaw", "near_miss", "empty")
    traces = {}
    for name in regimes:
        clips = capture_regime(name, seed=seed)
        traces[name] = emd_lplc2_traces(clips)
    maxima = np.max(np.vstack(list(traces.values())), axis=0)
    if maxima[0] <= 0 or maxima[1] <= 0:
        raise ValueError("Expected positive LPLC2 response on approach regime")

    out.mkdir(parents=True, exist_ok=True)
    with (out / "traces.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime", "transition", "additive_raw", "multiplicative_raw", "additive_normalized", "multiplicative_normalized"])
        for name, arr in traces.items():
            norm = arr / maxima
            for idx, (raw, nrm) in enumerate(zip(arr, norm), 1):
                w.writerow([name, idx, float(raw[0]), float(raw[1]), float(nrm[0]), float(nrm[1])])

    peaks = {
        name: {
            "additive_raw_peak": float(arr[:, 0].max()),
            "multiplicative_raw_peak": float(arr[:, 1].max()),
            "additive_norm_peak": float(arr[:, 0].max() / maxima[0]),
            "multiplicative_norm_peak": float(arr[:, 1].max() / maxima[1]),
        }
        for name, arr in traces.items()
    }
    app_mul = peaks["approach"]["multiplicative_norm_peak"]
    controls = ("recede", "translate", "yaw", "near_miss", "empty")
    tolerances = {
        "approach_is_peak_multiplicative": bool(app_mul == 1.0),
        "max_control_multiplicative_norm": max(peaks[c]["multiplicative_norm_peak"] for c in controls),
        "zero_false_positive_multiplicative_controls": all(peaks[c]["multiplicative_norm_peak"] <= 1e-3 for c in controls),
    }
    summary = {
        "sensor_geometry": {
            "resolution_hw": [96, 128],
            "vertical_fov_deg": 66.0,
            "sampling_hz": 10.0,
            "receptive_field_side_px": 80,
            "arm_strip_width_px": 32,
            "arm_threshold": 2.0,
        },
        "regimes": peaks,
        "acceptance_checks": tolerances,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=7)
    print(json.dumps(run(p.parse_args().out, p.parse_args().seed), indent=2))
