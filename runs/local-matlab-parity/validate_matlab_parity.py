"""Exact numerical parity verification between VIGIL_NEURO_FINAL/kaggle-stimulus/compare.py
and the authors' MATLAB implementation in references/EMD-LPLC2-GF-model.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from matv5 import load_patt

STIMULI_DIR = Path(
    r"D:\rover\references\EMD-LPLC2-GF-model\code\1_Response-Characteristics-Of-Single-LPLC2-Neurons-To-Different-Stimuli\Stimuli"
)
OUT_DIR = Path(r"D:\rover\runs\local-matlab-parity")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLIPS = (
    "Looming_50ms",
    "Square_expanded-motion",
    "Bar_horizontal-motion",
    "Edge_horizontal-motion",
    "Bar_vertical-motion",
    "Edge_vertical-motion",
    "Grating_horizontal-motion",
    "Grating_vertical-motion",
    "Cross_centrifugal-motion",
    "Cross_concentric-motion",
)


def conv2_same_matlab(A: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Exact MATLAB conv2(A, K, 'same') for 2D arrays.
    In MATLAB:
      full(I, J) = sum_{m, n} A(I - m, J - n) * K(m, n)  (0-based)
      same starts at floor(Mk/2), floor(Nk/2) in 0-based indices of full.
    Uses FFT-free exact 2D box/direct evaluation or 2D cumsum for binary kernels?
    Let's make it exact and fast via 2D FFT? Wait: float64 FFT has ~1e-11 rounding.
    Even better: K is a single solid rectangle of 1s! So we can use 2D integral image (exact float64 sums, or check row/col ranges).
    Actually, 2D integral image has subtraction rounding ~1e-12; even simpler: for exact MATLAB parity at (cy, cx) AND full array,
    we can use explicit shifted sums or 2D integral image. Let's extract the bounding box of 1s in K and compute exact sum via 1D sliding window (zero subtraction error!).
    """
    Ma, Na = A.shape
    Mk, Nk = K.shape
    # Find rectangle of 1s in K (all 4 kernels in lplc2conv2.m are solid rectangles of 1.0)
    rows = np.where(K.any(axis=1))[0]
    cols = np.where(K.any(axis=0))[0]
    r0, r1 = int(rows[0]), int(rows[-1]) + 1  # K[r0:r1, c0:c1] == 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    assert np.all(K[r0:r1, c0:c1] == 1.0) and np.sum(K) == (r1 - r0) * (c1 - c0)

    # Full convolution index I = oy + oy_off where oy_off = Mk // 2 (MATLAB floor(Mk/2))
    # For output (oy, ox), we sum A[ay, ax] where ay = oy + Mk//2 - m, m in [r0, r1-1]
    # => ay in [oy + Mk//2 - (r1 - 1), oy + Mk//2 - r0] inclusive!
    oy_off = Mk // 2
    ox_off = Nk // 2
    # Pad A with zeros so sliding sum needs no conditionals
    pad_top = oy_off
    pad_bot = oy_off
    pad_left = ox_off
    pad_right = ox_off
    # Actually let's use exact loop over kernel or cumsum? Let's just sum slices!
    # Wait: (r1-r0) is 40 or 50, (c1-c0) is 50 or 40 -> only 2000 slices? No, separable!
    # First sum along axis=0 for rows, then along axis=1 for cols (pure additions, NO subtractions!).
    # Wait: width is 40 or 50, summing 40 or 50 shifted slices in numpy takes < 0.2ms for (149, 199)!
    # Step 1: vertical sum over m in [r0, r1)
    vsum = np.zeros_like(A, dtype=np.float64)
    for m in range(r0, r1):
        # ay = oy + oy_off - m -> shift = oy_off - m
        s = oy_off - m
        if s >= 0:
            if s < Ma:
                vsum[: Ma - s, :] += A[s:, :]
        else:
            if -s < Ma:
                vsum[-s:, :] += A[: Ma + s, :]
    # Step 2: horizontal sum over n in [c0, c1)
    out = np.zeros_like(A, dtype=np.float64)
    for n in range(c0, c1):
        s = ox_off - n
        if s >= 0:
            if s < Na:
                out[:, : Na - s] += vsum[:, s:]
        else:
            if -s < Na:
                out[:, -s:] += vsum[:, : Na + s]
    return out


def verify_conv2_same():
    """Verify conv2_same_matlab against brute-force convolution on a small array."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(15, 19))
    K = np.zeros((10, 10), dtype=np.float64)
    K[3:7, 5:10] = 1.0  # mimics kernel1 scaled down 10x
    fast = conv2_same_matlab(A, K)
    # Brute force full conv
    Ma, Na = A.shape
    Mk, Nk = K.shape
    full = np.zeros((Ma + Mk - 1, Na + Nk - 1), dtype=np.float64)
    for i in range(Ma):
        for j in range(Na):
            full[i : i + Mk, j : j + Nk] += A[i, j] * K
    bf = full[Mk // 2 : Mk // 2 + Ma, Nk // 2 : Nk // 2 + Na]
    max_err = np.max(np.abs(fast - bf))
    assert max_err < 1e-13, f"conv2_same_matlab mismatch: {max_err}"


def response_compare_py_logic(path: Path) -> np.ndarray:
    """Exact numpy float64 copy of VIGIL_NEURO_FINAL/kaggle-stimulus/compare.py:response()."""
    data = load_patt(str(path))
    sampled = np.ascontiguousarray(data[::4, ::4, :].transpose(2, 0, 1), dtype=np.float64)
    del data
    movie = sampled * 255.0
    cy, cx = [(n - 1) // 2 - 1 for n in movie.shape[1:]]
    y, x = cy - 49, cx - 49
    if min(x, y) < 0 or y + 101 > movie.shape[1] or x + 101 > movie.shape[2]:
        raise ValueError("Unexpected stimulus dimensions")
    crop = movie[:, y : y + 101, x : x + 101]
    high = np.zeros_like(crop[0])
    delay = np.zeros_like(high)
    records = []
    for i in range(1, len(crop)):
        high = (25.0 / 26.0) * (high + crop[i] - crop[i - 1])
        off = np.maximum(0.05 - high, 0.0)
        lag = delay * (5.0 / 6.0) + off / 6.0
        horizontal = lag[:-1, :-1] * off[:-1, 1:] - off[:-1, :-1] * lag[:-1, 1:]
        vertical = lag[:-1, :-1] * off[1:, :-1] - off[:-1, :-1] * lag[1:, :-1]
        arms = np.array(
            [
                -horizontal[30:70, :50].sum(),
                horizontal[30:70, 50:].sum(),
                -vertical[:50, 30:70].sum(),
                vertical[50:, 30:70].sum(),
            ]
        )
        active = np.where(arms > 2.0, arms, 0.0)
        records.append([active.sum(), active.prod()])
        delay = lag
    return np.array(records, dtype=np.float64)


def response_matlab_exact(path: Path) -> np.ndarray:
    """Exact reproduction of MATLAB loom.m + emd.m + OffRect.m + lplc2conv2.m (additive and multiplicative)."""
    data = load_patt(str(path))
    patt = 255.0 * data[::4, ::4, :].astype(np.float64)  # (150, 200, 500)
    H, W, T = patt.shape

    STEP = 0.01
    tauL = 0.050
    dL = tauL / STEP
    a_low_1 = 1.0 / (dL + 1.0)
    a_low_2 = 1.0 - a_low_1

    tauH = 0.250
    dH = tauH / STEP
    b_high_1 = dH / (dH + 1.0)
    b_high_2 = b_high_1

    EMD_nx = H - 1
    EMD_ny = W - 1
    Fh_before = np.zeros((EMD_nx + 1, EMD_ny + 1), dtype=np.float64)
    Fd_Off_before = np.zeros((EMD_nx + 1, EMD_ny + 1), dtype=np.float64)

    # Kernels from lplc2conv2.m (1-based -> 0-based)
    side = 100
    Hside = side // 2  # 50
    kernels = np.zeros((side, side, 4), dtype=np.float64)
    # kernel(Hside-19:Hside+20, Hside+1:end, 1) -> 31..70, 51..100 (1-based) -> 30:70, 50:100 (0-based)
    kernels[Hside - 20 : Hside + 20, Hside:, 0] = 1.0
    # kernel(Hside-19:Hside+20, 1:Hside, 2) -> 30:70, 0:50
    kernels[Hside - 20 : Hside + 20, :Hside, 1] = 1.0
    # kernel(Hside+1:end, Hside-19:Hside+20, 3) -> 50:100, 30:70
    kernels[Hside:, Hside - 20 : Hside + 20, 2] = 1.0
    # kernel(1:Hside, Hside-19:Hside+20, 4) -> 0:50, 30:70
    kernels[:Hside, Hside - 20 : Hside + 20, 3] = 1.0

    # MATLAB: LPLC2_core(i) = LPLC2(floor(end/2), floor(end/2), i)
    # size(LPLC2, 1) = EMD_nx = 149 -> floor(149/2) = 74 (1-based) -> 73 (0-based)
    # size(LPLC2, 2) = EMD_ny = 199 -> floor(199/2) = 99 (1-based) -> 98 (0-based)
    cy = EMD_nx // 2 - 1 if EMD_nx % 2 == 0 else (EMD_nx - 1) // 2 - 1  # Wait: floor(149/2)=74 -> 0-based 73; (149-1)//2 - 1 = 73!
    # Wait: floor(end/2) in 1-based is end // 2! Converting 1-based index k to 0-based is k - 1 = (end // 2) - 1!
    cy = (EMD_nx // 2) - 1  # floor(149/2)-1 = 74-1 = 73; for even 150 -> 75-1 = 74
    cx = (EMD_ny // 2) - 1  # floor(199/2)-1 = 99-1 = 98

    threshold = 2.0
    newpicture = patt[:, :, 0]
    records = []

    for i in range(1, T):
        picture = newpicture
        newpicture = patt[:, :, i]

        # emd.m
        Fh = b_high_1 * (newpicture - picture) + b_high_2 * Fh_before
        # OffRect(Fh, 0.05): if Fh < 0.05: abs(Fh - 0.05) else 0
        Off = np.where(Fh < 0.05, np.abs(Fh - 0.05), 0.0)
        Fd_Off = a_low_1 * Off + a_low_2 * Fd_Off_before

        He_Off = Fd_Off[:-1, :-1] * Off[:-1, 1:]
        Hi_Off = Off[:-1, :-1] * Fd_Off[:-1, 1:]
        Ve_Off = Fd_Off[:-1, :-1] * Off[1:, :-1]
        Vi_Off = Off[:-1, :-1] * Fd_Off[1:, :-1]

        # loom.m LP assignment
        LP1 = Hi_Off  # leftward
        LP2 = He_Off  # rightward
        LP3 = Vi_Off  # upward
        LP4 = Ve_Off  # downward

        # lplc2conv2.m
        pre1 = conv2_same_matlab(LP1 - LP2, kernels[:, :, 0])
        pre2 = conv2_same_matlab(LP2 - LP1, kernels[:, :, 1])
        pre3 = conv2_same_matlab(LP3 - LP4, kernels[:, :, 2])
        pre4 = conv2_same_matlab(LP4 - LP3, kernels[:, :, 3])

        p1 = pre1[cy, cx] if pre1[cy, cx] > threshold else 0.0
        p2 = pre2[cy, cx] if pre2[cy, cx] > threshold else 0.0
        p3 = pre3[cy, cx] if pre3[cy, cx] > threshold else 0.0
        p4 = pre4[cy, cx] if pre4[cy, cx] > threshold else 0.0

        add_val = p1 + p2 + p3 + p4
        mul_val = p1 * p2 * p3 * p4
        records.append([add_val, mul_val])

        Fh_before = Fh
        Fd_Off_before = Fd_Off

    return np.array(records, dtype=np.float64)


def main():
    verify_conv2_same()
    print("Verified conv2_same_matlab against brute force (err < 1e-13).", flush=True)

    py_traces = {}
    mat_traces = {}
    hashes = {}

    for name in CLIPS:
        path = STIMULI_DIR / f"{name}.mat"
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        py_traces[name] = response_compare_py_logic(path)
        mat_traces[name] = response_matlab_exact(path)
        print(f"Computed {name}: frames={len(py_traces[name])}", flush=True)

    py_max = np.max(np.vstack(list(py_traces.values())), axis=0)
    mat_max = np.max(np.vstack(list(mat_traces.values())), axis=0)

    # Compare per stimulus (raw and globally normalized)
    tol_norm = 1e-12
    tol_raw_rel = 1e-12
    parity = {}
    print("\n--- EXACT PARITY REPORT (compare.py logic vs MATLAB loom.m+emd.m+lplc2conv2.m) ---")
    for name in CLIPS:
        p = py_traces[name]
        m = mat_traces[name]
        pn = p / py_max
        mn = m / mat_max

        raw_abs_diff = np.max(np.abs(p - m), axis=0)
        denom = np.maximum(np.abs(m), 1e-30)
        raw_rel_diff = np.max(np.where(np.abs(m) > 0, np.abs(p - m) / denom, np.abs(p - m)), axis=0)
        norm_abs_diff = np.max(np.abs(pn - mn), axis=0)

        agrees_add = bool(norm_abs_diff[0] <= tol_norm)
        agrees_mul = bool(norm_abs_diff[1] <= tol_norm)
        parity[name] = {
            "frames": int(len(p)),
            "all_finite_py": bool(np.isfinite(p).all()),
            "all_finite_matlab": bool(np.isfinite(m).all()),
            "py_normalized_peak": (p.max(axis=0) / py_max).tolist(),
            "matlab_normalized_peak": (m.max(axis=0) / mat_max).tolist(),
            "raw_max_abs_diff": raw_abs_diff.tolist(),
            "raw_max_rel_diff": raw_rel_diff.tolist(),
            "norm_max_abs_diff": norm_abs_diff.tolist(),
            "tolerance_norm_abs": tol_norm,
            "agrees_additive": agrees_add,
            "agrees_multiplicative": agrees_mul,
        }
        print(
            f"  {name:28s} | norm_diff_add={norm_abs_diff[0]:.3e} norm_diff_mul={norm_abs_diff[1]:.3e} "
            f"| rel_raw=({raw_rel_diff[0]:.3e}, {raw_rel_diff[1]:.3e}) | agree=({agrees_add}, {agrees_mul})"
        )

    out = {
        "global_max_py": py_max.tolist(),
        "global_max_matlab": mat_max.tolist(),
        "tolerance_norm_abs": tol_norm,
        "parity": parity,
    }
    (OUT_DIR / "matlab_parity_exact.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved exact parity report to {OUT_DIR / 'matlab_parity_exact.json'}")


if __name__ == "__main__":
    main()
