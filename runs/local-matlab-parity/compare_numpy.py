"""Local numpy-only equivalent of compare.py for validation & MATLAB reference cross-check.
Runs against the local .mat files in references/EMD-LPLC2-GF-model/code/1_Response.../Stimuli.
Produces comparison.json and response_traces.csv in runs/kaggle-lplc2-v1.
"""
import csv, json, hashlib, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from matv5 import load_patt

STIMULI_DIR = Path(r"D:\rover\references\EMD-LPLC2-GF-model\code\1_Response-Characteristics-Of-Single-LPLC2-Neurons-To-Different-Stimuli\Stimuli")
OUT = Path(r"D:\rover\runs\kaggle-lplc2-v1")
OUT.mkdir(parents=True, exist_ok=True)

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

# MATLAB filter parameters (from loom.m)
STEP = 0.01   # 10 ms temporal resolution
tauL = 0.050  # low-pass time constant  50 ms
tauH = 0.250  # high-pass time constant 250 ms
dL = tauL / STEP   # = 5
dH = tauH / STEP   # = 25
a_low_1 = 1.0 / (dL + 1)   # = 1/6
a_low_2 = 1.0 - a_low_1    # = 5/6
b_high_1 = dH / (dH + 1)   # = 25/26
b_high_2 = b_high_1         # = 25/26  (same as b_high_1 in MATLAB code)

THRESHOLD = 2.0
LPLC2_HALF = 50  # half-side of conv kernel (side=100)

def response_numpy(path):
    """Pure-numpy implementation matching loom.m + emd.m + lplc2conv2.m (additive and multiplicative)."""
    data = load_patt(str(path))
    # Match MATLAB: patt = 255*patt(1:4:end, 1:4:end, :)
    patt = (255.0 * data[::4, ::4, :]).astype(np.float64)  # shape (H, W, T)
    H, W, T = patt.shape

    # State arrays (MATLAB size is EMD_nx+1 x EMD_ny+1 == H x W)
    Fh = np.zeros((H, W))
    Fd_Off = np.zeros((H, W))

    records = []
    newpicture = patt[:, :, 0]

    for t in range(1, T):
        picture = newpicture
        newpicture = patt[:, :, t]

        # --- HIGH-PASS FILTER (emd.m) ---
        Fh = b_high_1 * (newpicture - picture) + b_high_2 * Fh
        # OFF rectification
        Off = np.maximum(0.05 - Fh, 0.0)   # OffRect(Fh, 0.05)  => max(0, threshold - signal)
        # --- LOW-PASS FILTER ---
        Fd_Off = a_low_1 * Off + a_low_2 * Fd_Off

        # EMD arms (MATLAB: He_Off = Fd_Off(i,j) * Off(i,j+1); Hi_Off = Off(i,j) * Fd_Off(i,j+1))
        # He = rightward motion (delayed left × instantaneous right)
        He_Off = Fd_Off[:H-1, :W-1] * Off[:H-1, 1:W]    # leftward arm
        Hi_Off = Off[:H-1, :W-1] * Fd_Off[:H-1, 1:W]    # rightward arm
        Ve_Off = Fd_Off[:H-1, :W-1] * Off[1:H, :W-1]    # downward arm
        Vi_Off = Off[:H-1, :W-1] * Fd_Off[1:H, :W-1]    # upward arm

        # LP inputs to lplc2conv2 (4-channel: leftward, rightward, upward, downward)
        # In MATLAB: LP(:,:,1)=Hi_Off (leftward), LP(:,:,2)=He_Off (rightward),
        #            LP(:,:,3)=Vi_Off (upward),    LP(:,:,4)=Ve_Off (downward)
        LP = np.stack([Hi_Off, He_Off, Vi_Off, Ve_Off], axis=-1)   # (H-1, W-1, 4)

        # --- LPLC2CONV2 ---
        # Kernel: side=100, Hside=50, threshold=2.0
        # kernel(:,:,1) => rows Hside-19:Hside+20, cols Hside+1:end  → right half, 40 rows centred
        # pre_LPLC2(:,:,1) = conv2(LP(:,:,1)-LP(:,:,2), kernel1, 'same')   → −horizontal
        # pre_LPLC2(:,:,2) = conv2(LP(:,:,2)-LP(:,:,1), kernel2, 'same')   → +horizontal
        # pre_LPLC2(:,:,3) = conv2(LP(:,:,3)-LP(:,:,4), kernel3, 'same')   → −vertical
        # pre_LPLC2(:,:,4) = conv2(LP(:,:,4)-LP(:,:,3), kernel4, 'same')   → +vertical
        #
        # For the CENTRAL PIXEL ONLY (as in compare.py), we don't do the full conv2;
        # we replicate the compare.py crop logic:
        #   cy, cx = (n-1)//2 - 1 for each spatial dim
        #   crop region: y:y+101, x:x+101  with y=cy-49, x=cx-49
        # Then sum arms in quadrants:
        #   arm0 = -horizontal[30:70, :50].sum()   left-motion right quadrant
        #   arm1 =  horizontal[30:70, 50:].sum()   right-motion left quadrant
        #   arm2 = -vertical[:50, 30:70].sum()      upward bottom quadrant
        #   arm3 =  vertical[50:, 30:70].sum()      downward top quadrant
        # horizontal = lag[:-1,:-1]*off[:-1,1:] - off[:-1,:-1]*lag[:-1,1:]  (== Fd_Off*Off right - Off*Fd_Off right)
        #            = He_Off - Hi_Off  (but compare.py uses lag=delay, off=Off directly)
        #
        # NOTE: compare.py carries its own state variables (high, delay) distinct from Fh/Fd_Off above.
        # We replicate exactly the compare.py PyTorch logic in numpy here for the central response.
        pass  # see below — we implement compare.py logic separately

    return None  # placeholder; use compare_py_numpy() below


def compare_py_numpy(path):
    """Exact numpy port of the PyTorch compare.py response() function."""
    data = load_patt(str(path))
    patt = np.ascontiguousarray(data[::4, ::4, :].transpose(2, 0, 1), dtype=np.float64)  # T, H, W
    movie = patt * 255.0

    T, H, W = movie.shape
    cy, cx = (H - 1) // 2 - 1, (W - 1) // 2 - 1
    y, x = cy - 49, cx - 49
    assert min(x, y) >= 0 and y + 101 <= H and x + 101 <= W, f"Bad dims {H}x{W}"
    crop = movie[:, y:y+101, x:x+101]   # T x 101 x 101

    high = np.zeros_like(crop[0])
    delay = np.zeros_like(high)
    records = []
    for i in range(1, len(crop)):
        high = (25.0 / 26.0) * (high + crop[i] - crop[i-1])
        off = np.maximum(0.05 - high, 0.0)
        lag = delay * (5.0 / 6.0) + off / 6.0
        horizontal = lag[:-1, :-1] * off[:-1, 1:] - off[:-1, :-1] * lag[:-1, 1:]
        vertical   = lag[:-1, :-1] * off[1:, :-1] - off[:-1, :-1] * lag[1:, :-1]
        arms = np.array([
            -horizontal[30:70, :50].sum(),
             horizontal[30:70, 50:].sum(),
            -vertical[:50, 30:70].sum(),
             vertical[50:, 30:70].sum(),
        ])
        active = np.where(arms > 2.0, arms, 0.0)
        records.append([active.sum(), active.prod()])
        delay = lag
    return np.array(records)   # (T-1, 2)


def matlab_reference_response(path):
    """Reproduce MATLAB loom.m + emd.m LPLC2_core (central pixel, additive) for cross-check.
    Returns array of shape (T-1,) of LPLC2_core values (additive mode, n!=1 in lplc2conv2).
    Note: full conv2('same') is expensive; we compute only the central pixel contribution.
    """
    data = load_patt(str(path))
    patt = (255.0 * data[::4, ::4, :]).astype(np.float64)
    H, W, T = patt.shape

    Fh = np.zeros((H, W))
    Fd_Off = np.zeros((H, W))
    newpicture = patt[:, :, 0]

    LPLC2_core = []
    cHalf = H // 2   # floor(end/2) in MATLAB (1-based → 0-based: H//2 - 1 for exact match)
    # MATLAB: LPLC2_core(i) = LPLC2(floor(end/2), floor(end/2), i)
    # end for LPLC2 is size(patt,1)-1 = H-1 (EMD output). floor((H-1)/2) → index (0-based: (H-1)//2 - 1)
    ci = (H - 1) // 2 - 1   # 0-based row index for MATLAB's floor(end/2)
    cj = (W - 1) // 2 - 1   # 0-based col index

    for t in range(1, T):
        picture = newpicture
        newpicture = patt[:, :, t]

        # HIGH-PASS
        Fh = b_high_1 * (newpicture - picture) + b_high_2 * Fh
        Off = np.maximum(0.05 - Fh, 0.0)
        # LOW-PASS
        Fd_Off = a_low_1 * Off + a_low_2 * Fd_Off

        # EMD arms at every pixel (H-1, W-1)
        He_Off = Fd_Off[:H-1, :W-1] * Off[:H-1, 1:W]      # rightward
        Hi_Off = Off[:H-1, :W-1] * Fd_Off[:H-1, 1:W]      # leftward
        Ve_Off = Fd_Off[:H-1, :W-1] * Off[1:H, :W-1]      # downward
        Vi_Off = Off[:H-1, :W-1] * Fd_Off[1:H, :W-1]      # upward

        # lplc2conv2 additive (n != 1):
        # pre_LPLC2(:,:,1) = conv2(LP1-LP2, kernel1, 'same')  then threshold
        # For central pixel ci,cj: kernel1 covers rows Hside-19:Hside+20, cols Hside+1:end (1-based)
        # i.e. rows [30:70], cols [50:100] in 0-based of the (H-1)x(W-1) EMD output grid
        # At central pixel ci,cj of LPLC2 output, conv2 'same' sums those LP cells within kernel support.
        # LP(:,:,1)=Hi_Off (leftward), LP(:,:,2)=He_Off (rightward)
        # For arm 1 (rightward quadrant, kernel1): LP1-LP2 = Hi_Off - He_Off at kernel1 support
        # kernel1 support centred on (ci,cj): rows ci+30:ci+70 → 30:70 relative, cols cj:cj+50 → right half
        # We use the same crop as compare.py but on the full EMD arrays.

        # Additive central pixel LPLC2:
        # arm contributions summed over kernel support regions:
        rH = Hi_Off - He_Off   # input to kernel1 and kernel2
        rV = Vi_Off - Ve_Off   # input to kernel3 and kernel4

        # Kernel1: right quadrant (cols Hside+1:end → 50:100), rows 30:70 → arm1 of compare.py
        # Note: compare.py's crop starts at (y,x) = (ci-49, cj-49); kernel1 covers [30:70, 50:100] within crop
        # Equivalent full-frame: rows (ci-49+30):(ci-49+70) = (ci-19):(ci+21), cols (cj-49+50):(cj-49+100) = (cj+1):(cj+51)
        kr0, kr1 = ci - 19, ci + 21     # row range (40 rows)
        kc_right = slice(cj + 1, cj + 51)  # cols for right quadrant
        kc_left  = slice(cj - 49, cj + 1)  # cols for left  quadrant
        kc_mid   = slice(cj - 19, cj + 21) # cols for centre (vertical kernels)
        kr_top_0, kr_top_1 = ci - 49, ci + 1
        kr_bot_0, kr_bot_1 = ci + 1, ci + 51

        def safe_sum(arr, r0, r1, cs):
            """Sum arr[r0:r1, cs] guarding bounds."""
            r0c = max(0, r0); r1c = min(arr.shape[0], r1)
            return arr[r0c:r1c, cs].sum() if r0c < r1c else 0.0

        # pre-LPLC2 for central pixel (additive, before thresholding)
        pre1 = safe_sum(rH,  kr0,       kr1,       kc_right)   # kernel1 (rightward arm)
        pre2 = safe_sum(-rH, kr0,       kr1,       kc_left)    # kernel2 (leftward arm)
        pre3 = safe_sum(rV,  kr_top_0,  kr_top_1,  kc_mid)     # kernel3 (upward arm)
        pre4 = safe_sum(-rV, kr_bot_0,  kr_bot_1,  kc_mid)     # kernel4 (downward arm)

        # Threshold + additive
        def thresh(x): return x if x > THRESHOLD else 0.0
        lplc2_add = thresh(pre1) + thresh(pre2) + thresh(pre3) + thresh(pre4)
        LPLC2_core.append(lplc2_add)

    return np.array(LPLC2_core)


if __name__ == "__main__":
    print("Running local numpy compare (compare.py logic + MATLAB reference cross-check)...")
    trace = {}
    hashes = {}
    matlab_ref = {}

    for name in CLIPS:
        path = STIMULI_DIR / (name + ".mat")
        if not path.exists():
            print(f"  MISSING: {path}")
            continue
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        t = compare_py_numpy(path)
        trace[name] = t
        mr = matlab_reference_response(path)
        matlab_ref[name] = mr
        print(f"  {name}: {len(t)} frames, additive_max={t[:,0].max():.4f}, mult_max={t[:,1].max():.6f}")

    if not trace:
        print("ERROR: No stimuli loaded.")
        raise SystemExit(1)

    # Check all 10 present and all finite
    missing = [n for n in CLIPS if n not in trace]
    all_finite = all(np.isfinite(v).all() for v in trace.values())
    print(f"\n--- COVERAGE ---")
    print(f"  Stimuli loaded: {len(trace)}/10, missing: {missing}")
    print(f"  All finite: {all_finite}")

    # Normalize
    maxima = np.max(np.vstack(list(trace.values())), axis=0)
    print(f"  Global maxima (additive, multiplicative): {maxima}")
    if np.any(maxima <= 0):
        raise ValueError("No reference response in one or more modes")

    # --- LOOMING vs CONTROLS summary ---
    print("\n--- LOOMING vs CONTROLS (additive normalized peak) ---")
    looming_add = (trace["Looming_50ms"][:, 0] / maxima[0]).max()
    looming_mul = (trace["Looming_50ms"][:, 1] / maxima[1]).max()
    print(f"  Looming_50ms          : add_norm={looming_add:.4f}  mul_norm={looming_mul:.6f}")

    control_groups = {
        "Horizontal controls": ["Bar_horizontal-motion", "Edge_horizontal-motion", "Grating_horizontal-motion"],
        "Vertical controls":   ["Bar_vertical-motion",  "Edge_vertical-motion",   "Grating_vertical-motion"],
        "Inward-motion controls": ["Cross_centrifugal-motion", "Cross_concentric-motion", "Square_expanded-motion"],
    }
    comparison_summary = {}
    for group, names in control_groups.items():
        peaks = []
        for n in names:
            if n in trace:
                ap = (trace[n][:, 0] / maxima[0]).max()
                mp = (trace[n][:, 1] / maxima[1]).max()
                peaks.append((n, ap, mp))
                print(f"  {n:40s}: add_norm={ap:.4f}  mul_norm={mp:.6f}")
        comparison_summary[group] = peaks

    # --- MATLAB cross-check ---
    print("\n--- MATLAB REFERENCE CROSS-CHECK (additive central response) ---")
    tol_abs = 0.05
    cross_check = {}
    for name in CLIPS:
        if name not in trace or name not in matlab_ref:
            continue
        py_add = trace[name][:, 0]
        mat_add = matlab_ref[name]
        # Compare normalised series (both normalised to their own max to remove absolute scale)
        py_norm = py_add / (py_add.max() + 1e-12)
        mat_norm = mat_add / (mat_add.max() + 1e-12)
        min_len = min(len(py_norm), len(mat_norm))
        diff = np.abs(py_norm[:min_len] - mat_norm[:min_len])
        max_diff = diff.max()
        mean_diff = diff.mean()
        agrees = bool(max_diff <= tol_abs)
        cross_check[name] = {
            "frames_py": len(py_add),
            "frames_matlab": len(mat_add),
            "max_abs_diff_normalised": float(max_diff),
            "mean_abs_diff_normalised": float(mean_diff),
            "tolerance": tol_abs,
            "agrees": agrees,
        }
        status = "OK" if agrees else "DISCREPANCY"
        print(f"  {name:40s}: max_diff={max_diff:.4f}  mean_diff={mean_diff:.4f}  [{status}]")

    # --- Write outputs ---
    with (OUT / "response_traces.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stimulus", "transition", "additive_normalized", "multiplicative_normalized"])
        for name, v in trace.items():
            w.writerows((name, i, *row) for i, row in enumerate(v / maxima, 1))

    summary = {
        "source": "local_numpy_run",
        "device": "CPU (numpy, no torch)",
        "stimuli_count": len(trace),
        "all_finite": all_finite,
        "sha256": hashes,
        "normalized_peak": {k: (v.max(axis=0) / maxima).tolist() for k, v in trace.items()},
        "looming_vs_controls": {
            "Looming_50ms": {"additive_norm_peak": float(looming_add), "multiplicative_norm_peak": float(looming_mul)},
            **{n: {"additive_norm_peak": float(ap), "multiplicative_norm_peak": float(mp)}
               for group_peaks in comparison_summary.values() for n, ap, mp in group_peaks},
        },
        "matlab_cross_check": cross_check,
        "method": "Central-pixel numpy port of PyTorch compare.py + MATLAB loom.m/emd.m/lplc2conv2.m additive central-pixel cross-check.",
        "tolerance_note": f"Normalised series compared at absolute tolerance {tol_abs}. MATLAB cross-check uses reconstructed central-pixel additive sum; not equivalent to full conv2('same') but tests filter chain parity.",
    }
    (OUT / "comparison.json").write_text(json.dumps(summary, indent=2))
    print(f"\nOutputs written to {OUT}")
    print("Done.")
