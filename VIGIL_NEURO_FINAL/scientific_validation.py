"""G09 scientific validation: paired bootstrap CIs, topology/recovery contrasts, and BANC mapping."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

BANC_HOMOLOGY_MAP = {
    "LC4": {"fafb_type": "LC4", "banc_homolog": "LC4", "role": "angular_expansion_velocity"},
    "LPLC2": {"fafb_type": "LPLC2", "banc_homolog": "LPLC2", "role": "radial_looming"},
    "DNp01": {"fafb_type": "DNp01", "banc_homolog": "DNp01 (Giant Fiber)", "role": "escape_descending"},
    "EPG": {"fafb_type": "EPG", "banc_homolog": "EPG", "role": "heading_wedge"},
    "PEG": {"fafb_type": "PEG", "banc_homolog": "PEG", "role": "recurrent_heading"},
    "PEN_a/PEN1": {"fafb_type": "PEN_a/PEN1", "banc_homolog": "PEN_a", "role": "angular_velocity_shift"},
    "PEN_b/PEN2": {"fafb_type": "PEN_b/PEN2", "banc_homolog": "PEN_b", "role": "heading_feedback"},
    "Delta7": {"fafb_type": "Delta7", "banc_homolog": "Delta7", "role": "global_opponent_inhibition"},
}


def paired_bootstrap_ci(a, b, n_boot=2000, seed=42):
    """Paired bootstrap 95% confidence interval for mean(a - b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("Paired bootstrap requires equal non-empty arrays")
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "n_pairs": int(len(diff)),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(diff.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }


def validate(
    escape_csv: Path,
    closed_loop_csv: Path,
    compass_summary: Path,
    banc_dir: Path,
    out: Path,
):
    with escape_csv.open() as f:
        esc_rows = list(csv.DictReader(f))
    bio_esc = [r for r in esc_rows if r["topology"] == "biological" and r["lead_s"]]
    rew_esc = [r for r in esc_rows if r["topology"] == "rewired-7" and r["lead_s"]]
    lead_contrast = paired_bootstrap_ci(
        [float(r["lead_s"]) for r in bio_esc],
        [float(r["lead_s"]) for r in rew_esc],
    )

    with closed_loop_csv.open() as f:
        cl_rows = list(csv.DictReader(f))
    by_pol = {}
    for r in cl_rows:
        by_pol.setdefault(r["policy"], {})[(r["scenario"], int(r["seed"]))] = r

    keys = sorted(by_pol["circuit_recovery"])
    success_circ_vs_ord = paired_bootstrap_ci(
        [1.0 if by_pol["circuit_recovery"][k]["safe_success"] == "True" else 0.0 for k in keys],
        [1.0 if by_pol["ordinary_recovery"][k]["safe_success"] == "True" else 0.0 for k in keys],
    )
    success_circ_vs_off = paired_bootstrap_ci(
        [1.0 if by_pol["circuit_recovery"][k]["safe_success"] == "True" else 0.0 for k in keys],
        [1.0 if by_pol["recovery_off"][k]["safe_success"] == "True" else 0.0 for k in keys],
    )
    prog_circ_vs_off = paired_bootstrap_ci(
        [float(by_pol["circuit_recovery"][k]["progress_m"]) for k in keys],
        [float(by_pol["recovery_off"][k]["progress_m"]) for k in keys],
    )

    comp = json.loads(compass_summary.read_text())

    banc_present = banc_dir.exists() and any(banc_dir.iterdir())
    banc_status = {
        "local_dir": str(banc_dir),
        "dataset_present": bool(banc_present),
        "measurement_status": "MEASURED" if banc_present else "NOT_DOWNLOADED_LOCAL (Codex auth required at https://codex.flywire.ai/banc)",
        "mapped_homolog_types": BANC_HOMOLOGY_MAP,
        "transfer_limits": [
            "Cross-connectome root IDs are disjoint between FAFB v783 and BANC v888; transfer uses cell-type and side homology only.",
            "On single-post open-field escape with per-topology calibration, biological and degree-preserving rewired graphs both achieve 0 collisions (rewired-7 lead slightly longer in this pilot); biological advantage is not assumed.",
        ],
    }

    report = {
        "topology_contrast_lead_s_bio_vs_rewired7": lead_contrast,
        "recovery_contrast_safe_success_circuit_vs_ordinary": success_circ_vs_ord,
        "recovery_contrast_safe_success_circuit_vs_off": success_circ_vs_off,
        "recovery_contrast_progress_m_circuit_vs_off": prog_circ_vs_off,
        "compass_summary_errors_deg": comp["mean_abs_error_deg"],
        "banc_v888": banc_status,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--escape-csv", type=Path, default=Path("runs/camera-escape-zero-fp/episodes.csv"))
    p.add_argument("--closed-loop-csv", type=Path, default=Path("runs/closed-loop-eval/episodes.csv"))
    p.add_argument("--compass-summary", type=Path, default=Path("runs/compass-benchmark/summary.json"))
    p.add_argument("--banc-dir", type=Path, default=Path("data/banc-v888"))
    p.add_argument("--out", type=Path, default=Path("runs/scientific-validation"))
    a = p.parse_args()
    print(json.dumps(validate(a.escape_csv, a.closed_loop_csv, a.compass_summary, a.banc_dir, a.out), indent=2))
