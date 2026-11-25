"""G02 audit: compare min_synapses 1 vs 5 on unfiltered FAFB v783 connections + NT sign coverage."""
import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from extract_graph import digest


def audit_unfiltered(connections: Path, connectome_dir: Path, out: Path, version="FAFB-v783"):
    """Single-pass pool over unfiltered FAFB connections comparing thresholds 1 and 5."""
    groups = {}
    all_selected = set()
    for name in ("escape", "compass"):
        ids = {line.strip() for line in (connectome_dir / name / "ids.txt").read_text().splitlines() if line.strip()}
        with (connectome_dir / name / "annotations.csv").open() as f:
            signs = {r["root_id"]: r["predicted_nt"] for r in csv.DictReader(f)}
        groups[name] = {"ids": ids, "signs": signs, "pairs": defaultdict(int), "rows": 0}
        all_selected.update(ids)

    opener = gzip.open if connections.suffix.lower() == ".gz" else open
    with opener(connections, "rt", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            a, b = row["pre_root_id"], row["post_root_id"]
            if a not in all_selected or b not in all_selected:
                continue
            n = int(row["syn_count"])
            for g in groups.values():
                if a in g["ids"] and b in g["ids"]:
                    g["rows"] += 1
                    if n > 0:
                        g["pairs"][a, b] += n

    conn_sha = digest(connections)
    out.mkdir(parents=True, exist_ok=True)
    report = {"version": version, "connection_file": str(connections.name), "connection_sha256": conn_sha, "circuits": {}}

    for name, g in groups.items():
        ids = sorted(g["ids"])
        sign_counts = Counter(g["signs"].get(i, "") or "UNKNOWN" for i in ids)
        known_sign = sum(v for k, v in sign_counts.items() if k != "UNKNOWN")
        thresh_reports = {}
        for minimum in (1, 5):
            edges = sorted((a, b, n) for (a, b), n in g["pairs"].items() if n >= minimum)
            connected = {k for a, b, _ in edges for k in (a, b)}
            sub_dir = out / name / f"min-{minimum}"
            sub_dir.mkdir(parents=True, exist_ok=True)
            with (sub_dir / "edges.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["pre_id", "post_id", "synapse_count"])
                w.writerows(edges)
            (sub_dir / "nodes.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
            manifest = {
                "version": version,
                "connection_sha256": conn_sha,
                "selected_nodes": len(ids),
                "retained_edges": len(edges),
                "total_synapses": int(sum(n for _, _, n in edges)),
                "source_rows": g["rows"],
                "min_synapses": minimum,
                "isolated_nodes": len(set(ids) - connected),
            }
            (sub_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
            thresh_reports[f"min_{minimum}"] = manifest

        report["circuits"][name] = {
            "sign_coverage": {
                "total_nodes": len(ids),
                "known_nt_nodes": known_sign,
                "unknown_nt_nodes": len(ids) - known_sign,
                "known_fraction": round(known_sign / len(ids), 4),
                "by_nt": dict(sign_counts),
            },
            **thresh_reports,
        }

    (out / "unfiltered_sensitivity.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--connections", type=Path, default=Path("data/fafb-v783/connections_princeton_no_threshold.csv.gz"))
    p.add_argument("--connectome", type=Path, default=Path("runs/connectome"))
    p.add_argument("--out", type=Path, default=Path("runs/connectome/unfiltered-audit"))
    args = p.parse_args()
    print(json.dumps(audit_unfiltered(args.connections, args.connectome, args.out), indent=2))
