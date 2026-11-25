"""Extract a directed induced graph from reviewed IDs and a local connection CSV/CSV.GZ.

The input IDs remain strings. The output reports pooled counts and input-file hashes.
"""
import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def op(p):
    if Path(p).suffix.lower() == ".gz":
        return gzip.open(p, "rt", newline="", encoding="utf-8-sig")
    return open(p, newline="", encoding="utf-8-sig")


def digest(p):
    checksum = hashlib.sha256()
    with open(p, "rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            checksum.update(block)
    return checksum.hexdigest()


def extract(ids, connections, out, pre, post, count, minimum, version):
    """Pool selected rows by ordered neuron pair, then apply the minimum to each sum."""
    if type(minimum) is not int or minimum < 1:
        raise ValueError("minimum must be a positive integer")
    roles = {pre, post, count}
    if len(roles) != 3:
        raise ValueError("Pre, post and count must identify three distinct columns")

    with open(ids, encoding="utf-8-sig") as stream:
        selected = {line.strip() for line in stream if line.strip()}
    if not selected or any(not (key.isascii() and key.isdigit()) for key in selected):
        raise ValueError("Supply reviewed numeric IDs, one per line")

    pairs = defaultdict(int)
    source_rows = 0
    with op(connections) as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)):
            raise ValueError(f"Duplicate column names in connection CSV: {headers}")
        if not roles.issubset(headers):
            raise ValueError(f"Column mismatch: {headers}")
        for row in reader:
            a, b = row[pre], row[post]
            if a not in selected or b not in selected:
                continue
            try:
                n = int(row[count])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid synapse count at line {reader.line_num}") from exc
            if n < 0:
                raise ValueError(f"Negative synapse count at line {reader.line_num}")
            source_rows += 1
            if n:
                pairs[a, b] += n

    edges = sorted((a, b, n) for (a, b), n in pairs.items() if n >= minimum)
    if not edges:
        raise ValueError("No retained edges; verify version, IDs and threshold")

    connected = {key for a, b, _ in edges for key in (a, b)}
    manifest = {
        "version": version,
        "connection_sha256": digest(connections),
        "ids_sha256": digest(ids),
        "selected_nodes": len(selected),
        "retained_edges": len(edges),
        "source_rows": source_rows,
        "min_synapses": minimum,
        "isolated_nodes": len(selected - connected),
        "scope": "induced directed graph; signs not inferred",
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "edges.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["pre_id", "post_id", "synapse_count"])
        writer.writerows(edges)
    (out / "nodes.txt").write_text("\n".join(sorted(selected)) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ids", "connections", "out", "pre", "post", "count", "version"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--minimum", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(extract(**vars(args)), indent=2))
