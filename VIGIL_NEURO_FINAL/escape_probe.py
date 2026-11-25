"""Probe direct anatomical inputs to annotated giant fibers using a normalized unit drive."""
import argparse
import csv
import json
from pathlib import Path


def probe(cells, edges):
    channels = ("LC4", "LPLC2")
    inputs = {i: {c: {} for c in channels} for i, (kind, _) in cells.items() if kind == "DNp01"}
    for pre, post, count in edges:
        channel = cells[pre][0]
        if post in inputs and channel in channels:
            sources = inputs[post][channel]
            sources[pre] = sources.get(pre, 0) + count
    results = []
    for target, groups in inputs.items():
        row = {"target": target, "side": cells[target][1]}
        total = sum(sum(group.values()) for group in groups.values())
        for channel, sources in groups.items():
            weight = sum(sources.values())
            row[channel+"_contacts"] = weight
            row[channel+"_sources"] = len(sources)
            row[channel+"_unit_drive"] = weight / total if total else None
        results.append(row)
    return results


def run(graph, out):
    with (graph/"annotations.csv").open() as f:
        cells = {r["root_id"]: (r["primary_type"],r["side"]) for r in csv.DictReader(f)}
    with (graph/"edges.csv").open() as f:
        results = probe(cells, ((r["pre_id"],r["post_id"],int(r["synapse_count"])) for r in csv.DictReader(f)))
    report = {"method": "One-step anatomical projection. Activate each source class alone at unit amplitude and divide by combined retained LC4 and LPLC2 synapse count for that target.",
              "limits": "Unsigned structural weighting. No measured conductance, temporal dynamics, retinal drive or other presynaptic classes included.",
              "source_manifest": json.loads((graph/"manifest.json").read_text()),"targets": results}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(results,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    run(args.graph,args.out)
