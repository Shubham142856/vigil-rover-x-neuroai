"""Controlled numerical check of signal propagation on biological and rewired graphs."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.sparse import diags, load_npz


def response(matrix, sources, targets, steps=60, pulse=10):
    if not sources.any() or not targets.any():
        raise ValueError("Stimulus and readout must each contain identified cells")
    totals=np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    weights=diags(1/np.maximum(1,totals)) @ matrix
    state=np.zeros(matrix.shape[0])
    trace=[]
    for t in range(steps):
        drive=sources.astype(float) if t<pulse else 0.
        state+=0.2*(np.tanh(drive+0.8*(weights@state))-state)
        trace.append(float(state[targets].mean()))
    return np.array(trace)


def run(graph,annotations,source,target,out):
    ids=(graph/"nodes.txt").read_text().splitlines()
    with annotations.open() as f:
        types={r["root_id"]:r["primary_type"] for r in csv.DictReader(f)}
    labels=np.array([types[i] for i in ids])
    out.mkdir(parents=True,exist_ok=True)
    peaks={}
    with (out/"traces.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["topology","step","mean_readout"])
        for file in sorted(graph.glob("*.npz")):
            trace=response(load_npz(file),labels==source,labels==target)
            if not np.isfinite(trace).all():raise ValueError("Nonfinite dynamics")
            peaks[file.stem]=float(trace.max())
            w.writerows((file.stem,i,x) for i,x in enumerate(trace,1))
    if "biological" not in peaks:raise ValueError("Missing biological matrix")
    report=dict(source=source,target=target,normalized_peak=peaks,model="Unsigned row-normalized count matrix; dimensionless rate update, leak 0.2, recurrent gain 0.8, tanh. Common 10-step unit pulse. Numerical wiring assay only; signs, physical time constants and sensory encoding unvalidated.")
    (out/"summary.json").write_text(json.dumps(report,indent=2))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("graph","annotations","out"):p.add_argument("--"+key,type=Path,required=True)
    for key in ("source","target"):p.add_argument("--"+key,required=True)
    print(run(**vars(p.parse_args())))
