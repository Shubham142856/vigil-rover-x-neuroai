"""Compare reference and GPU responses on identical stimuli and transitions."""
import csv
import json
import math
from pathlib import Path
import sys

def load(path):
    with path.open(encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    values={(r["stimulus"],int(r["transition"])): tuple(float(r[k]) for k in ("additive_normalized","multiplicative_normalized")) for r in rows}
    if len(values)!=len(rows) or len(values)!=4990 or not all(math.isfinite(x) for pair in values.values() for x in pair):
        raise ValueError(f"Incomplete or invalid trace: {path}")
    return values

if __name__=="__main__":
    root=Path(sys.argv[1])
    ref=load(root/"octave_reference.csv")
    gpu=load(Path(sys.argv[2]))
    if ref.keys()!=gpu.keys(): raise ValueError("Mismatched stimulus/transition keys")
    worst=max((abs(x-y),clip,t,mode) for (clip,t),v in ref.items() for mode,(x,y) in zip(("additive","multiplicative"),zip(v,gpu[clip,t])))
    report=dict(max_absolute_error=worst[0],stimulus=worst[1],transition=worst[2],mode=worst[3],tolerance=1e-12,passed=worst[0]<=1e-12)
    (root/"parity.json").write_text(json.dumps(report,indent=2))
    print(report)
