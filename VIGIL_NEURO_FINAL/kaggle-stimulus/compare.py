"""Compare central LPLC2 response modes on Zhao et al.'s ten published stimulus clips."""
import argparse
import csv
import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import torch

REV = "50c7c8f3f74052b2562ccba967c9f1032f37405b"
BASE = ("https://raw.githubusercontent.com/zhaojunyu01/EMD-LPLC2-GF-model/" + REV +
        "/code/1_Response-Characteristics-Of-Single-LPLC2-Neurons-To-Different-Stimuli/Stimuli/")
CLIPS = ("Looming_50ms","Square_expanded-motion","Bar_horizontal-motion","Edge_horizontal-motion",
         "Bar_vertical-motion","Edge_vertical-motion","Grating_horizontal-motion",
         "Grating_vertical-motion","Cross_centrifugal-motion","Cross_concentric-motion")


def response(path, device):
    data = loadmat(path, variable_names=["patt"])["patt"]
    sampled = np.ascontiguousarray(data[::4,::4,:].transpose(2,0,1), dtype=np.float64)
    del data
    movie = torch.as_tensor(sampled,device=device)*255
    # MATLAB floor(end/2) is a one-based output index. conv2(...,'same')
    # with an even 100-pixel kernel takes the full convolution starting at 50.
    cy,cx = [(n-1)//2-1 for n in movie.shape[1:]]
    y,x = cy-49,cx-49
    if min(x,y)<0 or y+101>movie.shape[1] or x+101>movie.shape[2]:
        raise ValueError("Unexpected stimulus dimensions")
    crop = movie[:,y:y+101,x:x+101]
    high = torch.zeros_like(crop[0])
    delay = torch.zeros_like(high)
    records = []
    for i in range(1,len(crop)):
        high = (25/26)*(high+crop[i]-crop[i-1])
        off = (0.05-high).clamp_min(0)
        lag = delay*(5/6)+off/6
        horizontal = lag[:-1,:-1]*off[:-1,1:] - off[:-1,:-1]*lag[:-1,1:]
        vertical = lag[:-1,:-1]*off[1:,:-1] - off[:-1,:-1]*lag[1:,:-1]
        arms = torch.stack((-horizontal[30:70,:50].sum(),horizontal[30:70,50:].sum(),
                            -vertical[:50,30:70].sum(),vertical[50:,30:70].sum()))
        active = torch.where(arms>2,arms,0)
        records.append(torch.stack((active.sum(),active.prod())))
        delay = lag
    return torch.stack(records).cpu().numpy()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path)
    p.add_argument("--out",type=Path,default=Path("/kaggle/working"))
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    trace={}; hashes={}
    for name in CLIPS:
        if a.input:
            path=a.input/(name+".mat")
        else:
            path=Path(tempfile.gettempdir())/(name+".mat")
            urllib.request.urlretrieve(BASE+name+".mat",path)
        hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        trace[name]=response(path,a.device)
        print(name,len(trace[name]),flush=True)
    maxima=np.max(np.vstack(list(trace.values())),axis=0)
    if np.any(maxima<=0): raise ValueError("No reference response in one or more modes")
    with (a.out/"response_traces.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["stimulus","transition","additive_normalized","multiplicative_normalized"])
        for name,v in trace.items():
            w.writerows((name,i,*row) for i,row in enumerate(v/maxima,1))
    summary=dict(revision=REV, device=torch.cuda.get_device_name(0) if a.device=="cuda" else "CPU",
                 sha256=hashes, normalized_peak={k:(v.max(axis=0)/maxima).tolist() for k,v in trace.items()},
                 method="Central-pixel PyTorch implementation of the published additive and multiplicative EMD-LPLC2 model. No parameter fitting. Independent parity with MATLAB outputs remains untested.")
    (a.out/"comparison.json").write_text(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
