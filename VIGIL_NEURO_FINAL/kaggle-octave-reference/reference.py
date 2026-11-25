"""Run the upstream MATLAB-language functions in GNU Octave, one clip and mode at a time."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

REV = "50c7c8f3f74052b2562ccba967c9f1032f37405b"
BASE = f"https://raw.githubusercontent.com/zhaojunyu01/EMD-LPLC2-GF-model/{REV}/code/1_Response-Characteristics-Of-Single-LPLC2-Neurons-To-Different-Stimuli/"
WORK = Path("/kaggle/working")
SRC = Path("/kaggle/temp/octave-source")
CLIPS = ("Looming_50ms", "Square_expanded-motion", "Bar_horizontal-motion", "Edge_horizontal-motion",
         "Bar_vertical-motion", "Edge_vertical-motion", "Grating_horizontal-motion",
         "Grating_vertical-motion", "Cross_centrifugal-motion", "Cross_concentric-motion")
SRC.mkdir(parents=True,exist_ok=True)
os.environ.update(DEBIAN_FRONTEND="noninteractive", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
subprocess.run(["apt-get","update","-qq"],check=True)
subprocess.run(["apt-get","install","-y","-qq","--no-install-recommends","octave","gnuplot-nox","fonts-freefont-otf"],check=True)
functions = ["loom.m","emd.m","OffRect.m","lplc2conv2.m"]
for name in functions:
    urllib.request.urlretrieve(BASE+name,SRC/name)
# Upstream loom uses the OFF-only interface. Drop only unused ON slots.
emd = SRC/"emd.m"
original = emd.read_bytes()
signature, rest = original.split(b"\n", 1)
assert b"Fd_On" not in rest and signature.count(b"Fd_On") == 2
signature = signature.replace(b"Fd_On, ", b"").replace(b"Fd_On_before, ", b"")
emd.write_bytes(signature+b"\n"+rest)
(WORK/"interface_patch.json").write_text(json.dumps(dict(
    upstream_sha256=hashlib.sha256(original).hexdigest(),
    executed_sha256=hashlib.sha256(emd.read_bytes()).hexdigest(),
    change="Removed unused Fd_On_before input and unassigned Fd_On output in emd.m signature; numerical body byte-identical."
),indent=2))
for name in CLIPS:
    urllib.request.urlretrieve(BASE+"Stimuli/"+name+".mat",SRC/(name+".mat"))
# Hide the upstream display; leave its numerical MATLAB source untouched.
def run_clip(name):
    for mode in (1,2):
        output=WORK/f"{name}-{mode}.txt"
        expr=(f"graphics_toolkit('gnuplot'); set(0,'defaultfigurevisible','off'); pause('off'); "
              f"s=loom('{name}.mat',{mode}); dlmwrite('{output}',s(:),'precision','%.17g');")
        log=WORK/f"{name}-{mode}.log"
        with log.open("w") as f:
            subprocess.run(["octave-cli","--no-gui","--quiet","--eval",expr],cwd=SRC,stdout=f,stderr=f,check=True)
        print(name,mode,"finished",flush=True)

with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(run_clip,CLIPS))
raw={name:np.column_stack([np.loadtxt(WORK/f"{name}-{mode}.txt") for mode in (2,1)]) for name in CLIPS}
assert all(r.shape==(499,2) and np.isfinite(r).all() for r in raw.values())
maxima=np.max(np.vstack(list(raw.values())),axis=0)
assert np.all(maxima>0)
with (WORK/"octave_reference.csv").open("w",newline="") as f:
    w=csv.writer(f);w.writerow(["stimulus","transition","additive_normalized","multiplicative_normalized"])
    for name,values in raw.items():
        w.writerows((name,i,*v) for i,v in enumerate(values/maxima,1))
(WORK/"provenance.json").write_text(json.dumps(dict(revision=REV,runtime=subprocess.check_output(["octave-cli","--version"],text=True).splitlines()[0],
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SRC.iterdir() if p.is_file()}),indent=2))
