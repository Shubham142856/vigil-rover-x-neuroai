"""Spatial motion sanity check for centered synthetic stimuli, before circuit fitting."""
import argparse
import json
from pathlib import Path
import numpy as np


def stimulus(kind):
    y,x=np.mgrid[-48:48,-48:48].astype(float)
    images=[]
    for t in range(20):
        radius={"approach":10+.55*t,"recede":20-.55*t}.get(kind,15)
        dx=x-(t-9.5 if kind=="translate" else 0)
        distance=np.hypot(dx,y)
        if kind=="rotate":
            angle=np.arctan2(y,x)-.045*t
            image=128+80*np.sin(4*angle)*np.exp(-((distance-20)/8)**2)
        else:
            image=45+160/(1+np.exp((radius-distance)/1.3))
        images.append(image.astype(np.uint8))
    return np.stack(images)


def radial_response(frames):
    """Gradient-weighted outward normal-flow projection in pixels per frame."""
    f=frames.astype(float)
    avg=(f[1:]+f[:-1])/2
    gy,gx=np.gradient(avg,axis=(1,2))
    y,x=np.mgrid[:f.shape[1],:f.shape[2]]
    y=y-f.shape[1]//2; x=x-f.shape[2]//2
    radius=np.maximum(np.hypot(x,y),1)
    power=(gx*gx+gy*gy).sum(axis=(1,2))
    return (-np.diff(f,axis=0)*(gx*x+gy*y)/radius).sum(axis=(1,2))/np.maximum(power,1)


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    clips={k:stimulus(k) for k in ("approach","recede","translate","rotate")}
    np.savez_compressed(out/"stimuli.npz",**clips)
    scores={k:float(np.median(radial_response(v))) for k,v in clips.items()}
    (out/"median_radial_px_per_frame.json").write_text(json.dumps(scores,indent=2))
    return scores


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",required=True,type=Path)
    print(run(p.parse_args().out))
