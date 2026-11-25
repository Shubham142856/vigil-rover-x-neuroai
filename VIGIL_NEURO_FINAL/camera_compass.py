"""Camera-anchored relative compass for an in-place turn and a single red landmark."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from heading_memory import advance, decode, wrap
from rover_sim import RoverSim


def bearing(rgb):
    """Image displacement to the right in radians, using the camera vertical FOV."""
    r,g,b=rgb.astype(float).transpose(2,0,1)
    mask=(r>60)&(r>1.7*g)&(r>1.7*b)
    _,cols=np.nonzero(mask)
    if len(cols)<6 or cols.min()==0 or cols.max()==rgb.shape[1]-1:
        return None
    focal=rgb.shape[0]/(2*np.tan(np.radians(66)/2))
    return float(np.arctan((cols.mean()-(rgb.shape[1]-1)/2)/focal))


def fresh_heading(packet,previous_capture):
    if not packet["frame_valid"] or packet["capture_s"]==previous_capture:
        return None
    if packet["time_s"]-packet["capture_s"]>0.051:
        return None
    return bearing(packet["rgb"])


def run(out,seeds):
    if seeds<1: raise ValueError("seeds must be positive")
    out.mkdir(parents=True,exist_ok=True)
    angles=np.linspace(-np.pi,np.pi,72,endpoint=False)
    checkpoints={3.95:[],5.0:[]}
    with (out/"trace.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["seed","time_s","capture_s","camera_update","truth_rad","compass_rad","odom_rad"])
        for seed in range(seeds):
            rng=np.random.default_rng(seed)
            mass=np.exp(10*np.cos(angles));mass/=mass.sum()
            odom=0.
            with RoverSim(scene="post",seed=seed,fault="freeze",fault_start_s=1,fault_end_s=4) as sim:
                initial=sim.observe()
                anchor=bearing(initial["rgb"])
                if anchor is None: raise ValueError("Starting landmark unavailable")
                previous=initial["capture_s"]
                for step in range(120):
                    sign=1 if (step+6)//12%2==0 else -1
                    obs=sim.step("left" if sign==1 else "right")
                    delta=(sign+.09+rng.normal(0,.05))/20
                    odom=wrap(odom+delta);mass=advance(mass,delta)
                    cue=fresh_heading(obs,previous)
                    previous=obs["capture_s"]
                    if cue is not None:
                        mass*=np.exp(4*np.cos(angles-wrap(cue-anchor)))
                        mass/=mass.sum()
                    estimate=decode(mass,angles)
                    truth=sim.evaluator()["yaw_rad"]
                    t=obs["time_s"]
                    w.writerow([seed,t,previous,cue is not None,truth,estimate,odom])
                    if t in checkpoints:
                        checkpoints[t].append(np.degrees(np.abs(wrap(np.array([estimate,odom])-truth))))
    result={str(t):dict(zip(["compass_error_deg","odometry_error_deg"],np.mean(v,axis=0))) for t,v in checkpoints.items()}
    report=dict(seeds=seeds,freeze_seconds=[1,4],mean_absolute_errors=result,
                scope="Relative heading for in-place turns with one red post. Initialized from the first camera frame. Seed varies sensor brightness and odometry noise.")
    (out/"summary.json").write_text(json.dumps(report,indent=2))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--seeds",type=int,default=8)
    args=p.parse_args(); print(run(args.out,args.seeds))
