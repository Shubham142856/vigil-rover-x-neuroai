"""Canonical circular heading comparator with simulated landmark dropout."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from rover_sim import RoverSim


def wrap(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi


def decode(mass, angles):
    vector = np.sum(mass*np.exp(1j*angles))
    if abs(vector)<1e-9:
        return None
    return float(np.angle(vector))


def advance(mass, delta):
    bins = delta*len(mass)/(2*np.pi)
    whole = int(np.floor(bins))
    frac = bins-whole
    return (1-frac)*np.roll(mass,whole)+frac*np.roll(mass,whole+1)


def trial(seed, writer):
    rng=np.random.default_rng(seed)
    angles=np.linspace(-np.pi,np.pi,72,endpoint=False)
    mass=np.exp(10*np.cos(angles))
    mass/=mass.sum()
    odom=0.
    errors=[]
    with RoverSim(scene="open",seed=seed) as sim:
        for step in range(140):
            action="left" if step<90 else "right"
            sim.step(action)
            truth=sim.evaluator()["yaw_rad"]
            delta=((1 if action=="left" else -1)+0.09+rng.normal(0,.05))/20
            odom=wrap(odom+delta)
            mass=advance(mass,delta)
            visible=step<20 or step>=100
            if visible and step%2==0:
                # Controlled absolute landmark observation synthesized from evaluator yaw.
                landmark=wrap(truth+rng.normal(0,.06))
                mass*=np.exp(4*np.cos(angles-landmark))
                mass/=mass.sum()
            estimate=decode(mass,angles)
            if estimate is None:
                raise ValueError("Compass lost directional concentration")
            err=np.degrees(abs(wrap(estimate-truth)))
            odom_err=np.degrees(abs(wrap(odom-truth)))
            writer.writerow([seed,(step+1)/20,visible,truth,estimate,odom,err,odom_err])
            errors.append((err,odom_err))
    return errors


def run(out,seeds):
    if seeds<1: raise ValueError("At least one seed required")
    out.mkdir(parents=True,exist_ok=True)
    with (out/"heading.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["seed","time_s","landmarks_available","truth_rad","compass_rad","odometry_rad","compass_error_deg","odometry_error_deg"])
        errors=np.asarray([trial(s,w) for s in range(seeds)])
    report={"seeds":seeds,"bias_rad_s":0.09,"model":"72-bin circular filter; synthetic landmark observations; commanded turns with biased noisy odometry."}
    for k,idx in [("last_blackout_sample",99),("one_second_after_return",119)]:
        report[k]=dict(zip(("compass_mean_abs_deg","odometry_mean_abs_deg"),errors[:,idx,:].mean(axis=0)))
    (out/"summary.json").write_text(json.dumps(report,indent=2))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--seeds",type=int,default=8)
    a=p.parse_args();print(run(a.out,a.seeds))
