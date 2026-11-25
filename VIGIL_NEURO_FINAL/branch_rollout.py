"""Evaluate bounded action branches from the same simulator and camera scheduler state."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import pybullet as pb

from rover_sim import RoverSim


def compare_actions(sim,actions,steps):
    if steps<=0 or not actions or len(set(actions))!=len(actions):
        raise ValueError("Supply distinct actions and a positive decision horizon")
    fields=("tick","speed","turn","events","touch","queue","packet","last_center","growth")
    saved={name:deepcopy(getattr(sim,name)) for name in fields}
    initial=sim.evaluator()
    state=pb.saveState(physicsClientId=sim.client)

    def restore():
        pb.restoreState(state,physicsClientId=sim.client)
        for name,value in saved.items():
            setattr(sim,name,deepcopy(value))

    results={}
    try:
        for action in actions:
            restore()
            hit=initial["body_contact"]
            for index in range(steps+20):
                sim.step(action if index<steps else "brake")
                hit=hit or sim.evaluator()["body_contact"]
            end=sim.evaluator()
            results[action]=dict(body_contact=bool(hit),forward_progress_m=end["x_m"]-initial["x_m"],
                                 heading_change_rad=end["yaw_rad"]-initial["yaw_rad"],
                                 final_speed_m_s=end["speed_m_s"])
    finally:
        try: restore()
        finally: pb.removeState(state,physicsClientId=sim.client)
    return results


def run(out):
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w") as f:
        for scene in ("open","post","wall","left","right"):
            for fault in ("none","freeze","delay"):
                with RoverSim(scene=scene,fault=fault) as sim:
                    for step in range(31):
                        if step in (0,10,20,30):
                            labels=compare_actions(sim,["forward","slow","brake","left","right"],24)
                            record=dict(world=scene,fault=fault,control_tick=step,horizon_s=1.2,terminal_brake_s=1,labels=labels)
                            f.write(json.dumps(record)+"\n")
                        sim.step("forward")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True,type=Path)
    run(p.parse_args().out)
