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
    use_snap = hasattr(sim, "snapshot_state") and hasattr(sim, "restore_state")
    if use_snap:
        saved = sim.snapshot_state()
    else:
        fields=("tick","speed","turn","events","touch","queue","packet","last_center","growth")
        saved={name:deepcopy(getattr(sim,name)) for name in fields}
    initial=sim.evaluator()
    state=pb.saveState(physicsClientId=sim.client)

    def restore():
        pb.restoreState(state,physicsClientId=sim.client)
        if use_snap:
            sim.restore_state(saved)
        else:
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


def evaluate_action_branches_v2(sim, actions, steps, initial_belief):
    """Counterfactual branch evaluation computing collision, progress, and true info gain G_t(a).

    G_t(a) = max(0, U_task(b_t) - U_task(b_{t+H}^(a))) * I[not hit] * I[cam_healthy_end].
    Never rewards blind turns during persistent sensor failure.
    """
    if steps <= 0 or not actions or len(set(actions)) != len(actions):
        raise ValueError("Supply distinct actions and a positive decision horizon")
    use_snap = hasattr(sim, "snapshot_state") and hasattr(sim, "restore_state")
    if use_snap:
        saved = sim.snapshot_state()
    else:
        fields = ("tick", "speed", "turn", "events", "touch", "queue", "packet", "last_center", "growth")
        saved = {name: deepcopy(getattr(sim, name)) for name in fields}
    initial_eval = sim.evaluator()
    initial_obs = sim.observe()
    u_init = float(initial_belief.task_uncertainty(float(initial_obs["time_s"])))
    pb_state = pb.saveState(physicsClientId=sim.client)

    def restore():
        pb.restoreState(pb_state, physicsClientId=sim.client)
        if use_snap:
            sim.restore_state(saved)
        else:
            for name, value in saved.items():
                setattr(sim, name, deepcopy(value))

    results = {}
    try:
        for action in actions:
            restore()
            branch_belief = initial_belief.copy()
            hit = initial_eval["body_contact"]
            obs = initial_obs
            for index in range(steps + 20):
                obs = sim.step(action if index < steps else "brake")
                hit = hit or sim.evaluator()["body_contact"]
                branch_belief.update(obs)
            end = sim.evaluator()
            u_end = float(branch_belief.task_uncertainty(float(obs["time_s"])))
            cam_ok = bool(branch_belief.camera_healthy)
            results[action] = {
                "body_contact": bool(hit),
                "forward_progress_m": float(end["x_m"] - initial_eval["x_m"]),
                "heading_change_rad": float(end["yaw_rad"] - initial_eval["yaw_rad"]),
                "final_speed_m_s": float(end["speed_m_s"]),
                "u_init": u_init,
                "u_end": u_end,
                "cam_healthy_end": cam_ok,
                "info_gain": 0.0,
            }
        # Causal treatment effect: measure uncertainty reduced by action relative to waiting ("brake")
        # so passive fault-window expiration never masquerades as active perception recovery.
        u_ref = results["brake"]["u_end"] if "brake" in results else u_init
        for act, res in results.items():
            if not res["body_contact"] and res["cam_healthy_end"] and act != "brake":
                res["info_gain"] = float(max(0.0, u_ref - res["u_end"]))
    finally:
        try:
            restore()
        finally:
            pb.removeState(pb_state, physicsClientId=sim.client)
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
