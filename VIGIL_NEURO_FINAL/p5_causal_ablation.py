"""P5 Section 9 Causal Diagnostic Ablation (Configurations A-G) on Matched G08 & R05 Worlds.

Measures isolated effects of each P5 architectural correction vs the frozen V1 baseline
and disaggregated outcome metrics across G08 and R05 without modifying held-out datasets
or premature retraining.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from blindspot_dataset import ORACLE_KEYS, SectorBelief
from camera_compass import bearing
from camera_escape import Network, visual_looming
from closed_loop_benchmark import R05_SCENARIOS, _load_neuro_cover_l, build_scenario
from compass_benchmark import ConnectomeCompass, load_compass_connectome
from evaluation_metrics_v2 import aggregate_episodes, classify_episode
from recovery_policy import choose_recovery_action
from safety_arbitrator_v2 import (
    ActionFeasibilityChecker,
    ArbitrationConfig,
    ObservationIntegrityMonitor,
    compute_arbitrated_risk,
    fuse_biological_visual_threat,
    select_protected_action_p5,
)
from scale_dataset import ACTIONS, extract_features
from spatial_belief_v2 import FactorizedSpatialBelief
from t4_t5_vision import DualPathwayT4T5Vision


CONFIGS = (
    "A_V1_Frozen",
    "B_Compass_IMU_Only",
    "C_Visual_Fusion_Only",
    "D_Risk_Arbitration_Only",
    "E_Action_Feasibility_Only",
    "F_Revised_Utility_Only",
    "G_Full_P5_Architecture",
)


def _predict_all(ncl_models, feats):
    ncl_models["m4"].step_core(feats)
    ncl_models["m8"].step_core(feats)
    p4 = {a: ncl_models["m4"].predict_action(a) for a in ACTIONS}
    p8 = {a: ncl_models["m8"].predict_action(a) for a in ACTIONS}
    return p4, p8


def run_causal_episode(scenario: str, config_name: str, seed: int, escape_assets: dict, compass_model: dict, env_mode: str, weights_path=None) -> dict:
    # Config flags
    use_p5_compass = config_name in ("B_Compass_IMU_Only", "G_Full_P5_Architecture")
    use_p5_vision = config_name in ("C_Visual_Fusion_Only", "G_Full_P5_Architecture")
    use_p5_arb = config_name in ("D_Risk_Arbitration_Only", "G_Full_P5_Architecture")
    use_p5_feas = config_name in ("E_Action_Feasibility_Only", "G_Full_P5_Architecture")
    use_p5_util = config_name in ("F_Revised_Utility_Only", "G_Full_P5_Architecture")

    belief_v1 = SectorBelief(max_age_s=0.6)
    belief_v2 = FactorizedSpatialBelief(max_age_s=0.6)
    vis_v2 = DualPathwayT4T5Vision()
    integrity_mon = ObservationIntegrityMonitor(max_freeze_probes=2)
    checker = ActionFeasibilityChecker()
    cfg = ArbitrationConfig()

    net = Network(escape_assets["npz"], escape_assets["labels"])
    thresh = escape_assets["threshold"]
    compass = ConnectomeCompass(compass_model)
    ncl_models = _load_neuro_cover_l(weights_path)

    vstate = None
    ncl_vstate = None
    last_cap = None
    ncl_last_cap = None
    anchor = None
    latched_escape = False
    escape_hold = 0
    last_scan_step = -20
    prev_action = "brake"
    actions_seq = []
    abstained = False
    whisker_events = 0
    prev_whisker = False
    outage_steps = 0
    u_start = None
    u_end = 0.0

    vis_out_latest = {"raw_lplc2": 0.0, "residual_lplc2": 0.0, "residual_lc4": 0.0, "lateral_imbalance": 0.0}

    with build_scenario(scenario, seed, env_mode=env_mode) as sim:
        obs = sim.observe()
        for step in range(80):
            if not ORACLE_KEYS.isdisjoint(obs):
                raise AssertionError("Oracle field leaked into observation")

            rep = integrity_mon.step(obs, imu_enabled=use_p5_compass or (config_name == "G_Full_P5_Architecture") or (config_name == "C_Visual_Fusion_Only"))
            snap_v1 = belief_v1.update(obs)
            w_any = bool(obs["whisker_left"] or obs["whisker_right"])
            if w_any and not prev_whisker:
                whisker_events += 1
            prev_whisker = w_any
            if not rep.camera_healthy:
                outage_steps += 1

            # Update compass cleanly
            dyaw_compass = 0.0
            if use_p5_compass and rep.delta_yaw_rad is not None:
                dyaw_compass = rep.delta_yaw_rad
            if obs["capture_s"] != last_cap:
                dt_bio = 0.1 if last_cap is None else max(0.05, obs["capture_s"] - last_cap)
                b_ang = bearing(obs["rgb"])
                if anchor is None and b_ang is not None:
                    anchor = b_ang
                cue = None if (b_ang is None or anchor is None) else (b_ang - anchor)
                _, comp_conc = compass.step(dyaw_compass, cue)

                if use_p5_vision:
                    vis_out_latest = vis_v2.step(obs["rgb"], dt_bio, rep.imu_gyro_z, rep.imu_gyro_y)
                else:
                    lplc2_d, lc4_d, vstate = visual_looming(obs["rgb"], vstate, dt_bio)
                    bio_score = net.update(lplc2_d, lc4_d)
                    if bio_score >= thresh:
                        if env_mode == "real_world":
                            escape_hold = 8
                        else:
                            latched_escape = True
                    # Legacy V1 compass bump trigger ONLY when use_p5_compass is False
                    if not use_p5_compass and comp_conc < 0.25:
                        if env_mode == "real_world":
                            escape_hold = max(escape_hold, 4)
                        else:
                            latched_escape = True
                last_cap = obs["capture_s"]
            else:
                comp_conc = belief_v2.compass_conc

            # Update spatial belief v2 for world-referenced evaluation
            snap_v2 = belief_v2.update(
                obs,
                delta_yaw_rad=rep.delta_yaw_rad,
                heading_unc_delta_rad=rep.heading_uncertainty_delta_rad,
                compass_conc=comp_conc,
            )
            if u_start is None:
                u_start = snap_v2["task_uncertainty"]
            u_end = snap_v2["task_uncertainty"]

            # M4 & M8 forward pass
            m4_preds = m8_preds = None
            if ncl_models is not None:
                dt_ncl = 0.1 if ncl_last_cap is None else max(0.05, obs["time_s"] - ncl_last_cap)
                feats, ncl_vstate = extract_features(obs, snap_v1, ncl_vstate, dt_ncl)
                ncl_last_cap = obs["time_s"]
                m4_preds, m8_preds = _predict_all(ncl_models, feats)

            if config_name == "G_Full_P5_Architecture":
                action, rsn = select_protected_action_p5(
                    obs,
                    snap_v2,
                    rep,
                    vis_out_latest,
                    sim.evaluator()["speed_m_s"],
                    m4_preds if step >= 8 else None,
                    m8_preds if step >= 8 else None,
                    prev_action,
                    checker,
                    cfg,
                    use_visual_fusion=True,
                    use_risk_arbitration=True,
                    use_feasibility_check=True,
                    use_revised_utility=True,
                )
                if "abstain" in rsn:
                    abstained = True
            else:
                # Configs A..F: run frozen V1 structure with exactly one isolated P5 replacement active
                if not use_p5_vision:
                    if env_mode == "real_world" and escape_hold > 0 and obs["obstacle_center"] < 0.08:
                        escape_hold -= 1
                    looming_alert = latched_escape or (escape_hold > 0)
                else:
                    looming_alert, _, _ = fuse_biological_visual_threat(vis_out_latest, obs, rep)

                if ncl_models is not None and step >= 8:
                    r4 = m4_preds["forward"]["contact_prob"]
                    r8 = m8_preds["forward"]["contact_prob"]
                    if use_p5_arb:
                        if compute_arbitrated_risk(r4, r8, cfg.kappa_disagree) >= cfg.risk_limit_eps:
                            looming_alert = True
                    else:
                        if r4 >= ncl_models["m4"].threshold and r8 >= ncl_models["m8"].threshold:
                            looming_alert = True

                if use_p5_feas:
                    feas_map = checker.feasible_actions(("forward", "slow", "left", "right", "brake"), snap_v2, obs, rep, sim.evaluator()["speed_m_s"])
                    if w_any:
                        action, rsn = choose_recovery_action(obs, snap_v1, looming_alert=False)
                    elif looming_alert:
                        action, rsn = "brake", "looming_alert"
                    elif not feas_map["forward"][0]:
                        action, rsn = "brake", f"abstain:{feas_map['forward'][1]}"
                    else:
                        action, rsn = "forward", "feas_open"
                else:
                    action, rsn = choose_recovery_action(obs, snap_v1, looming_alert=looming_alert)

                if "abstain" in rsn:
                    abstained = True
                elif ncl_models is not None and step >= 8 and snap_v1["camera_healthy"] and not w_any:
                    if use_p5_util:
                        action, u_rsn = select_protected_action_p5(
                            obs, snap_v2 if use_p5_feas else snap_v1, rep,
                            vis_out_latest, sim.evaluator()["speed_m_s"], m4_preds, m8_preds, prev_action, checker, cfg,
                            use_visual_fusion=False, use_risk_arbitration=use_p5_arb, use_feasibility_check=use_p5_feas, use_revised_utility=True,
                        )
                        if looming_alert and action == "forward":
                            action = "brake"
                        if "abstain" in u_rsn:
                            abstained = True
                    else:
                        states = snap_v1["sector_states"]
                        mid = len(states) // 2
                        lc = all(s == "observed_free" for s in states[:mid])
                        rc = all(s == "observed_free" for s in states[mid + 1 :])
                        if action == "brake" and looming_alert and step - last_scan_step >= 4:
                            cands = []
                            for sa, fr in (("left", lc), ("right", rc)):
                                if fr and m4_preds[sa]["contact_prob"] < ncl_models["m4"].threshold and m8_preds[sa]["recover_prob"] >= 0.50:
                                    cands.append((m8_preds[sa]["recover_prob"] + 0.5 * m8_preds[sa]["progress"], sa))
                            if cands:
                                cands.sort(reverse=True)
                                action = cands[0][1]
                                last_scan_step = step
                        elif action == "forward" and snap_v1["unknown_or_stale_fraction"] > 0.35 and step - last_scan_step >= 12:
                            cands = [
                                (m8_preds[sa]["recover_prob"], sa)
                                for sa, fr in (("left", lc), ("right", rc))
                                if fr and m4_preds[sa]["contact_prob"] < ncl_models["m4"].threshold and m8_preds[sa]["recover_prob"] >= 0.55
                            ]
                            if cands:
                                cands.sort(reverse=True)
                                action = cands[0][1]
                                last_scan_step = step

            integrity_mon.record_action(action, outage_active=(not rep.camera_healthy))
            prev_action = action
            actions_seq.append(action)
            obs = sim.step(action)

        ev = sim.evaluator()
    rec = classify_episode(
        scenario=scenario,
        x_m=ev["x_m"],
        y_m=ev["y_m"],
        body_collisions=int(ev["body_contact_events"]),
        whisker_events=whisker_events,
        abstained=abstained,
        outage_steps=outage_steps,
        actions_seq=actions_seq,
        u_task_start=float(u_start or 0.0),
        u_task_end=float(u_end),
    )
    rec["config"] = config_name
    rec["seed"] = seed
    rec["suite"] = "r05" if scenario in R05_SCENARIOS else "g08"
    return rec


def run_ablation(out_dir: Path, seeds: int = 4, weights_path=None) -> dict:
    escape_dir = Path("runs/graph-controls/escape")
    ann_path = Path("runs/connectome/escape/annotations.csv")
    compass_dir = Path("runs/connectome/compass")
    with ann_path.open() as f:
        types = {r["root_id"]: r["primary_type"] for r in csv.DictReader(f)}
    ids = (escape_dir / "nodes.txt").read_text().splitlines()
    escape_assets = {"npz": escape_dir / "biological.npz", "labels": np.array([types[i] for i in ids]), "threshold": 0.0238}
    compass_model = load_compass_connectome(compass_dir)

    g08_scenarios = ("open_healthy", "looming_post", "near_miss_post", "freeze_outage_clear", "blocked_outage", "occluded_hazard")
    all_rows = []
    report = {"seeds_per_scenario": seeds, "g08": {}, "r05": {}}
    for cfg_name in CONFIGS:
        g08_rows, r05_rows = [], []
        for sc in g08_scenarios:
            for s in range(seeds):
                g08_rows.append(run_causal_episode(sc, cfg_name, s, escape_assets, compass_model, "rover_sim", weights_path))
        for sc in R05_SCENARIOS:
            for s in range(seeds):
                r05_rows.append(run_causal_episode(sc, cfg_name, s, escape_assets, compass_model, "real_world", weights_path))
        report["g08"][cfg_name] = aggregate_episodes(g08_rows)
        report["r05"][cfg_name] = aggregate_episodes(r05_rows)
        all_rows.extend(g08_rows + r05_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "episodes_disaggregated.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    (out_dir / "ablation_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("runs/p5-causal-ablation"))
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--weights", type=Path, default=None)
    a = p.parse_args()
    rep = run_ablation(a.out, seeds=a.seeds, weights_path=a.weights)
    for suite in ("g08", "r05"):
        print(f"\n=== {suite.upper()} CAUSAL DIAGNOSTIC ABLATION ===")
        for cfg_name, m in rep[suite].items():
            print(
                f"{cfg_name:26s} | col={m['collision_episodes']} "
                f"CI95=[{m['collision_rate_clopper_pearson_95ci'][0]:.3f},{m['collision_rate_clopper_pearson_95ci'][1]:.3f}] | "
                f"goal={m['collision_free_goal_completion_rate']:.2f} | "
                f"stall={m['permanent_stall_rate']:.2f} | "
                f"prog={m['mean_progress_m']:.3f}m | sw={m['mean_action_switches']:.1f}"
            )
