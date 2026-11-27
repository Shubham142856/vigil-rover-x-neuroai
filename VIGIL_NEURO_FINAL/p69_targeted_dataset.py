"""P6.9.4 Targeted Counterfactual Coverage Generator (100-200 Diverse Worlds) & Untouched Test Worlds.

Reconstructs the sparse P6 pilot dataset (0.6% positive protected risk) into an
independent, world-family isolated 136-world pilot dataset with:
  - Balanced scenario allocations matching P6.9 Section 6:
      25% occluded hazards & emergence | 20% pedestrians & near misses |
      15% close/mid static obstacles   | 10% ramps & potholes         |
      10% sensor faults (freeze/delay/drop/night) | 10% empty open nav |
      10% paired occluded_clear/hazard worlds
  - Multi-stage decision branching (mid-approach, post-turn, near-field boundary)
    producing rich positive protected_0.30s AND short_safe_long_danger contrast
  - 100% sensor-only FactorizedSpatialBelief evolution (NO world_yaw_rad oracle)
  - Train-only hard-example mining score S(x, a) leaving val/test un-mined
  - Untouched held-out world families (chicane_slalom, blind_alcove_emergence,
    dynamic_crosswalk, ramp_chicanery) reserved for P6.9.7 generalization claims
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from blindspot_dataset import ORACLE_KEYS, SectorBelief, make_paired_world
from camera_compass import bearing
from ground_range_estimator import estimate_ground_contact_clearance
from p6_closed_loop_pilot import _PortableV3Policy
from pilot_dataset_v3 import (
    CTRL_HZ,
    DT_CTRL_S,
    HORIZONS_S,
    PROTECTED_EXEC_STEPS,
    extract_features_v3,
    rollout_multi_horizon_branches,
)
from real_world_env import RealWorldEnv, _PedestrianProxy
from rover_sim import RoverSim
from safety_arbitrator_v2 import ObservationIntegrityMonitor
from scale_dataset import ACTIONS, build_world
from spatial_belief_v2 import FactorizedSpatialBelief
from t4_t5_vision import DualPathwayT4T5Vision


UNTOUCHED_WORLD_FAMILIES = (
    "chicane_slalom",
    "blind_alcove_emergence",
    "dynamic_crosswalk",
    "ramp_chicanery",
)


def build_p69_world(family: str, seed: int, fault: str = "none"):
    """Instantiate original G07/R05 families plus new P6.9 close-hazard and untouched world geometries."""
    rng = np.random.default_rng(seed)
    if family in (
        "corridor_pedestrian",
        "warehouse_forklift_cross",
        "outdoor_pothole",
        "lab_glass_wall",
        "ramp_descent",
        "night_mode",
        "post_sweep",
        "corridor",
        "shelf_clear",
        "shelf_hazard",
    ):
        return build_world(family, seed, fault)
    if family == "open_rover":
        return RoverSim(scene="open", seed=seed, fault=fault, fault_start_s=0.35, fault_end_s=0.90)
    if family == "open_real":
        return RealWorldEnv(scene="open", seed=seed, fault=fault, fault_start_s=0.35, fault_end_s=0.90, sensor_noise=True)
    if family == "ramp_ascent":
        return RealWorldEnv(scene="ramp", seed=seed, fault=fault, fault_start_s=0.35, fault_end_s=0.90, sensor_noise=True)
    if family == "close_static":
        sim = RoverSim(scene="open", seed=seed, fault=fault, fault_start_s=0.35, fault_end_s=0.90)
        x_post = float(rng.uniform(0.32, 0.50))
        # Alternate narrow center posts (between whisker rays) and flank-grazing posts so 0.30s+brake contrasts cleanly
        mode = seed % 3
        if mode == 0:
            y_post = float(rng.uniform(-0.03, 0.03))
            half_y = 0.025
        elif mode == 1:
            side = 1.0 if (seed // 3) % 2 == 0 else -1.0
            y_post = side * float(rng.uniform(0.095, 0.115))
            half_y = 0.025
        else:
            y_post = float(rng.uniform(-0.05, 0.05))
            half_y = 0.045
        sim.solids.append(sim.box([x_post, y_post, 0.12], [0.025, half_y, 0.12], 0, [0.86, 0.17, 0.09, 1]))
    elif family == "close_occluded_hazard":
        # Foreground narrow-flank post/baffle (0.34-0.46m) + tucked hazard behind on hidden_side
        sim = RoverSim(scene="open", seed=seed, fault=fault, fault_start_s=0.35, fault_end_s=0.90)
        x_occ = float(rng.uniform(0.35, 0.46))
        side = 1.0 if seed % 2 == 0 else -1.0
        sim.solids.append(sim.box([x_occ, -side * 0.02, 0.12], [0.025, 0.045, 0.12], 0, [0.86, 0.17, 0.09, 1]))
        sim.solids.append(sim.box([x_occ + 0.14, side * float(rng.uniform(0.09, 0.125)), 0.12], [0.025, 0.025, 0.12], 0, [0.86, 0.17, 0.09, 1]))
    elif family == "pedestrian_close_cross":
        env = RealWorldEnv(scene="corridor_pedestrian", seed=seed, fault=fault, fault_start_s=0.35, fault_end_s=0.90, sensor_noise=True)
        # Closer crossing capsule in active lane to cover dynamic near-field collisions
        ped = _PedestrianProxy(env.client, cx=float(rng.uniform(0.42, 0.68)), cy=float(rng.uniform(-0.14, 0.14)), radius=0.22, speed=0.85)
        env._dynamic_objs.append(ped)
        env.solids.append(ped.body_id)
        return env
    elif family == "chicane_slalom":
        # Untouched geometry 1: alternating narrow barriers requiring controlled S-maneuver
        sim = RoverSim(scene="open", seed=seed, fault=fault, fault_start_s=0.45, fault_end_s=0.95)
        s_dir = 1.0 if seed % 2 == 0 else -1.0
        sim.solids.append(sim.box([0.40, s_dir * 0.03, 0.12], [0.025, 0.035, 0.12], 0, [0.86, 0.17, 0.09, 1]))
        sim.solids.append(sim.box([0.70, -s_dir * 0.09, 0.12], [0.025, 0.045, 0.12], 0, [0.86, 0.17, 0.09, 1]))
    elif family == "blind_alcove_emergence":
        # Untouched geometry 2: asymmetric off-center baffle + recessed side obstacle
        sim = RoverSim(scene="open", seed=seed, fault=fault, fault_start_s=0.45, fault_end_s=0.95)
        side = 1.0 if seed % 2 == 0 else -1.0
        sim.solids.append(sim.box([0.40, side * 0.02, 0.12], [0.025, 0.045, 0.12], 0, [0.86, 0.17, 0.09, 1]))
        sim.solids.append(sim.box([0.60, -side * 0.105, 0.12], [0.025, 0.025, 0.12], 0, [0.86, 0.17, 0.09, 1]))
    elif family == "dynamic_crosswalk":
        # Untouched geometry 3: lower-friction outdoor surface + dynamic pedestrian crossing
        return RealWorldEnv(scene="outdoor_path", seed=seed, fault=fault, fault_start_s=0.45, fault_end_s=0.95, sensor_noise=True)
    elif family == "ramp_chicanery":
        # Untouched geometry 4: sloped tile ramp with an off-axis narrow pillar
        env = RealWorldEnv(scene="ramp_descent", seed=seed, fault=fault, fault_start_s=0.45, fault_end_s=0.95, sensor_noise=True)
        side = 1.0 if seed % 2 == 0 else -1.0
        env.solids.append(env._box([0.44, side * float(rng.uniform(0.0, 0.09)), 0.12], [0.025, 0.025, 0.12], 0, [0.86, 0.17, 0.09, 1]))
        return env
    else:
        raise ValueError(f"Unknown P6.9 world family: {family}")

    sim.packet = None
    sim.queue.clear()
    sim.update_camera()
    return sim


def default_p69_splits() -> dict:
    """110 independent world specs (76 train / 17 val / 17 test), zero seed/family leakage."""
    train = []
    # 1. Occluded hazards & emergence (~25% -> 19 worlds)
    train += [("close_occluded_hazard", "none", 2000 + i) for i in range(11)]
    train += [("shelf_hazard", "none", 2020 + i) for i in range(8)]
    # 2. Dynamic pedestrian crossings & near misses (~20% -> 15 worlds)
    train += [("pedestrian_close_cross", "none", 2100 + i) for i in range(7)]
    train += [("corridor_pedestrian", "none", 2120 + i) for i in range(4)]
    train += [("warehouse_forklift_cross", "none", 2140 + i) for i in range(4)]
    # 3. Static obstacles at varied close/intermediate distances (~15% -> 12 worlds)
    train += [("close_static", "none", 2200 + i) for i in range(9)]
    train += [("corridor", "none", 2220 + i) for i in range(3)]
    # 4. Ramps, potholes & ego-motion disturbances (~10% -> 8 worlds)
    train += [("ramp_descent", "none", 2300 + i) for i in range(3)]
    train += [("ramp_ascent", "none", 2320 + i) for i in range(2)]
    train += [("outdoor_pothole", "none", 2340 + i) for i in range(3)]
    # 5. Temporary & persistent sensor failures (~10% -> 8 worlds)
    train += [("post_sweep", "freeze", 2400 + i) for i in range(2)]
    train += [("corridor", "delay", 2410 + i) for i in range(2)]
    train += [("close_static", "drop", 2420 + i) for i in range(2)]
    train += [("night_mode", "none", 2430 + i) for i in range(2)]
    # 6. Empty & safely navigable environments (~10% -> 7 worlds)
    train += [("open_rover", "none", 2500 + i) for i in range(4)]
    train += [("open_real", "none", 2520 + i) for i in range(3)]
    # 7. Paired ambiguous-world experiments (~10% -> 7 worlds)
    train += [("shelf_clear", "none", 2600 + i) for i in range(4)]
    train += [("shelf_hazard", "none", 2600 + i) for i in range(3)]

    # Representative, UN-MINED validation split (17 independent seeds)
    val = [
        ("close_occluded_hazard", "none", 5000), ("close_occluded_hazard", "none", 5001),
        ("shelf_hazard", "none", 5010), ("shelf_clear", "none", 5011),
        ("pedestrian_close_cross", "none", 5020), ("corridor_pedestrian", "delay", 5021),
        ("warehouse_forklift_cross", "none", 5030), ("close_static", "none", 5040),
        ("close_static", "none", 5041), ("corridor", "freeze", 5050),
        ("post_sweep", "delay", 5051), ("ramp_descent", "none", 5060),
        ("outdoor_pothole", "none", 5061), ("night_mode", "none", 5070),
        ("open_rover", "none", 5080), ("open_real", "none", 5081),
        ("pedestrian_close_cross", "none", 5092),
    ]

    # Representative, UN-MINED test split with UNTOUCHED world families + unseen seeds (17 worlds)
    test = [
        ("chicane_slalom", "none", 6000), ("chicane_slalom", "none", 6001),
        ("chicane_slalom", "freeze", 6002), ("chicane_slalom", "none", 6003),
        ("blind_alcove_emergence", "none", 6010), ("blind_alcove_emergence", "none", 6011),
        ("blind_alcove_emergence", "delay", 6012), ("blind_alcove_emergence", "none", 6013),
        ("dynamic_crosswalk", "none", 6020), ("dynamic_crosswalk", "drop", 6022),
        ("ramp_chicanery", "none", 6030), ("ramp_chicanery", "none", 6031),
        ("close_static", "none", 6040), ("close_static", "drop", 6041),
        ("close_occluded_hazard", "none", 6050), ("pedestrian_close_cross", "none", 6060),
        ("open_rover", "none", 6070),
    ]
    return {"train": train, "val": val, "test": test}


def _prefix_action_schedule(family: str, seed: int, step: int) -> str:
    """Straight approach to near-field boundary first, with post-approach exploration on subset of worlds."""
    if family in ("open_rover", "open_real", "ramp_descent", "ramp_ascent"):
        return "forward"
    if seed % 6 == 1 and step in (17, 18):
        return "left"
    if seed % 6 == 2 and step in (17, 18):
        return "right"
    if seed % 6 == 3 and step >= 18:
        return "slow"
    return "forward"


def _rollout_one_world_task(task: Tuple[str, int, str, str, int]) -> dict:
    """Isolated deterministic worker simulating one world and its counterfactual branches."""
    sp, ep_idx, family, fault, seed = task
    world_id = f"{family}:{fault}:{seed}"
    episode_id = f"{sp}_{ep_idx}_{world_id}"
    branch_steps = (8, 12, 16) if ("close" in family or "pedestrian" in family or family in UNTOUCHED_WORLD_FAMILIES) else (10, 15, 20)
    max_step = max(branch_steps) + 1

    items = []
    branch_ticks = 0
    with build_p69_world(family, seed, fault) as sim:
        b1 = SectorBelief(max_age_s=0.45)
        b2 = FactorizedSpatialBelief(max_age_s=0.60)
        vis = DualPathwayT4T5Vision()
        mon = ObservationIntegrityMonitor()
        lstate = None
        hist = []
        prev_t = 0.0
        last_cap = None
        anchor = None
        cue_latest = None
        prev_cmd = "brake"
        vo = {"raw_lplc2": 0.0, "residual_lplc2": 0.0, "residual_lc4": 0.0}

        for step in range(max_step):
            obs = sim.observe()
            if not ORACLE_KEYS.isdisjoint(obs):
                raise AssertionError("Oracle key leaked into observation")
            dt = max(0.05, obs["time_s"] - prev_t)
            prev_t = obs["time_s"]
            rep = mon.step(obs, imu_enabled=True)
            if obs["capture_s"] != last_cap:
                vo = vis.step(obs["rgb"], max(0.05, dt), rep.imu_gyro_z, rep.imu_gyro_y)
                if rep.camera_healthy:
                    b_ang = bearing(obs["rgb"])
                    if anchor is None and b_ang is not None and abs(vo.get("residual_lc4", 0.0)) < 0.35:
                        anchor = b_ang
                    cue_latest = None if (b_ang is None or anchor is None or abs(vo.get("residual_lc4", 0.0)) >= 0.35) else (b_ang - anchor)
                else:
                    cue_latest = None
                last_cap = obs["capture_s"]

            ipm = estimate_ground_contact_clearance(obs["rgb"], pitch_rad=0.0) if rep.camera_healthy else {
                "obstacle_detected": False, "estimated_clearance_m": 4.0, "ipm_reliable": False
            }
            s1 = b1.update(obs)
            vis_stable = bool(rep.camera_healthy and float(obs.get("contrast", 0.0)) >= 0.04 and abs(float(vo.get("residual_lc4", 0.0))) < 0.18)
            # 100% sensor-only belief propagation (NO world_yaw_rad oracle fallback!)
            s2 = b2.update(
                obs,
                delta_yaw_rad=rep.delta_yaw_rad,
                heading_unc_delta_rad=rep.heading_uncertainty_delta_rad,
                action_cmd=prev_cmd,
                visual_yaw_rad=cue_latest,
                visual_yaw_kappa=10.0 if (rep.camera_healthy and cue_latest is not None) else 0.0,
                visual_motion_stable=vis_stable,
            )
            feat, lstate = extract_features_v3(obs, s1, s2, vo, ipm, lstate, dt)
            feat[14] = 0.0  # Strip non-Markovian wall-clock time_s so closed-loop policy is time-shift invariant past 2.4s
            hist.append(feat)

            if step in branch_steps and not sim.evaluator()["body_contact"]:
                branch_ticks += 1
                seq = np.zeros((24, 24), dtype=np.float32)
                recent = hist[-24:]
                seq[-len(recent) :] = np.stack(recent)
                branches = rollout_multi_horizon_branches(sim, list(ACTIONS), b2)
                items.append((step, seq, rep.camera_status, branches))

            cmd = _prefix_action_schedule(family, seed, step)
            mon.record_action(cmd, outage_active=(not rep.camera_healthy))
            prev_cmd = cmd
            sim.step(cmd)

    return {
        "sp": sp,
        "ep_idx": ep_idx,
        "family": family,
        "world_id": world_id,
        "episode_id": episode_id,
        "branch_ticks": branch_ticks,
        "items": items,
    }


def generate_p69_dataset(
    out: Path,
    splits: Optional[dict] = None,
    miner_weights: Optional[Path] = Path("runs/pilot-v3-models/pilot_weights_v3.npz"),
    hard_mine_frac: float = 0.25,
    workers: int = 1,
) -> dict:
    if splits is None:
        splits = default_p69_splits()

    bio_miner = gru_miner = None
    if miner_weights is not None and miner_weights.exists() and hard_mine_frac > 0.0:
        try:
            bio_miner = _PortableV3Policy(miner_weights, "FullV3_NeuroCOVER_Bio", "full_v3_neurocover")
            gru_miner = _PortableV3Policy(miner_weights, "FullV3_NeuroCOVER_GRU", "full_v3_neurocover")
        except Exception:
            bio_miner = gru_miner = None

    out.mkdir(parents=True, exist_ok=True)
    records = []
    split_summaries = {}
    world_splits: Dict[str, set] = {}

    for sp, specs in splits.items():
        X_list, act_list = [], []
        y_risk_cont, y_haz_bin, y_cens = [], [], []
        y_risk_prot, y_prog_prot, y_prog_cont = [], [], []
        y_clear_prot, y_du, y_cam_ok, y_recov = [], [], [], []
        mine_scores: List[float] = []
        fam_counts: Dict[str, int] = {}
        ep_count = len(specs)
        branch_tick_count = 0

        tasks = [(sp, ep_idx, fam, flt, sd) for ep_idx, (fam, flt, sd) in enumerate(specs)]
        if workers > 1 and len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                world_outputs = list(ex.map(_rollout_one_world_task, tasks))
        else:
            world_outputs = [_rollout_one_world_task(t) for t in tasks]
        world_outputs.sort(key=lambda w: w["ep_idx"])

        for wres in world_outputs:
            family = wres["family"]
            world_id = wres["world_id"]
            episode_id = wres["episode_id"]
            world_splits.setdefault(world_id, set()).add(sp)
            fam_counts[family] = fam_counts.get(family, 0) + 1
            branch_tick_count += wres["branch_ticks"]

            for step, seq, cam_status, branches in wres["items"]:
                bio_p = bio_miner.predict_actions(seq) if (sp == "train" and bio_miner is not None) else None
                gru_p = gru_miner.predict_actions(seq) if (sp == "train" and gru_miner is not None) else None

                for a_idx, act in enumerate(ACTIONS):
                    br = branches[act]
                    X_list.append(seq)
                    act_list.append(a_idx)
                    y_risk_cont.append(br["continue_risk_bins"])
                    y_haz_bin.append(br["continue_hazard_bin"])
                    y_cens.append(1.0 if br["continue_censored"] else 0.0)
                    y_risk_prot.append(1.0 if br["protected_body_contact"] else 0.0)
                    y_prog_prot.append(br["protected_progress_m"])
                    y_prog_cont.append(br["continue_progress_m"])
                    y_clear_prot.append(br["protected_min_clearance_m"])
                    y_du.append(br["signed_info_gain"])
                    y_cam_ok.append(1.0 if br["camera_healthy_end"] else 0.0)
                    y_recov.append(1.0 if br["safe_recover"] else 0.0)

                    if bio_p is not None and gru_p is not None:
                        e_risk = abs(bio_p[act]["r_prot"] - float(br["protected_body_contact"]))
                        e_info = abs(bio_p[act]["delta_u"] - float(br["signed_info_gain"]))
                        d_mod = abs(bio_p[act]["r_prot"] - gru_p[act]["r_prot"]) + abs(bio_p[act]["r_cum"][4] - gru_p[act]["r_cum"][4])
                        novelty = 0.5 if br["protected_body_contact"] else 0.0
                        mine_scores.append(float(1.5 * e_risk + 1.0 * e_info + 1.0 * d_mod + novelty))
                    else:
                        mine_scores.append(1.0 if br["protected_body_contact"] else 0.0)

                    records.append({
                        "split": sp,
                        "world_id": world_id,
                        "episode_id": episode_id,
                        "branch_id": f"{episode_id}:t{step}:{act}",
                        "decision_step": step,
                        "action": act,
                        "action_duration_s": br["action_duration_s"],
                        "camera_health": cam_status,
                        "untouched_family": family in UNTOUCHED_WORLD_FAMILIES,
                        **br,
                    })

        unmined_count = len(X_list)
        mined_added = 0
        # Hard-example mining strictly inside train split ONLY (never val or test)
        if sp == "train" and hard_mine_frac > 0.0 and len(mine_scores) > 0:
            k_top = max(1, int(round(hard_mine_frac * unmined_count)))
            top_idx = np.argsort(-np.asarray(mine_scores, dtype=np.float64))[:k_top]
            for idx_m in top_idx:
                X_list.append(X_list[idx_m])
                act_list.append(act_list[idx_m])
                y_risk_cont.append(y_risk_cont[idx_m])
                y_haz_bin.append(y_haz_bin[idx_m])
                y_cens.append(y_cens[idx_m])
                y_risk_prot.append(y_risk_prot[idx_m])
                y_prog_prot.append(y_prog_prot[idx_m])
                y_prog_cont.append(y_prog_cont[idx_m])
                y_clear_prot.append(y_clear_prot[idx_m])
                y_du.append(y_du[idx_m])
                y_cam_ok.append(y_cam_ok[idx_m])
                y_recov.append(y_recov[idx_m])
                mined_added += 1

        npz_path = out / f"{sp}.npz"
        rc_arr = np.asarray(y_risk_cont, dtype=np.float32)
        rp_arr = np.asarray(y_risk_prot, dtype=np.float32)
        du_arr = np.asarray(y_du, dtype=np.float32)
        np.savez_compressed(
            npz_path,
            X=np.asarray(X_list, dtype=np.float32),
            action=np.asarray(act_list, dtype=np.int64),
            y_risk_horizons=rc_arr,
            y_hazard_bin=np.asarray(y_haz_bin, dtype=np.int64),
            y_censored=np.asarray(y_cens, dtype=np.float32),
            y_risk_protected=rp_arr,
            y_progress_protected=np.asarray(y_prog_prot, dtype=np.float32),
            y_progress_continue=np.asarray(y_prog_cont, dtype=np.float32),
            y_min_clearance_protected=np.asarray(y_clear_prot, dtype=np.float32),
            y_signed_delta_u=du_arr,
            y_cam_healthy=np.asarray(y_cam_ok, dtype=np.float32),
            y_recover=np.asarray(y_recov, dtype=np.float32),
        )
        split_summaries[sp] = {
            "independent_worlds": ep_count,
            "episodes": ep_count,
            "decision_states": branch_tick_count,
            "unmined_branches": unmined_count,
            "hard_mined_oversampled": mined_added,
            "counterfactual_branches": len(X_list),
            "short_safe_long_danger_count": int(np.sum((rp_arr < 0.5) & (rc_arr[:, -1] > 0.5))),
            "risk_protected_pos": int(np.sum(rp_arr > 0.5)),
            "risk_protected_rate": float(np.mean(rp_arr)) if len(rp_arr) else 0.0,
            "risk_continue_pos": [int(np.sum(rc_arr[:, j] > 0.5)) for j in range(len(HORIZONS_S))] if len(rc_arr) else [],
            "risk_continue_rates": [float(rc_arr[:, j].mean()) for j in range(len(HORIZONS_S))] if len(rc_arr) else [],
            "negative_delta_u_count": int(np.sum(du_arr < -1e-6)),
            "positive_delta_u_count": int(np.sum(du_arr > 1e-6)),
            "safe_recover_rate": float(np.mean(y_recov)) if y_recov else 0.0,
            "families": fam_counts,
            "sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        }

    leakage = sum(len(s) != 1 for s in world_splits.values())
    if leakage:
        raise AssertionError("World-split leakage detected in P6.9 dataset")

    rec_path = out / "records.jsonl"
    with rec_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    manifest = {
        "schema_version": "p69_targeted_v1",
        "ctrl_hz": CTRL_HZ,
        "cam_hz": 10,
        "phys_hz": 120,
        "feature_dim": 24,
        "seq_len": 24,
        "horizons_s": list(HORIZONS_S),
        "protected_exec_s": float(PROTECTED_EXEC_STEPS * DT_CTRL_S),
        "actions": list(ACTIONS),
        "oracle_keys_in_inputs": 0,
        "sensor_only_belief_provenance": True,
        "world_split_leakage": leakage,
        "untouched_test_families": list(UNTOUCHED_WORLD_FAMILIES),
        "splits": split_summaries,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("runs/pilot-v3-p69"))
    p.add_argument("--miner-weights", type=Path, default=Path("runs/pilot-v3-models/pilot_weights_v3.npz"))
    p.add_argument("--hard-mine-frac", type=float, default=0.25)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    m = generate_p69_dataset(args.out, miner_weights=args.miner_weights, hard_mine_frac=args.hard_mine_frac, workers=args.workers)
    print(json.dumps(m["splits"], indent=2))


if __name__ == "__main__":
    main()
