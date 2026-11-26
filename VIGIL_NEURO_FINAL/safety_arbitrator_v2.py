"""P5 Protected Autonomy Architecture (P5.1-P5.6) for VIGIL-ROVER X NeuroAI V2.

Organizes control into six strictly layered protection and decision stages:
  P5.1 ObservationIntegrityMonitor : timestamp deduplication, gyro bias/missing-IMU
       uncertainty propagation, and granular camera health classification.
  P5.2 IndependentEmergencyGuard   : tactile whisker reflexes, context-conditioned
       dual-pathway T4/T5 visual threat fusion (no blind AND-gate), and near-field stops.
  P5.3 ActionFeasibilityChecker    : swept-path & stopping-distance feasibility across
       ["forward", "slow", "left", "right", "brake", "reverse"].
  P5.4 UncertaintyRiskArbitrator   : min(1, max(R4, R8) + kappa*|R4 - R8|) heuristic
       preserving single-model warnings (never certified as a formal safety proof).
  P5.5 ContinuousUtilityOptimizer  : smooth sigmoid uncertainty weighting w_U(b_t),
       signed info-gain term, progress, risk, maneuver cost, and switching penalty.
  P5.6 NeuroCoverV2Controller      : full integrated executive with ablation toggles
       supporting causal configurations A..G on identical G08 & R05 scenarios.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from ground_range_estimator import AdaptiveBrakingModel, estimate_ground_contact_clearance
from recovery_policy import choose_recovery_action


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, float(x)))))


@dataclass
class IntegrityReport:
    camera_status: str          # "healthy" | "temp_delay" | "glare" | "frozen" | "blackout"
    camera_healthy: bool
    delta_yaw_rad: Optional[float]
    imu_available: bool
    imu_duplicate: bool
    imu_gyro_z: float
    imu_gyro_y: float
    heading_uncertainty_delta_rad: float
    probing_exhausted: bool


class ObservationIntegrityMonitor:
    """P5.1: Validate sensor timestamps, integrate bias-corrected IMU yaw, classify camera state."""

    def __init__(self, max_freeze_probes: int = 2):
        self.last_imu_t: Optional[float] = None
        self.last_cam_capture_s: Optional[float] = None
        self.stale_duration_s: float = 0.0
        self.frozen_duration_s: float = 0.0
        self.gyro_z_bias: float = 0.0
        self.prev_cmd: str = "brake"
        self.probe_attempts_in_outage: int = 0
        self.max_freeze_probes: int = max_freeze_probes
        self._last_rgb_hash: Optional[float] = None

    def reset(self) -> None:
        self.last_imu_t = None
        self.last_cam_capture_s = None
        self.stale_duration_s = 0.0
        self.frozen_duration_s = 0.0
        self.gyro_z_bias = 0.0
        self.prev_cmd = "brake"
        self.probe_attempts_in_outage = 0
        self._last_rgb_hash = None

    def record_action(self, action: str, outage_active: bool) -> None:
        self.prev_cmd = action
        if outage_active and action in ("left", "right"):
            self.probe_attempts_in_outage += 1
        elif not outage_active:
            self.probe_attempts_in_outage = 0

    def step(self, obs: dict, imu_enabled: bool = True) -> IntegrityReport:
        now = float(obs["time_s"])
        dt = 0.0 if self.last_imu_t is None else (now - self.last_imu_t)
        imu_dup = self.last_imu_t is not None and dt <= 1e-9
        raw_gz = obs.get("imu_gyro_z") if imu_enabled else None
        raw_gy = obs.get("imu_gyro_y") if imu_enabled else None

        imu_available = raw_gz is not None and np.isfinite(raw_gz)
        if not imu_available:
            # Missing IMU telemetry: never silently pretend angular velocity is known zero
            gz = 0.0
            gy = 0.0
            dyaw = None
            unc_delta = 0.08 * max(dt, 0.05) if self.last_imu_t is not None else 0.02
            if dt > 0:
                self.last_imu_t = now
        elif imu_dup:
            # Duplicate timestamp protection: do NOT double-integrate identical measurement
            gz = float(raw_gz) - self.gyro_z_bias
            gy = float(raw_gy) if (raw_gy is not None and np.isfinite(raw_gy)) else 0.0
            dyaw = 0.0
            unc_delta = 0.0
        else:
            self.last_imu_t = now
            raw_gz_f = float(raw_gz)
            # Stationary online bias estimation while commanded brake & low rotation
            if self.prev_cmd == "brake" and abs(raw_gz_f - self.gyro_z_bias) < 0.06:
                self.gyro_z_bias = 0.95 * self.gyro_z_bias + 0.05 * raw_gz_f
            gz = raw_gz_f - self.gyro_z_bias
            gy = float(raw_gy) if (raw_gy is not None and np.isfinite(raw_gy)) else 0.0
            dyaw = gz * dt if dt > 0.0 else 0.0
            unc_delta = 0.004 * dt

        # Granular camera health classification
        valid = obs.get("frame_valid") is True
        age = obs.get("frame_age_s")
        cap = obs.get("capture_s")
        brightness = float(obs.get("brightness", 0.5))
        contrast = float(obs.get("contrast", 0.2))
        rgb = obs.get("rgb")
        rgb_sig = float(rgb[::8, ::8].mean()) if rgb is not None else -1.0

        fresh_frame = valid and age is not None and float(age) <= 0.051
        if not valid or (age is not None and float(age) > 0.60):
            cam_status = "blackout"
            self.stale_duration_s += max(dt, 0.05)
        elif not fresh_frame:
            # Check if frame pixel signature & capture timestamp remain stuck -> frozen vs temp_delay
            same_cap = (cap is not None and self.last_cam_capture_s is not None and abs(float(cap) - self.last_cam_capture_s) < 1e-9)
            same_pix = (self._last_rgb_hash is not None and abs(rgb_sig - self._last_rgb_hash) < 1e-5)
            if same_cap or same_pix:
                self.frozen_duration_s += max(dt, 0.05)
                cam_status = "frozen" if self.frozen_duration_s >= 0.20 else "temp_delay"
            else:
                self.frozen_duration_s = 0.0
                cam_status = "temp_delay"
            self.stale_duration_s += max(dt, 0.05)
        elif brightness > 0.88 or (contrast < 0.02 and brightness > 0.80):
            cam_status = "glare"
            self.stale_duration_s = 0.0
            self.frozen_duration_s = 0.0
        else:
            cam_status = "healthy"
            self.stale_duration_s = 0.0
            self.frozen_duration_s = 0.0

        self.last_cam_capture_s = float(cap) if cap is not None else self.last_cam_capture_s
        self._last_rgb_hash = rgb_sig

        probing_exhausted = (
            cam_status in ("blackout", "frozen")
            and self.probe_attempts_in_outage >= self.max_freeze_probes
        )
        return IntegrityReport(
            camera_status=cam_status,
            camera_healthy=(cam_status == "healthy"),
            delta_yaw_rad=dyaw,
            imu_available=imu_available,
            imu_duplicate=imu_dup,
            imu_gyro_z=gz,
            imu_gyro_y=gy,
            heading_uncertainty_delta_rad=unc_delta,
            probing_exhausted=probing_exhausted,
        )


def fuse_biological_visual_threat(
    vis_out: dict,
    obs: dict,
    integrity: IntegrityReport,
    lplc2_thresh: float = 0.45,
    lc4_thresh: float = 0.35,
    near_red_thresh: float = 0.11,
) -> Tuple[bool, str, float]:
    """P5.2 Context-conditioned dual-pathway visual threat fusion (no unconditional AND gate).

    Preserves raw collision-course expansion on stationary walls (where ego-flow is forward
    translation, not rotation), while using compensated residual flow + lateral imbalance
    to interpret pitch/yaw ego-motion and near-miss lateral crossings.
    """
    raw_l2 = float(vis_out.get("raw_lplc2", 0.0))
    res_l2 = float(vis_out.get("residual_lplc2", 0.0))
    res_lc4 = float(vis_out.get("residual_lc4", 0.0))
    lat_imb = float(vis_out.get("lateral_imbalance", 0.0))
    obs_center = float(obs.get("obstacle_center", 0.0))

    # Near-field static obstacle fallback grounded in pinhole ground-contact IPM (fixes tall pedestrian 1.05m false stop)
    rgb = obs.get("rgb")
    if rgb is not None and isinstance(rgb, np.ndarray) and rgb.ndim == 3:
        ipm = estimate_ground_contact_clearance(rgb, pitch_rad=0.0)
        if ipm["obstacle_detected"] and ipm["ipm_reliable"] and ipm["estimated_clearance_m"] <= 0.30:
            return True, f"near_field_ipm_clearance={ipm['estimated_clearance_m']:.2f}m", float(min(1.0, 0.35 / max(ipm["estimated_clearance_m"], 0.05)))
    elif obs_center >= near_red_thresh:
        return True, "near_field_obstacle_center", float(min(1.0, obs_center / 0.25))

    # Rotational pitch/yaw disturbance regime (skip false triggers during large deliberate turns or steep ramp pitch)
    high_rotation = abs(integrity.imu_gyro_z) > 0.25 or abs(integrity.imu_gyro_y) > 0.18
    if high_rotation:
        score = max(res_l2, 0.8 * res_lc4)
        if res_l2 >= lplc2_thresh and res_lc4 >= lc4_thresh * 0.8:
            return True, "residual_expansion_during_rotation", score
        return False, "rotation_compensated_clear", score

    # Normal forward motion: preserve RAW stationary wall expansion (DO NOT require raw ∧ residual!)
    crossing_dominated = lat_imb > 1.8 * max(res_l2, 0.20) and raw_l2 < lplc2_thresh * 1.4
    score = max(raw_l2, res_l2, 0.7 * res_lc4)
    if (raw_l2 >= lplc2_thresh or res_l2 >= lplc2_thresh) and not crossing_dominated:
        return True, "biological_collision_cone_expansion", score
    if res_lc4 >= lc4_thresh and res_l2 >= 0.20 and not crossing_dominated:
        return True, "lc4_residual_threat", score
    return False, "visual_cone_clear", score


class ActionFeasibilityChecker:
    """P5.3 / P6.3 / P6.4 Swept-path, calibrated IPM ground-clearance & adaptive stopping validation."""

    def __init__(
        self,
        rover_half_width_m: float = 0.095,
        tau_reaction_s: float = 0.10,
        a_brake_m_s2: float = 0.80,
        d_margin_m: float = 0.06,
        max_safe_age_s: float = 0.60,
        max_heading_unc_rad: float = 0.55,
    ):
        self.half_w = rover_half_width_m
        self.braking = AdaptiveBrakingModel(tau_reaction_s=tau_reaction_s, base_decel_m_s2=a_brake_m_s2, d_margin_m=d_margin_m)
        self.max_age = max_safe_age_s
        self.max_head_unc = max_heading_unc_rad
        self.last_diagnostics: Dict[str, dict] = {}

    def stopping_distance(self, speed_m_s: float, friction_mu: float = 0.60) -> float:
        """Conservative stopping distance validated against PyBullet motor and friction dynamics."""
        return self.braking.stopping_distance_m(speed_m_s, friction_mu=friction_mu)

    def feasible_actions(
        self,
        actions: Tuple[str, ...],
        belief_snap: dict,
        obs: dict,
        integrity: IntegrityReport,
        speed_m_s: float,
    ) -> Dict[str, Tuple[bool, str]]:
        """Evaluate swept occupancy, staleness, heading uncertainty, and calibrated IPM clearance."""
        states = belief_snap["sector_states"]
        n = len(states)
        mid = n // 2
        ages = belief_snap.get("sector_ages_s", [0.0] * n)
        near_blk = belief_snap.get("near_blocked_sectors")
        world_occ = belief_snap.get("world_occupancy_probs")
        head_unc = float(belief_snap.get("heading_uncertainty_rad", 0.0))
        mu = float(obs.get("friction_mu", 0.60))

        w_left = bool(obs.get("whisker_left"))
        w_right = bool(obs.get("whisker_right"))
        d_stop = self.stopping_distance(speed_m_s, friction_mu=mu)

        rgb = obs.get("rgb")
        ipm = None
        est_clear = 4.0
        if integrity.camera_healthy and rgb is not None and isinstance(rgb, np.ndarray) and rgb.ndim == 3:
            ipm = estimate_ground_contact_clearance(rgb, pitch_rad=0.0)
            est_clear = float(ipm["estimated_clearance_m"])

        def sector_safe(indices, check_near_only: bool = False) -> Tuple[bool, str]:
            for idx in indices:
                st = states[idx]
                age = ages[idx]
                if age is None or not np.isfinite(age) or float(age) > self.max_age:
                    return False, "stale_or_unobserved_sector"
                if check_near_only and near_blk is not None:
                    # Only block forward path when calibrated IPM distance indicates stopping boundary reached (or manual unit-test flag)
                    if near_blk[idx]:
                        if ipm is not None and ipm["ipm_reliable"] and est_clear > d_stop + 0.10:
                            continue
                        return False, "near_field_occupied_sector"
                else:
                    if st == "occupied":
                        return False, "occupied_sector"
                    if st != "observed_free":
                        return False, "stale_or_unobserved_sector"
            return True, "clear"

        front_idx = range(max(0, mid - 1), min(n, mid + 2))
        left_idx = range(0, mid)
        right_idx = range(mid + 1, n)
        # Note: in manual dict-only unit tests (rgb synthetic/uniform without IPM), check_near_only triggers on explicit 'occupied'
        use_near_mode = near_blk is not None
        front_ok, front_rsn = sector_safe(front_idx, check_near_only=use_near_mode)
        left_ok, left_rsn = sector_safe(left_idx, check_near_only=use_near_mode)
        right_ok, right_rsn = sector_safe(right_idx, check_near_only=use_near_mode)

        out: Dict[str, Tuple[bool, str]] = {}
        self.last_diagnostics = {}
        for act in actions:
            rejected_by = []
            if act == "brake":
                allowed, rsn = True, "passive_stop_always_feasible"
            elif w_left and w_right:
                allowed, rsn = False, "bilateral_tactile_block"
                rejected_by.append(rsn)
            elif head_unc > self.max_head_unc and act in ("forward", "left", "right", "reverse"):
                allowed, rsn = False, "excessive_heading_uncertainty"
                rejected_by.append(rsn)
            elif act == "forward":
                if w_left or w_right:
                    rejected_by.append("tactile_contact_forward_blocked")
                if not integrity.camera_healthy and not front_ok:
                    rejected_by.append(f"camera_{integrity.camera_status}_front_unverified")
                elif not front_ok:
                    rejected_by.append(front_rsn)
                elif ipm is not None and ipm["obstacle_detected"] and ipm["ipm_reliable"] and est_clear <= d_stop + 0.05:
                    rejected_by.append("insufficient_ground_clearance")
                allowed = len(rejected_by) == 0
                rsn = rejected_by[0] if rejected_by else f"swept_clear_d_stop={d_stop:.2f}m"
            elif act == "slow":
                if w_left or w_right:
                    rejected_by.append("tactile_contact_slow_blocked")
                elif integrity.camera_status in ("blackout", "frozen") and not front_ok:
                    rejected_by.append(f"camera_{integrity.camera_status}_unverified")
                elif states[mid] == "occupied" and (not use_near_mode or (ipm is not None and est_clear <= self.stopping_distance(0.12, mu))):
                    rejected_by.append("center_occupied")
                allowed = len(rejected_by) == 0
                rsn = rejected_by[0] if rejected_by else "bounded_cautious_advance"
            elif act == "left":
                if w_left:
                    rejected_by.append("left_whisker_active")
                elif integrity.probing_exhausted:
                    rejected_by.append("persistent_outage_probe_limit")
                elif not left_ok:
                    rejected_by.append(left_rsn)
                allowed = len(rejected_by) == 0
                rsn = rejected_by[0] if rejected_by else "left_swept_free"
            elif act == "right":
                if w_right:
                    rejected_by.append("right_whisker_active")
                elif integrity.probing_exhausted:
                    rejected_by.append("persistent_outage_probe_limit")
                elif not right_ok:
                    rejected_by.append(right_rsn)
                allowed = len(rejected_by) == 0
                rsn = rejected_by[0] if rejected_by else "right_swept_free"
            elif act == "reverse":
                if world_occ is None:
                    rejected_by.append("blind_reverse_prohibited")
                else:
                    yaw = float(belief_snap.get("yaw_est_rad", 0.0))
                    m = len(world_occ)
                    ang_grid = np.linspace(-math.pi, math.pi, m, endpoint=False) + math.pi / m
                    rear_diff = np.abs((ang_grid - (yaw + math.pi) + math.pi) % (2.0 * math.pi) - math.pi)
                    rear_occ = [world_occ[k] for k in range(m) if rear_diff[k] <= math.radians(30.0)]
                    if not (rear_occ and all(p < 0.25 for p in rear_occ) and integrity.camera_healthy):
                        rejected_by.append("rear_unobserved_or_occupied")
                allowed = len(rejected_by) == 0
                rsn = rejected_by[0] if rejected_by else "verified_rear_clear"
            else:
                rejected_by.append(f"unknown_action_{act}")
                allowed, rsn = False, rejected_by[0]

            out[act] = (allowed, rsn)
            self.last_diagnostics[act] = {
                "action": act,
                "allowed": allowed,
                "rejected_by": rejected_by,
                "required_braking_distance_m": round(d_stop, 4),
                "estimated_clearance_m": round(est_clear, 4),
                "camera_health": integrity.camera_healthy,
            }
        return out


@dataclass
class ArbitrationConfig:
    """Parameters for calibrated disagreement arbitration & continuous utility optimization."""
    kappa_disagree: float = 0.50
    risk_limit_eps: float = 0.55
    alpha_progress: float = 1.0
    beta_info_gain: float = 0.85
    lambda_risk: float = 1.60
    rho_motion_cost: float = 0.12
    eta_switch_cost: float = 0.18
    tau_uncertainty: float = 0.35
    temp_uncertainty: float = 0.10
    use_continuous_u_weight: bool = True


def compute_arbitrated_risk(r4: float, r8: float, kappa: float = 0.50) -> float:
    """Bounded conservative disagreement risk heuristic in [0, 1] (NOT a formal safety bound)."""
    r4_c = float(np.clip(r4, 0.0, 1.0))
    r8_c = float(np.clip(r8, 0.0, 1.0))
    return float(min(1.0, max(r4_c, r8_c) + kappa * abs(r4_c - r8_c)))


def uncertainty_weight(u_task: float, tau_u: float = 0.35, temp: float = 0.10, continuous: bool = True) -> float:
    """Smooth sigmoid weighting w_U(b_t) or hard step baseline for comparison."""
    if continuous:
        return _sigmoid((float(u_task) - tau_u) / max(temp, 1e-4))
    return 1.0 if float(u_task) > tau_u else 0.0


def score_action_utility(
    action: str,
    prev_action: str,
    pred_progress: float,
    pred_delta_u: float,
    r_arb: float,
    u_task: float,
    cfg: ArbitrationConfig,
) -> float:
    """Revised Neuro-COVER-L continuous objective J_t(a)."""
    w_u = uncertainty_weight(u_task, cfg.tau_uncertainty, cfg.temp_uncertainty, cfg.use_continuous_u_weight)
    motion_cost = 1.0 if action in ("left", "right", "reverse") else (0.2 if action == "slow" else 0.0)
    switch_cost = 1.0 if (action != prev_action and action in ("left", "right") and prev_action in ("left", "right", "brake")) else (
        0.5 if (action != prev_action and action != "forward") else 0.0
    )
    return float(
        cfg.alpha_progress * pred_progress
        + cfg.beta_info_gain * w_u * pred_delta_u
        - cfg.lambda_risk * r_arb
        - cfg.rho_motion_cost * motion_cost
        - cfg.eta_switch_cost * switch_cost
    )


def select_protected_action_p5(
    obs: dict,
    belief_snap: dict,
    integrity: IntegrityReport,
    vis_out: dict,
    speed_m_s: float,
    m4_preds: Optional[Dict[str, dict]],
    m8_preds: Optional[Dict[str, dict]],
    prev_action: str,
    feasibility_checker: ActionFeasibilityChecker,
    cfg: ArbitrationConfig,
    use_visual_fusion: bool = True,
    use_risk_arbitration: bool = True,
    use_feasibility_check: bool = True,
    use_revised_utility: bool = True,
) -> Tuple[str, str]:
    """Full P5.2-P5.6 protected decision pipeline with isolated ablation switches."""
    w_left = bool(obs.get("whisker_left"))
    w_right = bool(obs.get("whisker_right"))

    # P5.2 Tier A: Unconditional tactile emergency protection (outside learned arbitration)
    if w_left and w_right:
        return "brake", "p5_2:bilateral_tactile_stop"
    if w_left or w_right:
        if use_feasibility_check:
            feas = feasibility_checker.feasible_actions(("left", "right", "brake"), belief_snap, obs, integrity, speed_m_s)
            esc_act = "right" if w_left else "left"
            if feas[esc_act][0]:
                return esc_act, f"p5_2:tactile_verified_{esc_act}"
            return "brake", f"p5_2:abstain_tactile_blocked({feas[esc_act][1]})"
        return choose_recovery_action(obs, belief_snap, looming_alert=False)

    # P5.2 Tier B: Context-conditioned biological looming / near-field collision guard
    if use_visual_fusion:
        threat, threat_rsn, _ = fuse_biological_visual_threat(vis_out, obs, integrity)
    else:
        threat, threat_rsn = (obs.get("obstacle_center", 0.0) > 0.12), "legacy_center_thresh"

    cands = ("forward", "slow", "left", "right", "brake")
    if use_feasibility_check:
        feas_map = feasibility_checker.feasible_actions(cands, belief_snap, obs, integrity, speed_m_s)
    else:
        feas_map = {a: (True, "unchecked") for a in cands}

    # Check if forward collision is imminent visually
    if threat:
        # Prefer safe verified detour if turning candidate is feasible and low arbitrated risk
        safe_turns = []
        for turn_act in ("left", "right"):
            if not feas_map[turn_act][0]:
                continue
            r_turn = 0.0
            if use_risk_arbitration and m4_preds and m8_preds:
                r_turn = compute_arbitrated_risk(m4_preds[turn_act]["contact_prob"], m8_preds[turn_act]["contact_prob"], cfg.kappa_disagree)
            if r_turn <= cfg.risk_limit_eps:
                u_turn = float(m8_preds[turn_act].get("recover_prob", 0.5)) if m8_preds else 0.5
                # Prefer continuing previous turn direction to suppress oscillation
                if turn_act == prev_action:
                    u_turn += 0.25
                safe_turns.append((u_turn - r_turn, turn_act))
        if safe_turns:
            safe_turns.sort(reverse=True)
            return safe_turns[0][1], f"p5_2:{threat_rsn}->verified_detour_{safe_turns[0][1]}"
        return "brake", f"p5_2:{threat_rsn}->protected_brake"

    # Non-healthy persistent camera check without visual threat
    if not any(feas_map[a][0] for a in ("forward", "slow", "left", "right")):
        return "brake", f"p5_3:abstain_all_infeasible({integrity.camera_status})"

    # P5.4 + P5.5: Score physically feasible actions using risk arbitration + Neuro-COVER-L utility
    if m4_preds is None or m8_preds is None:
        if feas_map["forward"][0]:
            return "forward", "p5_5:open_forward_no_learned_models"
        if feas_map["slow"][0]:
            return "slow", "p5_5:cautious_slow_no_learned_models"
        return "brake", "p5_3:abstain_unverified"

    u_task = float(belief_snap.get("task_uncertainty", 0.0))
    scored = []
    for act in cands:
        if not feas_map[act][0]:
            continue
        r4 = float(m4_preds[act]["contact_prob"])
        r8 = float(m8_preds[act]["contact_prob"])
        r_arb = compute_arbitrated_risk(r4, r8, cfg.kappa_disagree) if use_risk_arbitration else 0.5 * (r4 + r8)
        if act != "brake" and r_arb > cfg.risk_limit_eps:
            continue  # Rejected by calibrated conservative arbitration limit
        prog = float(m8_preds[act].get("progress", 0.0))
        # Map signed predicted utility/recoverability around 0.5 neutral baseline if continuous gain head absent
        du_pred = float(m8_preds[act].get("delta_u", m8_preds[act].get("recover_prob", 0.0) - 0.35))
        if use_revised_utility:
            j = score_action_utility(act, prev_action, prog, du_pred, r_arb, u_task, cfg)
        else:
            j = prog - 1.2 * r_arb
        scored.append((j, act, r_arb))

    if not scored:
        return "brake", "p5_4:abstain_risk_limit_exceeded"

    # Special disagreement protection: if single model flagged forward >= risk_limit_eps (caught above since r_arb >= max(r4,r8)),
    # forward was already filtered out, so best safe feasible ("slow", "left", "right", or "brake") wins cleanly.
    scored.sort(reverse=True)
    best_j, best_act, best_r = scored[0]
    return best_act, f"p5_5:utility_opt(a={best_act},J={best_j:.3f},R_arb={best_r:.3f})"
