"""P6.3 & P6.4: Calibrated Ground-Contact IPM Range Estimation & Adaptive Braking Model.

Replaces height-agnostic column-fraction heuristics (which confuse a 1.2 m pedestrian at
1.05 m with a close wall) with pinhole Inverse Perspective Mapping (IPM) to ground contact,
paired with an analytical adaptive braking envelope validated against PyBullet physics.

Uses calibrated intrinsics (fx=fy=73.914 px, cx=63.5, cy=47.5) and camera extrinsics
matching RoverSim/RealWorldEnv (eye_x=0.09 m forward, eye_z=0.14 m, tilt=0.04 rad down).
Strictly zero simulator oracle fields used.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np

from t4_t5_vision import DEFAULT_CX, DEFAULT_CY, DEFAULT_FX, DEFAULT_FY


CAM_HEIGHT_M = 0.14
CAM_OFFSET_X_M = 0.09
CAM_TILT_DOWN_RAD = math.atan2(0.04, 1.0)  # look target z=0.10 at 1.0 m ahead from z=0.14


def estimate_ground_contact_clearance(
    rgb: np.ndarray,
    pitch_rad: float = 0.0,
    fx: float = DEFAULT_FX,
    fy: float = DEFAULT_FY,
    cx: float = DEFAULT_CX,
    cy: float = DEFAULT_CY,
    cam_h: float = CAM_HEIGHT_M,
    cam_tilt: float = CAM_TILT_DOWN_RAD,
    bumper_x_m: float = 0.10,
) -> Dict[str, float]:
    """Estimate ground-plane forward clearance [m] from camera RGB via bottom ground-contact IPM.

    Separates upper protrusion silhouettes (red or darker than sky reference) above the floor
    horizon and projects the lowest contiguous obstacle row v_bot in the central collision strip
    onto the ground plane Z = h_cam / tan(tilt + pitch + atan((v_bot - cy)/fy)).
    """
    h, w, _ = rgb.shape
    arr = rgb.astype(np.float64)
    r, g, b = arr.transpose(2, 0, 1)
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    # Central horizontal corridor strip matching calibrated rover chassis footprint at ~0.35 m
    # atan(0.10m / 0.35m) * 73.91px ≈ 20.8px -> cx(63.5) +- 21.5 -> [42, 86]
    x_lo, x_hi = 42, 86
    gray_c = gray[:, x_lo:x_hi]
    r_c, g_c, b_c = r[:, x_lo:x_hi], g[:, x_lo:x_hi], b[:, x_lo:x_hi]

    # Floor color baseline estimated from clean bottom corner patches if unobstructed, else bright sky ref
    sky_ref = float(np.percentile(gray[:14, :], 75))
    floor_ref = float(np.median(gray[h - 12 :, :]))

    # Obstacle row indicators:
    # 1. Red chromatic targets (posts, carts, vests)
    red_mask = (r_c > 55.0) & (r_c > 1.65 * g_c) & (r_c > 1.65 * b_c)
    red_row_frac = red_mask.mean(axis=1)

    # 2. Achromatic / dark silhouette distinct from both sky and uniform floor:
    horizon_dark_cols = (gray_c[18:40, :] < 0.72 * max(sky_ref, 40.0)).mean(axis=0) > 0.22
    has_horiz_obstacle = bool(np.any(horizon_dark_cols) or np.any(red_row_frac[12:55] > 0.12))

    if not has_horiz_obstacle:
        return {
            "obstacle_detected": False,
            "v_bot_px": 0.0,
            "estimated_range_m": 4.0,
            "estimated_clearance_m": 3.9,
            "range_uncertainty_m": 0.25,
            "ipm_reliable": True,
        }

    # Trace lowest CONTIGUOUS obstacle row descending from upper horizon band [16..52].
    # Prevents isolated dark floor tile cracks/shadows near bottom rows from causing false 0.21m flickers.
    dark_obj = (gray_c < 0.78 * max(floor_ref, 35.0)) & (np.abs(gray_c - floor_ref) > 14.0)
    occ_row = (red_row_frac > 0.10) | (dark_obj.mean(axis=1) > 0.20)

    upper_seeds = np.where(occ_row[16:54])[0] + 16
    if len(upper_seeds) == 0:
        return {
            "obstacle_detected": False,
            "v_bot_px": 0.0,
            "estimated_range_m": 4.0,
            "estimated_clearance_m": 3.9,
            "range_uncertainty_m": 0.30,
            "ipm_reliable": False,
        }

    row_cur = int(upper_seeds[0])
    v_bot_cont = row_cur
    gap = 0
    for r_idx in range(row_cur, h - 1):
        if occ_row[r_idx]:
            v_bot_cont = r_idx
            gap = 0
        else:
            gap += 1
            if gap > 2:
                # Check if a later horizon-connected obstacle starts inside [16:54] before terminating
                if r_idx < 54 and np.any(occ_row[r_idx + 1 : 54]):
                    next_seed = r_idx + 1 + int(np.argmax(occ_row[r_idx + 1 : 54]))
                    v_bot_cont = next_seed
                    gap = 0
                    continue
                break

    v_bot = float(v_bot_cont)
    # Angular depression below camera principal ray
    alpha_ray = math.atan2(v_bot - cy, fy)
    alpha_eff = cam_tilt + float(pitch_rad) + alpha_ray

    # Pixel quantization (+-1.5 px) and pitch (+-0.015 rad) uncertainty propagation
    if alpha_eff <= 0.025:
        # Horizon or above-horizon obstacle: beyond reliable IPM floor distance
        z_est = 2.0
        sigma_z = 0.75
        ipm_ok = False
    else:
        z_cam = cam_h / math.tan(alpha_eff)
        z_est = float(np.clip(z_cam + CAM_OFFSET_X_M, 0.10, 4.0))
        d_z_d_alpha = cam_h / (math.sin(alpha_eff) ** 2 + 1e-6)
        sigma_alpha = math.hypot(1.5 / fy, 0.015)
        sigma_z = float(min(1.5, d_z_d_alpha * sigma_alpha))
        ipm_ok = True

    clearance = max(0.0, z_est - bumper_x_m)
    return {
        "obstacle_detected": True,
        "v_bot_px": v_bot,
        "estimated_range_m": z_est,
        "estimated_clearance_m": clearance,
        "range_uncertainty_m": sigma_z,
        "ipm_reliable": ipm_ok,
    }


@dataclass
class AdaptiveBrakingModel:
    """P6.4 Physical stopping horizon and distance envelope matching PyBullet rover dynamics."""

    tau_reaction_s: float = 0.10    # motor lag (~0.08 s) + half control step (0.025 s)
    base_decel_m_s2: float = 0.80   # braking ramp 0.80 m/s^2 in RoverSim & nominal tile friction mu=0.6
    d_margin_m: float = 0.05        # conservative bumper margin (half whisker length 0.10 m)

    def effective_decel(self, friction_mu: float = 0.60) -> float:
        grip_scale = min(1.0, max(0.30, float(friction_mu)) / 0.60)
        return self.base_decel_m_s2 * grip_scale

    def stopping_horizon_s(self, speed_m_s: float, friction_mu: float = 0.60) -> float:
        v = max(0.0, abs(float(speed_m_s)))
        return float(self.tau_reaction_s + v / max(self.effective_decel(friction_mu), 0.1))

    def stopping_distance_m(self, speed_m_s: float, friction_mu: float = 0.60, extra_margin_m: float = 0.0) -> float:
        v = max(0.0, abs(float(speed_m_s)))
        a = max(self.effective_decel(friction_mu), 0.1)
        return float(v * self.tau_reaction_s + (v * v) / (2.0 * a) + self.d_margin_m + max(0.0, extra_margin_m))
