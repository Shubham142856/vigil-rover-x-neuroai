"""L4 Factorized Probabilistic Spatial Belief & Decision-Relevant Task Uncertainty U_task.

Replaces red-only single-flag sector memory with strictly separated probabilistic
representations over a 180-deg forward hemisphere (15 angular sectors, 12 deg each):
  - p_occ(s):      Bayesian log-odds occupancy probability in (0, 1), updated by
                   color-agnostic vertical protrusion cues + red chroma + T4/T5 motion.
  - visible(s):    bool mask indicating whether sector s falls inside current healthy FOV
                   without being occluded.
  - ages_s(s):     seconds since sector s was last cleanly observed.
  - u_dyn(s):      dynamic obstacle uncertainty driven by IMU-compensated residual flow.
  - cam_healthy:   boolean sensor freshness state (frame_valid & age <= 0.051s).
  - compass_conc:  external circular concentration C_t in [0, 1] (kept decoupled).
"""
from __future__ import annotations

import math
from copy import deepcopy
from typing import Optional

import numpy as np

from t4_t5_vision import DEFAULT_CX, DEFAULT_FX


def _binary_entropy_bits(p: np.ndarray) -> np.ndarray:
    """Binary Shannon entropy H_bin(p) in [0, 1] bits."""
    pc = np.clip(p, 1e-6, 1.0 - 1e-6)
    return -(pc * np.log2(pc) + (1.0 - pc) * np.log2(1.0 - pc))


class FactorizedSpatialBelief:
    """Factorized 15-sector probabilistic belief spanning [-90 deg, +90 deg] ego bearing."""

    def __init__(
        self,
        n_sectors: int = 15,
        span_rad: float = math.pi,
        cam_hfov_rad: Optional[float] = None,
        max_age_s: float = 0.6,
        fx: float = DEFAULT_FX,
        cx: float = DEFAULT_CX,
    ):
        self.n_sectors = n_sectors
        self.span_rad = span_rad
        self.edges = np.linspace(-span_rad / 2.0, span_rad / 2.0, n_sectors + 1)
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.sector_width = span_rad / n_sectors
        if cam_hfov_rad is None:
            cam_hfov_rad = 2.0 * math.atan(64.0 / fx)  # ≈ 81.78 deg matching 128px @ 73.914px focal
        self.cam_hfov_rad = cam_hfov_rad
        self.max_age_s = max_age_s
        self.fx = fx
        self.cx = cx

        # Explicitly separated state vectors (never collapsed prematurely)
        self.log_odds = np.zeros(n_sectors, dtype=np.float64)  # 0.0 -> prior p_occ = 0.5
        self.visible = np.zeros(n_sectors, dtype=bool)
        self.last_obs_s = np.full(n_sectors, -np.inf, dtype=np.float64)
        self.u_dyn = np.zeros(n_sectors, dtype=np.float64)
        self.compass_conc = 1.0
        self.camera_healthy = False
        self._last_time_s: Optional[float] = None

    def copy(self) -> FactorizedSpatialBelief:
        return deepcopy(self)

    def advance_ego_yaw(self, delta_yaw_rad: float) -> None:
        """Rotate stored spatial belief vectors opposite to rover delta_yaw (+left shifts world right)."""
        if abs(delta_yaw_rad) < 1e-7:
            return
        # Rover yaw +delta_yaw shifts world landmarks by -delta_yaw in ego angle coordinates.
        # Query old grid at theta_old = theta_new + delta_yaw; boundary reverts to unknown prior.
        query = self.centers + float(delta_yaw_rad)
        self.log_odds = np.interp(query, self.centers, self.log_odds, left=0.0, right=0.0)
        self.u_dyn = np.interp(query, self.centers, self.u_dyn, left=0.0, right=0.0)
        # Interpolate observation timestamps carefully (finite check)
        prev_t = np.where(np.isfinite(self.last_obs_s), self.last_obs_s, -1e9)
        interp_t = np.interp(query, self.centers, prev_t, left=-1e9, right=-1e9)
        self.last_obs_s = np.where(interp_t > -1e8, interp_t, -np.inf)

    @property
    def occupancy_probs(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(self.log_odds, -10.0, 10.0)))

    def ages(self, now: float) -> np.ndarray:
        return np.where(np.isfinite(self.last_obs_s), np.maximum(0.0, now - self.last_obs_s), np.inf)

    def update(
        self,
        obs: dict,
        gyro_z_rad_s: Optional[float] = None,
        res_motion_cols: Optional[np.ndarray] = None,
        compass_conc: Optional[float] = None,
    ) -> dict:
        now = float(obs["time_s"])
        dt = 0.05 if self._last_time_s is None else max(0.0, now - self._last_time_s)
        self._last_time_s = now

        if gyro_z_rad_s is None:
            gyro_z_rad_s = float(obs.get("imu_gyro_z", 0.0))
        self.advance_ego_yaw(float(gyro_z_rad_s) * dt)

        # Gentle uncertainty regression toward prior over unseen time
        self.log_odds *= math.exp(-dt / 3.0)
        self.u_dyn *= math.exp(-dt / 0.6)
        if compass_conc is not None:
            self.compass_conc = float(np.clip(compass_conc, 0.0, 1.0))

        fresh = (
            obs.get("frame_valid") is True
            and obs.get("frame_age_s") is not None
            and float(obs["frame_age_s"]) <= 0.051
        )
        if not fresh:
            self.camera_healthy = False
            self.visible.fill(False)
            return self.snapshot(now)

        self.camera_healthy = True
        rgb = obs["rgb"]
        r, g, b = rgb.astype(np.float64).transpose(2, 0, 1)
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        # 1. Color-agnostic upper-horizon vertical obstacle protrusion cue
        sky_ref = float(np.percentile(gray[:14, :], 75))
        dark_protrusion = (gray[18:40, :] < 0.72 * max(sky_ref, 40.0)).mean(axis=0)
        # 2. Classic red obstacle chroma cue (preserves distant red post detection)
        red_hit = ((r > 60.0) & (r > 1.7 * g) & (r > 1.7 * b)).mean(axis=0)
        col_occ_cue = (dark_protrusion > 0.22) | (red_hit > 0.08)

        col_angles = np.arctan((np.arange(rgb.shape[1]) - self.cx) / self.fx)
        fov_half = 0.5 * self.cam_hfov_rad

        self.visible.fill(False)
        for s in range(self.n_sectors):
            in_fov = abs(self.centers[s]) <= fov_half
            in_sec = (col_angles >= self.edges[s]) & (col_angles <= self.edges[s + 1])
            if in_fov and in_sec.any():
                self.visible[s] = True
                self.last_obs_s[s] = now
                if col_occ_cue[in_sec].any():
                    self.log_odds[s] = min(4.0, self.log_odds[s] + 2.2)
                else:
                    self.log_odds[s] = max(-3.0, self.log_odds[s] - 1.6)
                if res_motion_cols is not None:
                    sec_mot = float(res_motion_cols[in_sec].mean())
                    self.u_dyn[s] = min(1.0, self.u_dyn[s] + 0.4 * sec_mot)

        return self.snapshot(now)

    def task_uncertainty(self, now: float, gamma_age: float = 0.4, gamma_dyn: float = 0.35) -> float:
        """Compute decision-relevant uncertainty U_task(b_t) in bits + staleness + dynamic ambiguity.

        Corridor attention weights:
          - If center planning cone (|theta| <= 24 deg) is open/clear, weights focus on the
            forward cone (which drops toward 0 in open space so turns yield zero info gain).
          - If center planning cone is blocked or ambiguous (p_occ > 0.45), attention shifts
            to bypass flank corridors (24 deg .. 80 deg) to measure recoverability of side detours.
        """
        p_occ = self.occupancy_probs
        ages = self.ages(now)
        center_mask = np.abs(self.centers) <= math.radians(24.0)
        flank_mask = (np.abs(self.centers) > math.radians(24.0)) & (np.abs(self.centers) <= math.radians(82.0))

        forward_blocked = bool(np.any(p_occ[center_mask] > 0.45))
        weights = np.zeros(self.n_sectors, dtype=np.float64)
        if forward_blocked:
            weights[center_mask] = 0.25
            weights[flank_mask] = 1.0
        else:
            weights[center_mask] = 1.0
            weights[flank_mask] = 0.0
        weights /= max(float(weights.sum()), 1e-9)

        ent = _binary_entropy_bits(p_occ)
        # Grace window of 0.10s (one 10Hz camera period) avoids inter-frame age sawtooth noise
        eff_age = np.maximum(0.0, ages - 0.10)
        denom = max(self.max_age_s - 0.10, 1e-3)
        age_norm = np.where(np.isfinite(ages), np.minimum(1.0, eff_age / denom), 1.0)
        per_sector_u = ent + gamma_age * age_norm + gamma_dyn * self.u_dyn
        return float(np.sum(weights * per_sector_u))

    def snapshot(self, now: Optional[float] = None) -> dict:
        if now is None:
            now = 0.0 if self._last_time_s is None else self._last_time_s
        p_occ = self.occupancy_probs
        ages = self.ages(now)
        states = []
        for s in range(self.n_sectors):
            if p_occ[s] >= 0.60:
                states.append("occupied")
            elif not np.isfinite(ages[s]) or ages[s] > self.max_age_s or p_occ[s] > 0.30:
                states.append("stale_or_unknown")
            else:
                states.append("observed_free")
        # Legacy compatibility: also expose center-9 corridor states matching SectorBelief indices
        mid_lo = (self.n_sectors - 9) // 2
        fov9_states = states[mid_lo : mid_lo + 9] if self.n_sectors >= 9 else states
        return {
            "camera_healthy": bool(self.camera_healthy),
            "sector_states": fov9_states,
            "full_sector_states": states,
            "sector_ages_s": [None if not np.isfinite(a) else round(float(a), 4) for a in ages],
            "occupancy_probs": [round(float(v), 4) for v in p_occ],
            "visibility": [bool(v) for v in self.visible],
            "dynamic_uncertainty": [round(float(v), 4) for v in self.u_dyn],
            "compass_concentration": float(self.compass_conc),
            "task_uncertainty": self.task_uncertainty(now),
            "free_fraction": float(sum(st == "observed_free" for st in fov9_states) / len(fov9_states)),
            "unknown_or_stale_fraction": float(sum(st != "observed_free" for st in fov9_states) / len(fov9_states)),
        }
