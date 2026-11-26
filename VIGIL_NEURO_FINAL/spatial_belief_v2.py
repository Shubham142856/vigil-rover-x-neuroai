"""L4 Factorized World-Referenced Probabilistic Spatial Belief & Task Uncertainty U_task.

Maintains a 360-deg world-referenced angular map (30 bins, 12 deg each) relative to the
episode start/goal axis (phi = 0 rad), with separated representations:
  - p_occ(phi):    Bayesian log-odds occupancy probability in (0, 1) updated by
                   color-agnostic vertical protrusion cues + red chroma + T4/T5 motion.
  - visible(phi):  bool mask indicating whether world sector lies inside current healthy FOV.
  - ages_s(phi):   seconds since world sector was last observed cleanly.
  - u_dyn(phi):    dynamic obstacle uncertainty driven by IMU-compensated residual flow.
  - cam_healthy:   boolean sensor freshness state (frame_valid & age <= 0.051s).
  - compass_conc:  external circular concentration C_t in [0, 1] (kept decoupled).
"""
from __future__ import annotations

import math
from copy import deepcopy
from typing import Optional

import numpy as np

from t4_t5_vision import DEFAULT_CX, DEFAULT_FX


def _wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _binary_entropy_bits(p: np.ndarray) -> np.ndarray:
    """Binary Shannon entropy H_bin(p) in [0, 1] bits."""
    pc = np.clip(p, 1e-6, 1.0 - 1e-6)
    return -(pc * np.log2(pc) + (1.0 - pc) * np.log2(1.0 - pc))


class FactorizedSpatialBelief:
    """360-deg world-referenced factorized probabilistic belief (12 deg angular resolution)."""

    def __init__(
        self,
        n_sectors: int = 30,
        span_rad: float = 2.0 * math.pi,
        cam_hfov_rad: Optional[float] = None,
        max_age_s: float = 0.6,
        fx: float = DEFAULT_FX,
        cx: float = DEFAULT_CX,
    ):
        self.n_sectors = n_sectors
        self.span_rad = span_rad
        self.sector_width = span_rad / n_sectors
        self.centers = np.linspace(-math.pi, math.pi, n_sectors, endpoint=False) + 0.5 * self.sector_width
        self.centers = _wrap_angle(self.centers)
        if cam_hfov_rad is None:
            cam_hfov_rad = 2.0 * math.atan(64.0 / fx)  # ≈ 81.78 deg matching 128px @ 73.914px focal
        self.cam_hfov_rad = cam_hfov_rad
        self.max_age_s = max_age_s
        self.fx = fx
        self.cx = cx

        # Explicitly separated world-referenced state vectors
        self.log_odds = np.zeros(n_sectors, dtype=np.float64)
        self.near_blocked = np.zeros(n_sectors, dtype=bool)
        self.visible = np.zeros(n_sectors, dtype=bool)
        self.last_obs_s = np.full(n_sectors, -np.inf, dtype=np.float64)
        self.u_dyn = np.zeros(n_sectors, dtype=np.float64)
        self.yaw_est_rad = 0.0
        self.heading_uncertainty_rad = 0.02
        self.compass_conc = 1.0
        self.camera_healthy = False
        self._last_time_s: Optional[float] = None

    def copy(self) -> FactorizedSpatialBelief:
        return deepcopy(self)

    def advance_ego_yaw(self, delta_yaw_rad: float) -> None:
        """Update internal world heading estimate without discarding unseen world sectors."""
        self.yaw_est_rad = float(_wrap_angle(self.yaw_est_rad + float(delta_yaw_rad)))

    @property
    def occupancy_probs(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(self.log_odds, -10.0, 10.0)))

    def ages(self, now: float) -> np.ndarray:
        return np.where(np.isfinite(self.last_obs_s), np.maximum(0.0, now - self.last_obs_s), np.inf)

    def _ego_index(self, ego_angle_rad: float) -> int:
        world_ang = _wrap_angle(self.yaw_est_rad + ego_angle_rad)
        diffs = np.abs(_wrap_angle(self.centers - world_ang))
        return int(np.argmin(diffs))

    def ego_window(self, n_ego: int = 9, fov_rad: float = math.radians(78.0)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample egocentric (occupancy_probs, last_obs_s, near_blocked) centered around current yaw_est_rad."""
        ego_angles = np.linspace(-0.5 * fov_rad, 0.5 * fov_rad, n_ego)
        p_occ = self.occupancy_probs
        occ_ego = np.empty(n_ego, dtype=np.float64)
        obs_ego = np.empty(n_ego, dtype=np.float64)
        near_ego = np.zeros(n_ego, dtype=bool)
        for i, ang in enumerate(ego_angles):
            idx = self._ego_index(float(ang))
            occ_ego[i] = p_occ[idx]
            obs_ego[i] = self.last_obs_s[idx]
            near_ego[i] = bool(self.near_blocked[idx])
        return occ_ego, obs_ego, near_ego

    def update(
        self,
        obs: dict,
        delta_yaw_rad: Optional[float] = None,
        heading_unc_delta_rad: float = 0.0,
        res_motion_cols: Optional[np.ndarray] = None,
        compass_conc: Optional[float] = None,
        world_yaw_rad: Optional[float] = None,
    ) -> dict:
        now = float(obs["time_s"])
        dt = 0.05 if self._last_time_s is None else max(0.0, now - self._last_time_s)
        self._last_time_s = now

        if world_yaw_rad is not None:
            self.yaw_est_rad = float(_wrap_angle(world_yaw_rad))
        elif delta_yaw_rad is not None:
            self.advance_ego_yaw(delta_yaw_rad)
        else:
            gyro_z = obs.get("imu_gyro_z")
            if gyro_z is not None and np.isfinite(gyro_z):
                self.advance_ego_yaw(float(gyro_z) * dt)
            else:
                self.heading_uncertainty_rad = min(math.pi, self.heading_uncertainty_rad + 0.08 * dt)

        self.heading_uncertainty_rad = float(np.clip(self.heading_uncertainty_rad + heading_unc_delta_rad, 0.01, math.pi))

        # World-referenced belief uncertainty grows back toward prior as observations age
        self.log_odds *= math.exp(-dt / 4.0)
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

        # 1. Strictly above-horizon obstacle protrusion cue (rows 18:40 avoid flat ground below row 40)
        sky_ref = float(np.percentile(gray[:14, :], 75))
        dark_horizon = (gray[18:40, :] < 0.72 * max(sky_ref, 40.0)).mean(axis=0)
        red_hit = ((r > 60.0) & (r > 1.7 * g) & (r > 1.7 * b)).mean(axis=0)
        col_occ_cue = (dark_horizon > 0.22) | (red_hit > 0.08)
        # Near-field range proxy (large subtended vertical angle <= ~0.42 m stopping horizon)
        col_near_cue = (dark_horizon > 0.55) | (red_hit > 0.42)

        # Camera column offset: col 0 is LEFT (+alpha_c), col 127 is RIGHT (-alpha_c)
        col_ego_angles = -np.arctan((np.arange(rgb.shape[1]) - self.cx) / self.fx)
        fov_half = 0.5 * self.cam_hfov_rad
        half_w = 0.5 * self.sector_width

        self.visible.fill(False)
        for k in range(self.n_sectors):
            rel_ang = float(_wrap_angle(self.centers[k] - self.yaw_est_rad))
            if abs(rel_ang) <= fov_half:
                in_sec = np.abs(_wrap_angle(col_ego_angles - rel_ang)) <= half_w
                if in_sec.any():
                    self.visible[k] = True
                    self.last_obs_s[k] = now
                    self.near_blocked[k] = bool(col_near_cue[in_sec].any())
                    if col_occ_cue[in_sec].any():
                        self.log_odds[k] = min(4.0, self.log_odds[k] + 2.2)
                    else:
                        self.log_odds[k] = max(-3.0, self.log_odds[k] - 1.6)
                    if res_motion_cols is not None:
                        sec_mot = float(res_motion_cols[in_sec].mean())
                        self.u_dyn[k] = min(1.0, self.u_dyn[k] + 0.4 * sec_mot)

        return self.snapshot(now)

    def task_uncertainty(self, now: float, gamma_age: float = 0.4, gamma_dyn: float = 0.35) -> float:
        """Compute world-referenced decision-relevant uncertainty U_task over fixed goal region.

        World sectors in [-82 deg, +82 deg] around the fixed goal axis (+x, phi = 0) define
        the stable comparison region across actions, ensuring turning away from an unknown
        region never creates fake uncertainty reduction.
        """
        p_occ = self.occupancy_probs
        ages = self.ages(now)
        center_mask = np.abs(self.centers) <= math.radians(24.0)
        flank_mask = (np.abs(self.centers) > math.radians(24.0)) & (np.abs(self.centers) <= math.radians(82.0))

        # If goal forward corridor in world coordinates is blocked/ambiguous, flank bypass uncertainty matters
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

        # Build egocentric 9-sector view around yaw_est_rad for compatibility with recovery/controller checks
        occ_ego, obs_ego, near_ego = self.ego_window(n_ego=9, fov_rad=math.radians(78.0))
        ego_ages = np.where(np.isfinite(obs_ego), np.maximum(0.0, now - obs_ego), np.inf)
        fov9_states = []
        for s in range(len(occ_ego)):
            if occ_ego[s] >= 0.60:
                fov9_states.append("occupied")
            elif not np.isfinite(ego_ages[s]) or ego_ages[s] > self.max_age_s or occ_ego[s] > 0.30:
                fov9_states.append("stale_or_unknown")
            else:
                fov9_states.append("observed_free")

        return {
            "camera_healthy": bool(self.camera_healthy),
            "sector_states": fov9_states,
            "near_blocked_sectors": [bool(v) for v in near_ego],
            "sector_ages_s": [None if not np.isfinite(a) else round(float(a), 4) for a in ego_ages],
            "occupancy_probs": [round(float(v), 4) for v in occ_ego],
            "world_occupancy_probs": [round(float(v), 4) for v in p_occ],
            "visibility": [bool(v) for v in self.visible],
            "dynamic_uncertainty": [round(float(v), 4) for v in self.u_dyn],
            "yaw_est_rad": float(self.yaw_est_rad),
            "heading_uncertainty_rad": float(self.heading_uncertainty_rad),
            "compass_concentration": float(self.compass_conc),
            "task_uncertainty": self.task_uncertainty(now),
            "free_fraction": float(sum(st == "observed_free" for st in fov9_states) / len(fov9_states)),
            "unknown_or_stale_fraction": float(sum(st != "observed_free" for st in fov9_states) / len(fov9_states)),
        }
