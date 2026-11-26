"""L1/L2 Dual-Pathway Drosophila T4/T5 Motion & Ego-Motion-Compensated Looming Vision.

Implements the perceptual reconstruction in VIGIL-ROVER X NeuroAI V2:
  1. Calibrated pinhole camera geometry matching PyBullet 66 deg vertical FOV
     on 96x128 (fx = fy ≈ 73.914 px, cx = 63.5, cy = 47.5).
  2. Compound luminance high-pass filtering split into ON (T4) and OFF (T5) channels,
     driving 4-direction Hassenstein-Reichardt correlators (left, right, up, down)
     independent of surface color (works on gray walls, dark objects, and red targets).
  3. Dual-pathway evaluation:
       - Pathway A (Raw visual motion): preserves direct looming / stationary-wall approach.
       - Pathway B (IMU-rotated residual motion): warps delayed state by exact pinhole
         3D rotation homography K R(omega_y dt, omega_z dt) K^-1 to reject camera yaw and
         ramp-pitch self-motion while isolating moving obstacles.
  4. LPLC2 4-quadrant outbound opponency + LC4 collision-course balance gating.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


DEFAULT_FX = 96.0 / (2.0 * math.tan(math.radians(66.0) / 2.0))
DEFAULT_FY = DEFAULT_FX
DEFAULT_CX = 63.5
DEFAULT_CY = 47.5


def to_normalized_luminance(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB [H, W, 3] to contrast-normalized luminance [0, 1]."""
    arr = rgb.astype(np.float64)
    lum = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]) / 255.0
    return np.clip(lum, 0.0, 1.0)


def warp_rotation_homography(
    img: np.ndarray,
    delta_yaw_rad: float,
    delta_pitch_rad: float = 0.0,
    fx: float = DEFAULT_FX,
    fy: float = DEFAULT_FY,
    cx: float = DEFAULT_CX,
    cy: float = DEFAULT_CY,
) -> np.ndarray:
    """Warp 2D scalar image forward from t-1 to t under camera yaw (+CCW left) & pitch (+nose down).

    Uses inverse mapping p_{t-1} ~ K R_cam^T K^{-1} p_t with bilinear interpolation.
    In right-handed camera coordinates (X=right, Y=down, Z=forward):
      - Rover yaw +d_yaw (left) rotates scene right -> Ry(+d_yaw) passive transform
      - Rover pitch +d_pitch (down) rotates scene up -> Rx(+d_pitch) passive transform
    """
    if abs(delta_yaw_rad) < 1e-7 and abs(delta_pitch_rad) < 1e-7:
        return img.copy()

    h, w = img.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    xn = (xs - cx) / fx
    yn = (ys - cy) / fy

    # Passive scene rotation R_cam = R_x(d_pitch) @ R_y(d_yaw)
    cyw, syw = math.cos(delta_yaw_rad), math.sin(delta_yaw_rad)
    cpt, spt = math.cos(delta_pitch_rad), math.sin(delta_pitch_rad)
    # R_y(d_yaw): [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    # R_x(d_pitch): [[1, 0, 0], [0, c, s], [0, -s, c]] (nose down moves +Z up -> -Y)
    r_yaw = np.array([[cyw, 0.0, syw], [0.0, 1.0, 0.0], [-syw, 0.0, cyw]], dtype=np.float64)
    r_pitch = np.array([[1.0, 0.0, 0.0], [0.0, cpt, spt], [0.0, -spt, cpt]], dtype=np.float64)
    r_inv = (r_pitch @ r_yaw).T  # inverse mapping dst(t) -> src(t-1)

    sx = r_inv[0, 0] * xn + r_inv[0, 1] * yn + r_inv[0, 2]
    sy = r_inv[1, 0] * xn + r_inv[1, 1] * yn + r_inv[1, 2]
    sz = np.where(np.abs(r_inv[2, 0] * xn + r_inv[2, 1] * yn + r_inv[2, 2]) < 1e-6, 1e-6,
                  r_inv[2, 0] * xn + r_inv[2, 1] * yn + r_inv[2, 2])

    u = fx * (sx / sz) + cx
    v = fy * (sy / sz) + cy
    valid = (u >= 0.0) & (u <= w - 1.0) & (v >= 0.0) & (v <= h - 1.0)
    uc = np.clip(u, 0.0, w - 1.0)
    vc = np.clip(v, 0.0, h - 1.0)

    x0 = np.floor(uc).astype(int)
    y0 = np.floor(vc).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = uc - x0
    wy = vc - y0

    sampled = (
        img[y0, x0] * (1.0 - wy) * (1.0 - wx)
        + img[y1, x0] * wy * (1.0 - wx)
        + img[y0, x1] * (1.0 - wy) * wx
        + img[y1, x1] * wy * wx
    )
    # Border fallback to current coordinates on invalid frustum margins prevents sharp border edges
    return np.where(valid, sampled, img[np.clip(ys.astype(int), 0, h - 1), np.clip(xs.astype(int), 0, w - 1)])


class _SinglePathT4T5:
    """Stateful ON/OFF Hassenstein-Reichardt correlator bank with 4-quadrant LPLC2 readout."""

    def __init__(
        self,
        tau_h: float = 0.25,
        tau_l: float = 0.05,
        contrast_thresh: float = 0.015,
        quad_thresh: float = 0.02,
    ):
        self.tau_h = tau_h
        self.tau_l = tau_l
        self.contrast_thresh = contrast_thresh
        self.quad_thresh = quad_thresh
        self.prev_lum: Optional[np.ndarray] = None
        self.high: Optional[np.ndarray] = None
        self.lag_on: Optional[np.ndarray] = None
        self.lag_off: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.prev_lum = None
        self.high = None
        self.lag_on = None
        self.lag_off = None

    def update(self, lum: np.ndarray, dt: float, warp_fn=None) -> dict:
        h, w = lum.shape
        dt = max(float(dt), 1e-3)
        if self.prev_lum is None:
            self.prev_lum = lum.copy()
            self.high = np.zeros_like(lum)
            self.lag_on = np.zeros_like(lum)
            self.lag_off = np.zeros_like(lum)
            return {
                "lplc2": 0.0,
                "lc4": 0.0,
                "t4_energy": 0.0,
                "t5_energy": 0.0,
                "lateral_imbalance": 0.0,
                "quad_out": (0.0, 0.0, 0.0, 0.0),
            }

        prev_l = self.prev_lum if warp_fn is None else warp_fn(self.prev_lum)
        prev_h = self.high if warp_fn is None else warp_fn(self.high)
        prev_lon = self.lag_on if warp_fn is None else warp_fn(self.lag_on)
        prev_loff = self.lag_off if warp_fn is None else warp_fn(self.lag_off)

        alpha_h = self.tau_h / (self.tau_h + dt)
        alpha_l = self.tau_l / (self.tau_l + dt)

        self.high = alpha_h * (prev_h + lum - prev_l)
        on_ch = np.maximum(self.high - self.contrast_thresh, 0.0)
        off_ch = np.maximum(-self.high - self.contrast_thresh, 0.0)

        # Causal delayed channels: combine warped lagged filter with prior step, then update after EMD
        lag_on = alpha_l * prev_lon + (1.0 - alpha_l) * on_ch
        lag_off = alpha_l * prev_loff + (1.0 - alpha_l) * off_ch

        # T4 (ON) & T5 (OFF) 4-cardinal EMD arrays:
        # horizontal > 0: rightward motion; vertical > 0: downward motion
        t4_h = prev_lon[:-1, :-1] * on_ch[:-1, 1:] - on_ch[:-1, :-1] * prev_lon[:-1, 1:]
        t5_h = prev_loff[:-1, :-1] * off_ch[:-1, 1:] - off_ch[:-1, :-1] * prev_loff[:-1, 1:]
        t4_v = prev_lon[:-1, :-1] * on_ch[1:, :-1] - on_ch[:-1, :-1] * prev_lon[1:, :-1]
        t5_v = prev_loff[:-1, :-1] * off_ch[1:, :-1] - off_ch[:-1, :-1] * prev_loff[1:, :-1]

        self.lag_on = lag_on
        self.lag_off = lag_off
        self.prev_lum = lum.copy()

        horiz = t4_h + t5_h
        vert = t4_v + t5_v

        cy, cx = (h - 1) // 2, (w - 1) // 2
        half = min(38, cy, cx)
        strip = min(18, cy, cx)

        # Normalize quadrant outward flow by pixel count for scale-invariance across strip sizes
        ql = -horiz[cy - strip : cy + strip, cx - half : cx]
        qr = horiz[cy - strip : cy + strip, cx : cx + half]
        qu = -vert[cy - half : cy, cx - strip : cx + strip]
        qd = vert[cy : cy + half, cx - strip : cx + strip]

        q_out = np.array([
            float(np.maximum(ql, 0.0).mean()) * 1e3,
            float(np.maximum(qr, 0.0).mean()) * 1e3,
            float(np.maximum(qu, 0.0).mean()) * 1e3,
            float(np.maximum(qd, 0.0).mean()) * 1e3,
        ], dtype=np.float64)

        active = np.maximum(q_out - self.quad_thresh, 0.0)
        # Geometric-mean LPLC2 radial opponency: drops to 0 if any quadrant fails to expand
        lplc2 = float(np.prod(active) ** 0.25)

        # Lateral drift suppression (collision-course vs crossing/near-miss):
        # Uniform horizontal translation makes ql and qr opposite in sign -> net lateral energy
        net_lat = abs(float(horiz[cy - strip : cy + strip, cx - half : cx + half].mean()) * 1e3)
        sym_exp = 0.5 * (q_out[0] + q_out[1])
        radial_gate = max(0.0, 1.0 - net_lat / max(sym_exp + 1e-6, 0.15))

        lc4 = float(q_out.mean() * radial_gate)

        return {
            "lplc2": lplc2,
            "lc4": lc4,
            "t4_energy": float((np.mean(np.abs(t4_h)) + np.mean(np.abs(t4_v))) * 1e3),
            "t5_energy": float((np.mean(np.abs(t5_h)) + np.mean(np.abs(t5_v))) * 1e3),
            "lateral_imbalance": float(net_lat),
            "quad_out": tuple(float(v) for v in q_out),
        }


class DualPathwayT4T5Vision:
    """Dual-pathway L1/L2 visual processor: Raw expansion + IMU ego-motion residual."""

    def __init__(
        self,
        fx: float = DEFAULT_FX,
        fy: float = DEFAULT_FY,
        cx: float = DEFAULT_CX,
        cy: float = DEFAULT_CY,
    ):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.raw_path = _SinglePathT4T5()
        self.residual_path = _SinglePathT4T5()
        self._last_lum: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.raw_path.reset()
        self.residual_path.reset()
        self._last_lum = None

    def step(
        self,
        rgb: np.ndarray,
        dt: float,
        gyro_z_rad_s: float = 0.0,
        gyro_y_rad_s: float = 0.0,
    ) -> dict:
        lum = to_normalized_luminance(rgb)
        raw_out = self.raw_path.update(lum, dt, warp_fn=None)

        dyaw = float(gyro_z_rad_s) * float(dt)
        dpitch = float(gyro_y_rad_s) * float(dt)
        warp_fn = None if (abs(dyaw) < 1e-7 and abs(dpitch) < 1e-7) else (
            lambda arr: warp_rotation_homography(
                arr, delta_yaw_rad=dyaw, delta_pitch_rad=dpitch,
                fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
            )
        )
        res_out = self.residual_path.update(lum, dt, warp_fn=warp_fn)
        return {
            "raw_lplc2": raw_out["lplc2"],
            "raw_lc4": raw_out["lc4"],
            "raw_t4": raw_out["t4_energy"],
            "raw_t5": raw_out["t5_energy"],
            "residual_lplc2": res_out["lplc2"],
            "residual_lc4": res_out["lc4"],
            "residual_t4": res_out["t4_energy"],
            "residual_t5": res_out["t5_energy"],
            "lateral_imbalance": res_out["lateral_imbalance"],
            "quad_out_raw": raw_out["quad_out"],
            "quad_out_residual": res_out["quad_out"],
        }
