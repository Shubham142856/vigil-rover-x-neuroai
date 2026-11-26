"""
real_sensor_model.py — Realistic sensor noise models for VIGIL-ROVER X.

Models are grounded in published sensor datasheets and calibration literature:
  Camera:   OV2640 / Raspberry Pi Camera Module 2 characteristics
  IMU:      MPU-6050 datasheet (noise density, bias instability)
  Encoder:  Generic quadrature encoder, 360 CPR
  Whisker:  Mechanical contact switch (debounce, false-trigger)

All computations are pure-NumPy; no cv2 required.
All parameters are documented with their physical source.

Usage:
    model = RealSensorModel(seed=42)
    noisy_frame = model.camera(raw_rgb_uint8, angular_velocity_rad_s)
    gyro, accel  = model.imu(true_angular_vel_rad_s, true_accel_m_s2, dt)
    ticks        = model.encoder(true_speed_m_s, wheel_radius_m, dt)
    contact      = model.whisker(raw_contact_bool, now_s)
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Camera noise  (OV2640 / Pi Camera Module 2 approximations)
# ---------------------------------------------------------------------------

@dataclass
class CameraParams:
    """Camera intrinsics calibrated to PyBullet 66 deg vertical FOV on 96x128."""
    # Intrinsics matching pb.computeProjectionMatrixFOV(66, 128/96, ...):
    # fy = H / (2 * tan(66°/2)) ≈ 73.914 px; square pixels -> fx = fy ≈ 73.914 px (H-FOV ≈ 81.8°)
    fx: float = 96.0 / (2.0 * math.tan(math.radians(66.0) / 2.0))
    fy: float = 96.0 / (2.0 * math.tan(math.radians(66.0) / 2.0))
    cx: float = 63.5   # principal point x [px]
    cy: float = 47.5   # principal point y [px]

    # Radial distortion coefficients (barrel-ish, typical cheap lens)
    # Source: empirical characterisation of Pi Camera at similar FOV
    k1: float = -0.30
    k2: float = 0.10

    # Noise
    read_noise_dn: float = 2.0        # Gaussian read noise [digital numbers]
    shot_noise_factor: float = 0.015  # photon shot noise ∝ √pixel_value
    max_dn: int = 255

    # Rolling shutter: pixels per rad/s of rotation
    shutter_px_per_rad_s: float = 2.0  # conservative estimate


class RealSensorModel:
    """
    Applies realistic sensor noise to raw physics observations.

    Thread-safety: not thread-safe; use one instance per environment.
    """

    def __init__(self, seed: int = 42, cam: Optional[CameraParams] = None,
                 imu_gyro_noise: float = 0.005,    # rad/s  (MPU-6050 spec)
                 imu_gyro_bias_drift: float = 3e-4, # rad/s² (MPU-6050 bias instability)
                 imu_accel_noise: float = 0.012,    # m/s²  (MPU-6050 spec)
                 encoder_cpr: int = 360,            # counts per revolution
                 encoder_slip_prob: float = 0.05,   # probability of losing a tick
                 whisker_debounce_s: float = 0.030, # 30 ms mechanical debounce
                 whisker_false_trigger_prob: float = 0.05):
        self._rng = np.random.default_rng(seed)
        self.cam  = cam or CameraParams()

        # IMU state
        self._gyro_bias   = 0.0
        self._gyro_y_bias = 0.0
        self._gyro_noise  = imu_gyro_noise
        self._gyro_drift  = imu_gyro_bias_drift
        self._accel_noise = imu_accel_noise

        # Encoder state
        self._enc_cpr        = encoder_cpr
        self._enc_slip_prob  = encoder_slip_prob
        self._enc_accum_left  = 0.0
        self._enc_accum_right = 0.0

        # Whisker state
        self._whisker_debounce_s = whisker_debounce_s
        self._whisker_false_prob = whisker_false_trigger_prob
        self._whisker_last_true_L: Optional[float] = None
        self._whisker_last_true_R: Optional[float] = None

        # Distortion lookup table (precomputed for 96×128)
        self._distortion_map = self._precompute_distortion(96, 128)

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _precompute_distortion(self, H: int, W: int) -> np.ndarray:
        """
        Build a pixel-remapping array for barrel/pincushion distortion.
        Returns src_coords[dst_y, dst_x] = (src_x, src_y) in float.

        Uses the standard Brown-Conrady model:
            r² = x_n² + y_n²
            x_d = x_n (1 + k1 r² + k2 r⁴)
        where x_n, y_n are normalised (divided by focal length) coordinates.
        """
        c = self.cam
        ys, xs = np.mgrid[0:H, 0:W]
        # Normalised coordinates
        xn = (xs - c.cx) / c.fx
        yn = (ys - c.cy) / c.fy
        r2 = xn ** 2 + yn ** 2
        factor = 1 + c.k1 * r2 + c.k2 * r2 ** 2
        # Distorted normalised coords → pixel
        src_x = c.fx * xn * factor + c.cx
        src_y = c.fy * yn * factor + c.cy
        # Clamp to valid range
        src_x = np.clip(src_x, 0, W - 1).astype(np.float32)
        src_y = np.clip(src_y, 0, H - 1).astype(np.float32)
        return np.stack([src_y, src_x], axis=-1)   # [H, W, 2]

    @staticmethod
    def _bilinear(img: np.ndarray, coords: np.ndarray) -> np.ndarray:
        """Pure-NumPy bilinear sampling. img: [H,W,C], coords: [H,W,2] float."""
        H, W, C = img.shape
        y = coords[..., 0]; x = coords[..., 1]
        y0 = np.floor(y).astype(int); y1 = np.minimum(y0 + 1, H - 1)
        x0 = np.floor(x).astype(int); x1 = np.minimum(x0 + 1, W - 1)
        wy = (y - y0)[..., None]; wx = (x - x0)[..., None]
        return (img[y0, x0] * (1 - wy) * (1 - wx) +
                img[y1, x0] * wy       * (1 - wx) +
                img[y0, x1] * (1 - wy) * wx       +
                img[y1, x1] * wy       * wx       ).astype(np.uint8)

    def camera(self, raw: np.ndarray, angular_velocity_rad_s: float = 0.0) -> np.ndarray:
        """
        Apply realistic camera degradation to a raw RGB frame.

        1. Radial distortion (barrel)
        2. Rolling-shutter smear horizontal streak
        3. Read noise + shot noise
        4. Exposure clamp (overexposure, underexposure)

        Args:
            raw:                   uint8 [96, 128, 3] RGB frame from physics.
            angular_velocity_rad_s: yaw rate for rolling-shutter calculation.

        Returns:
            uint8 [96, 128, 3] degraded frame.
        """
        frame = raw.astype(np.float32)

        # 1. Radial distortion
        frame = self._bilinear(raw, self._distortion_map).astype(np.float32)

        # 2. Rolling-shutter smear: shift each row by a fraction of a pixel
        shutter_shift = self.cam.shutter_px_per_rad_s * angular_velocity_rad_s
        if abs(shutter_shift) > 0.1:
            H, W, _ = frame.shape
            for row in range(H):
                shift = int(round(shutter_shift * row / H))
                if shift != 0:
                    frame[row] = np.roll(frame[row], shift, axis=0)

        # 3. Read noise + shot noise
        read_noise = self._rng.normal(0, self.cam.read_noise_dn, frame.shape)
        shot_noise = self._rng.normal(0, self.cam.shot_noise_factor * np.sqrt(np.maximum(frame, 1)), frame.shape)
        frame = frame + read_noise + shot_noise

        # 4. Exposure clamp
        frame = np.clip(frame, 0, self.cam.max_dn)
        return frame.astype(np.uint8)

    @staticmethod
    def apply_camera_noise(raw: np.ndarray, angular_velocity_rad_s: float = 0.0,
                            seed: int = 0) -> np.ndarray:
        """
        Stateless convenience wrapper for use from RealWorldEnv.render().
        Creates a minimal RealSensorModel instance with the given seed.
        Note: does NOT maintain IMU/encoder state — use the full instance for that.
        """
        model = RealSensorModel(seed=seed)
        return model.camera(raw, angular_velocity_rad_s)

    # ------------------------------------------------------------------
    # IMU  (MPU-6050 model)
    # ------------------------------------------------------------------

    def imu(self, true_gyro_z_rad_s: float, true_accel_x_m_s2: float,
            dt: float, true_gyro_y_rad_s: float = 0.0) -> dict:
        """
        Returns a noisy IMU reading.

        Gyro model: white noise + random-walk bias drift (yaw z and pitch y)
        Accel model: white noise only (bias assumed calibrated)

        Args:
            true_gyro_z_rad_s:  True yaw rate [rad/s].
            true_accel_x_m_s2:  True forward acceleration [m/s²].
            dt:                 Timestep [s].
            true_gyro_y_rad_s:  True body pitch rate [rad/s].

        Returns:
            dict with keys: gyro_z_rad_s, gyro_y_rad_s, accel_x_m_s2
        """
        dt_pos = max(float(dt), 1e-6)
        # Bias random walk
        self._gyro_bias += float(self._rng.normal(0, self._gyro_drift * math.sqrt(dt_pos)))
        self._gyro_y_bias += float(self._rng.normal(0, self._gyro_drift * math.sqrt(dt_pos)))
        # Measurement
        gyro_z_meas = true_gyro_z_rad_s + self._gyro_bias + float(self._rng.normal(0, self._gyro_noise))
        gyro_y_meas = true_gyro_y_rad_s + self._gyro_y_bias + float(self._rng.normal(0, self._gyro_noise))
        accel_meas = true_accel_x_m_s2 + float(self._rng.normal(0, self._accel_noise))
        return {"gyro_z_rad_s": gyro_z_meas, "gyro_y_rad_s": gyro_y_meas, "accel_x_m_s2": accel_meas}

    # ------------------------------------------------------------------
    # Encoder  (quadrature, 360 CPR)
    # ------------------------------------------------------------------

    def encoder(self, speed_m_s: float, wheel_radius_m: float, dt: float,
                side: str = "left") -> dict:
        """
        Convert true wheel speed to encoder tick count with quantisation + slip.

        Args:
            speed_m_s:      True wheel surface speed [m/s].
            wheel_radius_m: Physical wheel radius [m].
            dt:             Timestep [s].
            side:           'left' or 'right' (maintains separate accumulator).

        Returns:
            dict: ticks (int), delta_rad (float, noisy), slip_occurred (bool)
        """
        omega = speed_m_s / max(wheel_radius_m, 1e-6)   # [rad/s]
        delta_rad_true = omega * dt

        # Quantise to nearest tick
        rad_per_tick = 2 * math.pi / self._enc_cpr
        if side == "left":
            self._enc_accum_left += delta_rad_true
            accum = self._enc_accum_left
        else:
            self._enc_accum_right += delta_rad_true
            accum = self._enc_accum_right

        ticks = int(accum / rad_per_tick)
        remainder = accum - ticks * rad_per_tick
        if side == "left":
            self._enc_accum_left = remainder
        else:
            self._enc_accum_right = remainder

        # Slip: randomly drop one tick on high dynamics
        slip = False
        if ticks != 0 and float(self._rng.random()) < self._enc_slip_prob:
            ticks -= int(math.copysign(1, ticks))
            slip = True

        delta_rad_meas = ticks * rad_per_tick
        return {"ticks": ticks, "delta_rad": delta_rad_meas, "slip_occurred": slip}

    # ------------------------------------------------------------------
    # Whisker / contact switch
    # ------------------------------------------------------------------

    def whisker(self, raw_left: bool, raw_right: bool, now_s: float) -> dict:
        """
        Apply mechanical debounce and false-trigger noise to raw contact signals.

        Args:
            raw_left, raw_right: True physics contact booleans.
            now_s:               Current simulation time [s].

        Returns:
            dict: left (bool), right (bool), false_trigger_occurred (bool)
        """
        false_trigger = False

        def debounce(raw, last_true_time):
            if raw:
                return True, now_s
            # Hysteresis: remain True for debounce window after last true contact
            if last_true_time is not None and (now_s - last_true_time) < self._whisker_debounce_s:
                return True, last_true_time
            return False, last_true_time

        left_out, self._whisker_last_true_L = debounce(raw_left, self._whisker_last_true_L)
        right_out, self._whisker_last_true_R = debounce(raw_right, self._whisker_last_true_R)

        # False trigger noise
        if not left_out and float(self._rng.random()) < self._whisker_false_prob / 100:
            left_out = True; false_trigger = True
        if not right_out and float(self._rng.random()) < self._whisker_false_prob / 100:
            right_out = True; false_trigger = True

        return {"left": left_out, "right": right_out, "false_trigger": false_trigger}
