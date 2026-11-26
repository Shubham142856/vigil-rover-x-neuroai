"""
real_world_env.py — Reality-grounded PyBullet environment for VIGIL-ROVER X.

Extends the benchmark RoverSim with:
  - Procedurally-textured terrain (height-map bumps, friction zones)
  - Multi-zone lighting with flicker noise
  - Dynamic obstacles: moving cart, swinging door, pedestrian proxy
  - Actuator model: PWM→torque ramp, stall, back-EMF
  - Scene registry: open, corridor, warehouse_bay, outdoor_path, lab_floor, ramp

The neural/AI stack (camera_escape, compass_benchmark, recovery_policy) is
unchanged — this module is a drop-in replacement for RoverSim that uses the
identical .observe() / .step() / .evaluator() interface.

Real-world grounding choices (all from hardware references):
  Rover:   ~0.20×0.16 m chassis, 0.14 m wheelbase, max 0.35 m/s
  Camera:  96×128 px, 66° H-FOV, 10 Hz  (matches OV2640 / Pi camera)
  Physics: 120 Hz fixed timestep (PyBullet default)
  Terrain: height-map bumps ±3 mm  (tile floor tolerance)
  Friction: tile μ=0.6, gravel μ=0.4, rubber μ=0.9  (ASTM D2047)
  Lighting: 300–500 lux ambient + 2 point lights ±10% @ 2 Hz flicker
  Motor:   τ_motor=0.08 s, stall_torque=0.35 N·m  (typical 25 mm gearmotor)
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import pybullet as pb

# ---------------------------------------------------------------------------
# Terrain generator
# ---------------------------------------------------------------------------

def _build_terrain(client, rng, scene_cfg):
    """Create a smoothed heightfield-based terrain matching scene_cfg."""
    rows, cols = 64, 64
    cell_m = 8.0 / max(rows, cols)   # 8×8 m arena

    # Smooth micro-undulations: ±3 mm RMS for tile, ±8 mm for outdoor
    bump_std = scene_cfg.get("bump_std_m", 0.003)
    heights = rng.normal(0, bump_std, (rows, cols)).astype(np.float32)
    for _ in range(2):
        padded = np.pad(heights, 1, mode="edge")
        heights = (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
            + padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:]
            + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
        ) / 9.0

    # Flatten the spawn patch around arena center (0, 0) -> indices [rows//2, cols//2]
    cr, cc = rows // 2, cols // 2
    heights -= heights[cr, cc]
    heights[max(0, cr - 2):min(rows, cr + 3), max(0, cc - 2):min(cols, cc + 3)] = 0.0

    # Ramp: add linear gradient along +x (col axis)
    if scene_cfg.get("ramp", False):
        angle_deg = scene_cfg.get("ramp_angle_deg", 5.0)
        slope = -math.tan(math.radians(angle_deg)) if scene_cfg.get("descent", False) else math.tan(math.radians(angle_deg))
        for c in range(cols):
            heights[:, c] += slope * (c - cc) * cell_m

    terrain_shape = pb.createCollisionShape(
        pb.GEOM_HEIGHTFIELD,
        meshScale=[cell_m, cell_m, 1.0],
        heightfieldTextureScaling=rows // 2,
        heightfieldData=heights.flatten().tolist(),
        numHeightfieldRows=rows,
        numHeightfieldColumns=cols,
        physicsClientId=client,
    )
    terrain_body = pb.createMultiBody(
        0, terrain_shape, -1, [0, 0, -0.002],
        physicsClientId=client,
    )
    # Friction per scene
    mu = scene_cfg.get("friction_mu", 0.6)
    pb.changeDynamics(terrain_body, -1, lateralFriction=mu, physicsClientId=client)
    return terrain_body


# ---------------------------------------------------------------------------
# Dynamic obstacle controllers
# ---------------------------------------------------------------------------

class _MovingCart:
    """Rectangular cart/forklift patrolling along the x or y axis at constant speed."""

    def __init__(self, client, x0, y0, speed_m_s=0.5, axis="x", span=1.5):
        self.client = client
        self.speed = speed_m_s
        self.axis = axis
        self.dir = 1.0
        half = [0.16, 0.14, 0.28] if axis == "y" else [0.3, 0.2, 0.4]
        z0 = half[2]
        shape = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=half, physicsClientId=client)
        visual = pb.createVisualShape(
            pb.GEOM_BOX, halfExtents=half,
            rgbaColor=[0.88, 0.22, 0.14, 1], physicsClientId=client)
        self.body = pb.createMultiBody(0, shape, visual, [x0, y0, z0], physicsClientId=client)
        pb.changeDynamics(self.body, -1, lateralFriction=0.4, physicsClientId=client)
        self._x, self._y, self._z = x0, y0, z0
        self._min = (y0 - span if axis == "y" else x0 - span)
        self._max = (y0 + span if axis == "y" else x0 + span)

    def update(self, dt):
        coord = self._y if self.axis == "y" else self._x
        coord += self.dir * self.speed * dt
        if coord >= self._max:
            coord = self._max; self.dir = -1.0
        elif coord <= self._min:
            coord = self._min; self.dir = 1.0
        if self.axis == "y":
            self._y = coord
        else:
            self._x = coord
        pb.resetBasePositionAndOrientation(
            self.body, [self._x, self._y, self._z], [0, 0, 0, 1],
            physicsClientId=self.client)

    def get_state(self):
        return (self._x, self._y, self.dir)

    def set_state(self, st):
        self._x, self._y, self.dir = st
        pb.resetBasePositionAndOrientation(
            self.body, [self._x, self._y, self._z], [0, 0, 0, 1],
            physicsClientId=self.client)

    @property
    def body_id(self):
        return self.body


class _SwingingDoor:
    """Door that opens and closes on a fixed hinge (kinematic)."""

    def __init__(self, client, x0, y0, period_s=4.0):
        self.client = client
        self.period = period_s
        self.t = 0.0
        shape = pb.createCollisionShape(
            pb.GEOM_BOX, halfExtents=[0.02, 0.4, 0.9], physicsClientId=client)
        visual = pb.createVisualShape(
            pb.GEOM_BOX, halfExtents=[0.02, 0.4, 0.9],
            rgbaColor=[0.6, 0.6, 0.9, 0.8], physicsClientId=client)
        self.body = pb.createMultiBody(
            0, shape, visual, [x0, y0, 0.9], physicsClientId=client)
        self._x0, self._y0 = x0, y0

    def update(self, dt):
        self.t += dt
        angle = math.pi / 3 * math.sin(2 * math.pi * self.t / self.period)
        c, s = math.cos(angle), math.sin(angle)
        hinge_x = self._x0
        hinge_y = self._y0 - 0.4
        pivot_x = hinge_x + 0.4 * s
        pivot_y = hinge_y + 0.4 * c
        q = pb.getQuaternionFromEuler([0, 0, angle])
        pb.resetBasePositionAndOrientation(
            self.body, [pivot_x, pivot_y, 0.9], q,
            physicsClientId=self.client)

    @property
    def body_id(self):
        return self.body


class _PedestrianProxy:
    """Capsule-shaped pedestrian walking a figure-eight path."""

    def __init__(self, client, cx, cy, radius=0.5, speed=0.8):
        self.client = client
        self.cx, self.cy = cx, cy
        self.r, self.speed = radius, speed
        self.t = 0.0
        shape = pb.createCollisionShape(
            pb.GEOM_CAPSULE, radius=0.14, height=1.2, physicsClientId=client)
        visual = pb.createVisualShape(
            pb.GEOM_CAPSULE, radius=0.14, length=1.2,
            rgbaColor=[0.88, 0.22, 0.14, 1], physicsClientId=client)  # high-vis vest
        self.body = pb.createMultiBody(
            0, shape, visual, [cx, cy, 0.7], physicsClientId=client)

    def update(self, dt):
        self.t += dt
        # Lemniscate of Bernoulli (figure-eight)
        denom = 1 + math.sin(self.speed * self.t) ** 2
        x = self.cx + self.r * math.cos(self.speed * self.t) / denom
        y = self.cy + self.r * math.sin(self.speed * self.t) * math.cos(self.speed * self.t) / denom
        pb.resetBasePositionAndOrientation(
            self.body, [x, y, 0.7], [0, 0, 0, 1],
            physicsClientId=self.client)

    @property
    def body_id(self):
        return self.body


# ---------------------------------------------------------------------------
# Scene registry
# ---------------------------------------------------------------------------

SCENES = {
    # (scene_cfg, static_obstacle_specs, dynamic_obstacle_factories)
    # static_obstacle_specs: list of (pos, half, color)
    # dynamic_obstacle_factories: list of callables(client, rng) -> dynamic obj

    "open": {
        "bump_std_m": 0.003, "friction_mu": 0.6,
        "statics": [],
        "dynamics": [],
        "lighting_lux": 400,
    },
    "corridor": {
        "bump_std_m": 0.002, "friction_mu": 0.65,
        "statics": [
            ([0, 1.1, 0.5],  [4.0, 0.05, 0.5], [0.7, 0.7, 0.7, 1]),  # left wall
            ([0, -1.1, 0.5], [4.0, 0.05, 0.5], [0.7, 0.7, 0.7, 1]),  # right wall
        ],
        "dynamics": ["door@(1.5,0.9)", "pedestrian@(2.0,0.0,r=0.4)"],
        "lighting_lux": 300,
    },
    "warehouse_bay": {
        "bump_std_m": 0.005, "friction_mu": 0.55,
        "statics": [
            ([1.5, 0.6, 0.5],  [0.05, 0.6, 0.5], [0.5, 0.5, 0.5, 1]),  # rack left
            ([1.5, -0.6, 0.5], [0.05, 0.6, 0.5], [0.5, 0.5, 0.5, 1]),  # rack right
        ],
        "dynamics": ["cart@(1.0,0.0,speed=0.5)", "pedestrian@(2.0,0.5,r=0.6)"],
        "lighting_lux": 250,
    },
    "outdoor_path": {
        "bump_std_m": 0.008, "friction_mu": 0.45,
        "statics": [
            ([0.8, 0, 0.1], [0.05, 0.05, 0.1], [0.4, 0.4, 0.4, 1]),  # pothole edge
        ],
        "dynamics": ["pedestrian@(1.5,0.3,r=0.5,speed=0.7)"],
        "lighting_lux": 10000,  # outdoor
    },
    "lab_floor": {
        "bump_std_m": 0.001, "friction_mu": 0.9,
        "statics": [
            ([0.9, 0, 0.3], [0.01, 0.4, 0.3], [0.85, 0.95, 1.0, 0.3]),  # glass wall
        ],
        "dynamics": [],
        "lighting_lux": 500,
    },
    "ramp": {
        "bump_std_m": 0.002, "friction_mu": 0.7,
        "ramp": True, "ramp_angle_deg": 5.0,
        "statics": [],
        "dynamics": [],
        "lighting_lux": 400,
    },
    # R05 real-world scenarios (explicit transfer benchmark cases)
    "corridor_pedestrian": {
        "bump_std_m": 0.002, "friction_mu": 0.65,
        "statics": [
            ([0.8, 0.42, 0.3],  [1.5, 0.03, 0.3], [0.65, 0.68, 0.72, 1]),
            ([0.8, -0.42, 0.3], [1.5, 0.03, 0.3], [0.65, 0.68, 0.72, 1]),
        ],
        "dynamics": ["pedestrian@cx=1.05,cy=0.0,r=0.32,speed=0.9"],
        "lighting_lux": 300,
    },
    "warehouse_forklift_cross": {
        "bump_std_m": 0.004, "friction_mu": 0.55,
        "statics": [
            ([1.4, 0.65, 0.4],  [0.05, 0.35, 0.4], [0.45, 0.48, 0.52, 1]),
            ([1.4, -0.65, 0.4], [0.05, 0.35, 0.4], [0.45, 0.48, 0.52, 1]),
        ],
        "dynamics": ["cart@x0=0.90,y0=-0.42,speed=0.42,axis=y,span=0.50"],
        "lighting_lux": 250,
    },
    "outdoor_pothole": {
        "bump_std_m": 0.008, "friction_mu": 0.45,
        "statics": [
            ([0.68, 0.04, 0.09], [0.04, 0.05, 0.09], [0.78, 0.22, 0.14, 1]),
        ],
        "dynamics": [],
        "lighting_lux": 10000,
    },
    "lab_glass_wall": {
        "bump_std_m": 0.001, "friction_mu": 0.90,
        "statics": [
            ([0.65, 0.0, 0.25], [0.015, 0.35, 0.25], [0.65, 0.90, 0.95, 0.30]),
        ],
        "dynamics": [],
        "lighting_lux": 500,
    },
    "ramp_descent": {
        "bump_std_m": 0.002, "friction_mu": 0.70,
        "ramp": True, "ramp_angle_deg": 5.0, "descent": True,
        "statics": [
            ([0.85, 0.0, 0.10], [0.03, 0.04, 0.12], [0.86, 0.17, 0.09, 1]),
        ],
        "dynamics": [],
        "lighting_lux": 400,
    },
    "night_mode": {
        "bump_std_m": 0.003, "friction_mu": 0.60,
        "statics": [
            ([0.65, 0.0, 0.12], [0.03, 0.025, 0.12], [0.86, 0.17, 0.09, 1]),
        ],
        "dynamics": [],
        "lighting_lux": 15,
    },
    # Legacy scenes kept for backward compat with closed_loop_benchmark
    "wall":  {"bump_std_m": 0.003, "friction_mu": 0.6,
               "statics": [([0.55, 0, 0.12], [0.03, 0.22, 0.12], [0.86, 0.17, 0.09, 1])],
               "dynamics": [], "lighting_lux": 400},
    "left":  {"bump_std_m": 0.003, "friction_mu": 0.6,
               "statics": [([0.55, 0.07, 0.12], [0.03, 0.028, 0.12], [0.86, 0.17, 0.09, 1])],
               "dynamics": [], "lighting_lux": 400},
    "right": {"bump_std_m": 0.003, "friction_mu": 0.6,
               "statics": [([0.55, -0.07, 0.12], [0.03, 0.028, 0.12], [0.86, 0.17, 0.09, 1])],
               "dynamics": [], "lighting_lux": 400},
    "post":  {"bump_std_m": 0.003, "friction_mu": 0.6,
               "statics": [([0.55, 0, 0.12], [0.03, 0.02, 0.12], [0.86, 0.17, 0.09, 1])],
               "dynamics": [], "lighting_lux": 400},
}


def _parse_dynamic(spec, client, rng):
    """Parse a dynamic obstacle spec string and return an object."""
    if spec.startswith("cart@"):
        args = spec[5:].strip("()")
        parts = dict(p.split("=") for p in args.split(","))
        x0 = float(parts.get("x0", 1.0))
        y0 = float(parts.get("y0", 0.0))
        speed = float(parts.get("speed", 0.5))
        axis = parts.get("axis", "x")
        span = float(parts.get("span", 1.5))
        return _MovingCart(client, x0, y0, speed, axis=axis, span=span)
    elif spec.startswith("door@"):
        args = spec[5:].strip("()")
        nums = [float(v) for v in args.split(",")]
        return _SwingingDoor(client, nums[0], nums[1])
    elif spec.startswith("pedestrian@"):
        args = spec[11:].strip("()")
        parts = {}
        for p in args.split(","):
            if "=" in p:
                k, v = p.split("="); parts[k] = float(v)
            else:
                vals = [float(v) for v in args.split(",") if "=" not in v]
                if len(vals) >= 2:
                    parts.setdefault("cx", vals[0]); parts.setdefault("cy", vals[1])
        cx = parts.get("cx", 2.0); cy = parts.get("cy", 0.0)
        r  = parts.get("r", 0.5); speed = parts.get("speed", 0.8)
        return _PedestrianProxy(client, cx, cy, r, speed)
    else:
        raise ValueError(f"Unknown dynamic spec: {spec!r}")


# ---------------------------------------------------------------------------
# Motor model
# ---------------------------------------------------------------------------

class _MotorModel:
    """First-order lag + stall + back-EMF speed limiting."""
    TAU = 0.08          # time constant [s]
    STALL_TORQUE = 0.35 # [N·m]
    MAX_SPEED = 0.35    # [m/s]
    MASS = 1.0          # rover mass [kg] for torque→force

    def __init__(self, friction_mu: float = 0.6, descent_deg: float = 0.0):
        self._speed = 0.0
        self._friction_mu = max(0.3, float(friction_mu))
        self._descent_sin = math.sin(math.radians(descent_deg))

    def update(self, target_speed, dt):
        # Traction limits response rate on low-friction surfaces (gravel/wet floor)
        grip_scale = min(1.0, self._friction_mu / 0.6)
        alpha = min(1.0, (dt / self.TAU) * grip_scale)
        self._speed += (target_speed - self._speed) * alpha
        # Downhill gravity bias (~g*sin(slope)*dt) on ramp_descent when coasting
        if self._descent_sin > 0.0 and target_speed > 0.0:
            self._speed += 9.81 * self._descent_sin * dt * 0.15
        self._speed = float(np.clip(self._speed, -self.MAX_SPEED, self.MAX_SPEED))
        return self._speed

    @property
    def speed(self):
        return self._speed


# ---------------------------------------------------------------------------
# Main RealWorldEnv class
# ---------------------------------------------------------------------------

class RealWorldEnv:
    """
    Reality-grounded rover environment.

    Drop-in replacement for RoverSim with identical observe()/step()/evaluator()
    interface, extended with real-world physics and sensor fidelity.
    """

    CTRL_HZ = 20          # controller rate
    PHYS_HZ = 120         # physics rate
    CAM_HZ  = 10          # camera rate
    STEPS_PER_CTRL = PHYS_HZ // CTRL_HZ   # 6
    CTRL_PER_CAM   = CTRL_HZ // CAM_HZ    # 2

    def __init__(self, scene="open", seed=7,
                 fault="none", fault_start_s=0.2, fault_end_s=0.5,
                 delay_s=0.3, sensor_noise=True):
        """
        Args:
            scene:        One of SCENES keys.
            seed:         RNG seed for terrain and color jitter.
            fault:        Camera fault: none | drop | freeze | delay.
            fault_start_s / fault_end_s: Fault window.
            delay_s:      Camera pipeline latency [s] (default 30 ms real-world).
            sensor_noise: Apply RealSensorModel noise if True.
        """
        self.client = pb.connect(pb.DIRECT)
        pb.setGravity(0, 0, -9.81, physicsClientId=self.client)
        pb.setTimeStep(1 / self.PHYS_HZ, physicsClientId=self.client)

        self._tick = 0          # controller ticks
        self._rng  = np.random.default_rng(seed)
        self._fault = fault
        self._fault_window = (fault_start_s, fault_end_s)
        self._delay_s = delay_s
        self._sensor_noise = sensor_noise

        self._speed = 0.0
        self._prev_speed = 0.0
        self._turn  = 0.0
        self._motor = _MotorModel()
        self._touch = False
        self._events = 0
        self._packet = None
        self._queue  = []
        self._last_center = 0.0
        self._growth = 0.0
        self._rgb0   = np.zeros((96, 128, 3), dtype=np.uint8)
        self._last_imu = {"gyro_z_rad_s": 0.0, "gyro_y_rad_s": 0.0, "accel_x_m_s2": 0.0}
        self._last_enc_speed = 0.0

        # Lighting & motor state
        cfg = SCENES.get(scene, SCENES["open"])
        self._scene_cfg  = cfg
        self._base_lux   = cfg.get("lighting_lux", 400)
        self._flicker_t  = 0.0
        self._tint       = self._rng.uniform(0.85, 1.0)
        self._motor = _MotorModel(
            friction_mu=cfg.get("friction_mu", 0.6),
            descent_deg=cfg.get("ramp_angle_deg", 0.0) if cfg.get("descent", False) else 0.0,
        )
        self._sensor_model = None
        if self._sensor_noise:
            from real_sensor_model import RealSensorModel
            self._sensor_model = RealSensorModel(seed=seed)

        # Build world
        self._static_bodies = []
        self._dynamic_objs  = []

        # Terrain
        try:
            _build_terrain(self.client, self._rng, cfg)
        except Exception:
            # PyBullet DIRECT may not support heightfield on all platforms;
            # fall back to flat ground
            self._flat_ground()

        # Rover body (low chassis sliding friction; tire grip is modeled in _MotorModel)
        self.rover = self._box([0, 0, 0.05], [0.10, 0.08, 0.05], 1.0,
                                [0.1, 0.3, 0.5, 1])
        pb.changeDynamics(self.rover, -1,
                          lateralFriction=0.05,
                          physicsClientId=self.client)

        # Static obstacles
        for pos, half, color in cfg.get("statics", []):
            b = self._box(pos, half, 0.0, color)
            self._static_bodies.append(b)
        # Legacy: wall/post single static
        if not cfg.get("statics") and scene not in SCENES:
            pass  # bare scene, no extra geometry

        # Dynamic obstacles
        for spec in cfg.get("dynamics", []):
            obj = _parse_dynamic(spec, self.client, self._rng)
            self._dynamic_objs.append(obj)
            self._static_bodies.append(obj.body_id)  # collision detection

        # All solid bodies (for whisker ray & collision)
        self.solids = list(self._static_bodies)

        self._update_camera()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _flat_ground(self):
        shape = pb.createCollisionShape(
            pb.GEOM_BOX, halfExtents=[4, 4, 0.03], physicsClientId=self.client)
        visual = pb.createVisualShape(
            pb.GEOM_BOX, halfExtents=[4, 4, 0.03],
            rgbaColor=[0.55, 0.57, 0.58, 1], physicsClientId=self.client)
        pb.createMultiBody(0, shape, visual, [0, 0, -0.03],
                           physicsClientId=self.client)

    def _box(self, pos, half, mass, color):
        c = self.client
        shape  = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=half, physicsClientId=c)
        visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=half, rgbaColor=color, physicsClientId=c)
        return pb.createMultiBody(mass, shape, visual, pos, physicsClientId=c)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self): return self
    def __exit__(self, *a): pb.disconnect(self.client)
    def close(self): pb.disconnect(self.client)

    # ------------------------------------------------------------------
    # Physics queries
    # ------------------------------------------------------------------

    def pose(self):
        pos, q = pb.getBasePositionAndOrientation(self.rover, physicsClientId=self.client)
        return pos, pb.getEulerFromQuaternion(q)[2]

    def whiskers(self):
        (x, y, _), yaw = self.pose()
        c, s = math.cos(yaw), math.sin(yaw)
        def tip(length, side):
            return [x + length * c - side * s, y + length * s + side * c, 0.06]
        rays = pb.rayTestBatch(
            [tip(0.101, k) for k in (0.07, -0.07)],
            [tip(0.20,  k) for k in (0.07, -0.07)],
            physicsClientId=self.client,
        )
        return tuple(r[0] in self.solids for r in rays)

    # ------------------------------------------------------------------
    # Camera / lighting
    # ------------------------------------------------------------------

    def _lighting_tint(self):
        """Auto-exposed per-frame tint [0.35, 1.0] with 2 Hz flicker noise."""
        lux_norm = min(1.0, (max(self._base_lux, 1.0) / 300.0) ** 0.35)
        flicker = 0.04 * math.sin(2 * math.pi * 2.0 * self._flicker_t)
        return float(np.clip(self._tint * lux_norm + flicker, 0.35, 1.0))

    def render(self):
        (x, y, _), yaw = self.pose()
        c, s = math.cos(yaw), math.sin(yaw)
        eye = [x + 0.09 * c, y + 0.09 * s, 0.14]
        view = pb.computeViewMatrix(eye, [eye[0] + c, eye[1] + s, 0.10], [0, 0, 1])
        proj = pb.computeProjectionMatrixFOV(66, 128 / 96, 0.015, 8)
        rgba = pb.getCameraImage(128, 96, view, proj,
                                 renderer=pb.ER_TINY_RENDERER,
                                 physicsClientId=self.client)[2]
        frame = (np.asarray(rgba).reshape(96, 128, 4)[:, :, :3] *
                 self._lighting_tint()).astype(np.uint8)

        # Apply persistent sensor noise model if enabled
        if self._sensor_model is not None:
            frame = self._sensor_model.camera(frame, self._angular_velocity())
        return frame

    def _angular_velocity(self):
        """Current yaw rate [rad/s] for rolling-shutter smear calculation."""
        try:
            _, ang_vel = pb.getBaseVelocity(self.rover, physicsClientId=self.client)
            return float(ang_vel[2])
        except Exception:
            return 0.0

    def _update_camera(self):
        now = self._tick / self.CTRL_HZ
        self._flicker_t = now
        fault = self._fault if self._fault_window[0] <= now < self._fault_window[1] else "none"
        if self._tick % self.CTRL_PER_CAM == 0 and fault != "drop":
            if fault == "freeze" and self._packet is not None:
                capture, image = self._packet[:2]
            else:
                capture, image = now, self.render()
            delivery = now + (self._delay_s if fault == "delay" else 0.0)
            self._queue.append((delivery, capture, image))

        pending = []
        for delivery, capture, image in self._queue:
            if delivery > now + 1e-9:
                pending.append((delivery, capture, image))
            elif self._packet is None or capture >= self._packet[0]:
                center = image[:, 44:84].astype(float)
                fraction = float(np.mean(center[:, :, 0] > 1.7 * np.maximum(1, center[:, :, 1])))
                self._growth = fraction - self._last_center
                self._last_center = fraction
                self._packet = (capture, image, now)
        self._queue = pending

    # ------------------------------------------------------------------
    # Public interface (identical to RoverSim)
    # ------------------------------------------------------------------

    def observe(self):
        now = self._tick / self.CTRL_HZ
        capture, image, delivery = self._packet or (None, self._rgb0, None)
        left, right = self.whiskers()
        if self._sensor_model is not None:
            w = self._sensor_model.whisker(left, right, now)
            left, right = w["left"], w["right"]
        return dict(
            time_s=now, rgb=image.copy(),
            capture_s=capture, delivery_s=delivery,
            frame_valid=capture is not None,
            frame_age_s=now - capture if capture is not None else None,
            whisker_left=left, whisker_right=right,
            brightness=float(image.mean() / 255),
            contrast=float(image.std() / 255),
            obstacle_center=self._last_center,
            looming_center=self._growth,
            imu_gyro_z=float(self._last_imu["gyro_z_rad_s"]),
            imu_gyro_y=float(self._last_imu["gyro_y_rad_s"]),
            imu_accel_x=float(self._last_imu["accel_x_m_s2"]),
            encoder_speed_m_s=float(self._last_enc_speed),
            # Reality-grounded extras (available to policy but not required)
            scene_lux=self._base_lux,
            terrain_bump_rms_m=self._scene_cfg.get("bump_std_m", 0.003),
            friction_mu=self._scene_cfg.get("friction_mu", 0.6),
        )

    def step(self, action):
        targets = {"forward": (0.35, 0), "slow": (0.12, 0),
                   "brake": (0, 0), "left": (0, 1.0), "right": (0, -1.0)}
        target_speed, turn = targets[action]

        # Safety override: tactile contact → brake
        if any(self.whiskers()):
            target_speed, turn = 0.0, 0.0

        dt_ctrl = 1.0 / self.CTRL_HZ
        prev_speed = self._prev_speed
        actual_speed = self._motor.update(target_speed, dt_ctrl)
        self._prev_speed = actual_speed

        ang_vel_accum = np.zeros(3, dtype=float)
        for _ in range(self.STEPS_PER_CTRL):
            _, yaw = self.pose()
            vx = actual_speed * math.cos(yaw)
            vy = actual_speed * math.sin(yaw)
            pb.resetBaseVelocity(self.rover, [vx, vy, 0], [0, 0, turn],
                                 physicsClientId=self.client)
            pb.stepSimulation(physicsClientId=self.client)
            _, av = pb.getBaseVelocity(self.rover, physicsClientId=self.client)
            ang_vel_accum += np.asarray(av, dtype=float)
            touching = any(pb.getContactPoints(self.rover, o, physicsClientId=self.client)
                           for o in self.solids)
            self._events += int(touching and not self._touch)
            self._touch = touching

        av_mean = ang_vel_accum / max(1, self.STEPS_PER_CTRL)
        _, yaw = self.pose()
        true_gyro_z = float(av_mean[2])
        true_gyro_y = float(-math.sin(yaw) * av_mean[0] + math.cos(yaw) * av_mean[1])
        true_accel_x = (actual_speed - prev_speed) / dt_ctrl
        if self._sensor_model is not None:
            self._last_imu = self._sensor_model.imu(true_gyro_z, true_accel_x, dt_ctrl, true_gyro_y_rad_s=true_gyro_y)
            enc_l = self._sensor_model.encoder(actual_speed, 0.032, dt_ctrl, side="left")
            enc_r = self._sensor_model.encoder(actual_speed, 0.032, dt_ctrl, side="right")
            self._last_enc_speed = 0.5 * (enc_l["delta_rad"] + enc_r["delta_rad"]) * 0.032 / dt_ctrl
        else:
            self._last_imu = {"gyro_z_rad_s": true_gyro_z, "gyro_y_rad_s": true_gyro_y, "accel_x_m_s2": true_accel_x}
            self._last_enc_speed = float(actual_speed)

        # Advance dynamic obstacles
        for obj in self._dynamic_objs:
            obj.update(dt_ctrl)

        self._tick += 1
        self._update_camera()
        return self.observe()

    def evaluator(self):
        (x, y, _), yaw = self.pose()
        velocity, _ = pb.getBaseVelocity(self.rover, physicsClientId=self.client)
        return dict(
            x_m=x, y_m=y, yaw_rad=yaw,
            speed_m_s=math.hypot(*velocity[:2]),
            body_contact=self._touch,
            body_contact_events=self._events,
        )

    def snapshot_state(self):
        """Capture complete Python/scheduler/dynamic state for exact branch rollback."""
        from copy import deepcopy
        dyn = []
        for obj in self._dynamic_objs:
            if hasattr(obj, "get_state"):
                dyn.append(("cart", obj.get_state()))
            elif hasattr(obj, "t"):
                dyn.append(("t", obj.t))
            else:
                dyn.append(("none", None))
        return {
            "scalars": (self._tick, self._speed, self._prev_speed, self._turn, self._events, self._touch,
                        self._last_center, self._growth, self._flicker_t, self._motor._speed, self._last_enc_speed),
            "imu": deepcopy(self._last_imu),
            "queue": deepcopy(self._queue),
            "packet": deepcopy(self._packet),
            "dyn": dyn,
            "sensor": deepcopy(self._sensor_model),
        }

    def restore_state(self, snap):
        """Restore complete Python/scheduler/dynamic state after pb.restoreState."""
        from copy import deepcopy
        (self._tick, self._speed, self._prev_speed, self._turn, self._events, self._touch,
         self._last_center, self._growth, self._flicker_t, self._motor._speed, self._last_enc_speed) = snap["scalars"]
        self._last_imu = deepcopy(snap["imu"])
        self._queue = deepcopy(snap["queue"])
        self._packet = deepcopy(snap["packet"])
        self._sensor_model = deepcopy(snap["sensor"])
        for obj, (kind, val) in zip(self._dynamic_objs, snap["dyn"]):
            if kind == "cart":
                obj.set_state(val)
            elif kind == "t":
                obj.t = val
                obj.update(0.0)
