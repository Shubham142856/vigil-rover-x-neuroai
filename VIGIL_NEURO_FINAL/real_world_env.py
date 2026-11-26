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
    """Create a heightfield-based terrain matching scene_cfg."""
    rows, cols = 64, 64
    cell_m = 8.0 / max(rows, cols)   # 8×8 m arena

    # Base bumps: ±3 mm RMS for tile, ±8 mm for outdoor, 0 for ramp
    bump_std = scene_cfg.get("bump_std_m", 0.003)
    heights = rng.normal(0, bump_std, (rows, cols)).astype(np.float32)

    # Ramp: add linear gradient
    if scene_cfg.get("ramp", False):
        angle_deg = scene_cfg.get("ramp_angle_deg", 5.0)
        slope = math.tan(math.radians(angle_deg))
        for r in range(rows):
            heights[r, :] += slope * (r - rows // 2) * cell_m

    # Flatten the starting 1 m zone so rover spawns on flat ground
    flat_cells = max(1, int(1.0 / cell_m))
    heights[:, :flat_cells] = 0.0

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
        0, terrain_shape, -1, [0, 0, 0],
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
    """Rectangular cart bouncing along the x-axis at constant speed."""

    def __init__(self, client, x0, y0, speed_m_s=0.5):
        self.client = client
        self.speed = speed_m_s
        self.dir = 1.0
        shape = pb.createCollisionShape(
            pb.GEOM_BOX, halfExtents=[0.3, 0.2, 0.4], physicsClientId=client)
        visual = pb.createVisualShape(
            pb.GEOM_BOX, halfExtents=[0.3, 0.2, 0.4],
            rgbaColor=[0.9, 0.7, 0.1, 1], physicsClientId=client)
        self.body = pb.createMultiBody(
            5.0, shape, visual, [x0, y0, 0.4], physicsClientId=client)
        pb.changeDynamics(self.body, -1, lateralFriction=0.4,
                          physicsClientId=client)
        # Kinematic body
        pb.changeDynamics(self.body, -1, mass=0, physicsClientId=client)
        self._x, self._y = x0, y0
        self._x_min, self._x_max = x0 - 1.5, x0 + 1.5

    def update(self, dt):
        self._x += self.dir * self.speed * dt
        if self._x >= self._x_max:
            self._x = self._x_max; self.dir = -1.0
        elif self._x <= self._x_min:
            self._x = self._x_min; self.dir = 1.0
        pb.resetBasePositionAndOrientation(
            self.body, [self._x, self._y, 0.4], [0, 0, 0, 1],
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
            pb.GEOM_CAPSULE, radius=0.2, height=1.4, physicsClientId=client)
        visual = pb.createVisualShape(
            pb.GEOM_CAPSULE, radius=0.2, length=1.4,
            rgbaColor=[0.2, 0.5, 0.9, 1], physicsClientId=client)
        self.body = pb.createMultiBody(
            0, shape, visual, [cx, cy, 0.9], physicsClientId=client)

    def update(self, dt):
        self.t += dt
        # Lemniscate of Bernoulli (figure-eight)
        denom = 1 + math.sin(self.speed * self.t) ** 2
        x = self.cx + self.r * math.cos(self.speed * self.t) / denom
        y = self.cy + self.r * math.sin(self.speed * self.t) * math.cos(self.speed * self.t) / denom
        pb.resetBasePositionAndOrientation(
            self.body, [x, y, 0.9], [0, 0, 0, 1],
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
        parts = {k: float(v) for k, v in (p.split("=") for p in args.split(","))}
        x0 = parts.get("x0", 1.0); y0 = parts.get("y0", 0.0)
        speed = parts.get("speed", 0.5)
        return _MovingCart(client, x0, y0, speed)
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

    def __init__(self):
        self._speed = 0.0

    def update(self, target_speed, dt):
        # First-order lag
        self._speed += (target_speed - self._speed) * (dt / self.TAU)
        # Back-EMF: cannot exceed max
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
                 delay_s=0.03, sensor_noise=True):
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
        self._turn  = 0.0
        self._motor = _MotorModel()
        self._touch = False
        self._events = 0
        self._packet = None
        self._queue  = []
        self._last_center = 0.0
        self._growth = 0.0
        self._rgb0   = np.zeros((96, 128, 3), dtype=np.uint8)

        # Lighting state
        cfg = SCENES.get(scene, SCENES["open"])
        self._base_lux   = cfg.get("lighting_lux", 400)
        self._flicker_t  = 0.0
        self._tint       = self._rng.uniform(0.85, 1.0)

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

        # Rover body
        self.rover = self._box([0, 0, 0.05], [0.10, 0.08, 0.05], 1.0,
                                [0.1, 0.3, 0.5, 1])
        pb.changeDynamics(self.rover, -1,
                          lateralFriction=cfg.get("friction_mu", 0.6),
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
        """Compute per-frame tint [0.7, 1.0] with 2 Hz flicker noise."""
        lux_norm = min(1.0, self._base_lux / 10000.0)   # outdoor = 1.0
        flicker = 0.1 * math.sin(2 * math.pi * 2.0 * self._flicker_t)
        return float(np.clip(self._tint * lux_norm + flicker * 0.05, 0.3, 1.0))

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

        # Apply sensor noise if enabled (delegated to RealSensorModel)
        if self._sensor_noise:
            try:
                from real_sensor_model import RealSensorModel
                frame = RealSensorModel.apply_camera_noise(frame, self._angular_velocity())
            except ImportError:
                pass   # sensor model optional — degrade gracefully
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
            delivery = now + (self._delay_s if fault == "delay" else self._delay_s)
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
            # Reality-grounded extras (available to policy but not required)
            scene_lux=self._base_lux,
            terrain_bump_rms_m=SCENES.get("open", {}).get("bump_std_m", 0.003),
        )

    def step(self, action):
        targets = {"forward": (0.35, 0), "slow": (0.12, 0),
                   "brake": (0, 0), "left": (0, 1.0), "right": (0, -1.0)}
        target_speed, turn = targets[action]

        # Safety override: tactile contact → brake
        if any(self.whiskers()):
            target_speed, turn = 0.0, 0.0

        dt_ctrl = 1.0 / self.CTRL_HZ
        actual_speed = self._motor.update(target_speed, dt_ctrl)

        for _ in range(self.STEPS_PER_CTRL):
            _, yaw = self.pose()
            vx = actual_speed * math.cos(yaw)
            vy = actual_speed * math.sin(yaw)
            pb.resetBaseVelocity(self.rover, [vx, vy, 0], [0, 0, turn],
                                 physicsClientId=self.client)
            pb.stepSimulation(physicsClientId=self.client)
            touching = any(pb.getContactPoints(self.rover, o, physicsClientId=self.client)
                           for o in self.solids)
            self._events += int(touching and not self._touch)
            self._touch = touching

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
