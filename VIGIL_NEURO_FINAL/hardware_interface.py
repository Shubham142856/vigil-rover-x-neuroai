"""
hardware_interface.py — Abstract hardware interface for VIGIL-ROVER X.

This is the ONLY file that changes when moving from simulator to real hardware.
The neural/AI stack (camera_escape, compass_benchmark, recovery_policy,
NeuroCOVER-L) always calls HardwareInterface methods; it never knows whether
the backend is PyBullet, ROS 2, or a serial motor controller.

Implementations:
    SimInterface    — wraps RealWorldEnv (default, fully working)
    ROSInterface    — ROS 2 topic bridge (STUB — raises NotImplementedError)
    SerialInterface — Arduino/ESP32 serial bridge (STUB — raises NotImplementedError)

Stubs are labelled NOT_IMPLEMENTED and document exactly what parameters
must be provided from hardware before they can be activated.

Real-world transfer checklist (fill in before activating a stub):
  □ Camera intrinsics calibrated: fx, fy, cx, cy, k1, k2 from checkerboard
  □ Motor time constant τ and stall torque measured from datasheet
  □ Encoder CPR confirmed from motor spec
  □ IMU noise floor measured from static bench recording
  □ Friction coefficients measured per terrain type
  □ ROS 2 topic names confirmed: /camera/image_raw, /imu/data, /cmd_vel
  □ Serial protocol spec agreed: baud rate, packet format
"""

from __future__ import annotations

import abc
import json
import time
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class HardwareInterface(abc.ABC):
    """
    Common interface for the rover's hardware (sim or real).

    All methods return the same dict schemas used throughout G02–G10 so the
    neural stack requires zero modification.
    """

    @abc.abstractmethod
    def observe(self) -> dict:
        """
        Return current sensor observation.

        Required keys (identical to RoverSim / RealWorldEnv):
            time_s          float   simulation / wall-clock time [s]
            rgb             ndarray uint8 [96, 128, 3]
            capture_s       float | None  camera capture timestamp
            delivery_s      float | None  camera delivery timestamp
            frame_valid     bool
            frame_age_s     float | None
            whisker_left    bool
            whisker_right   bool
            brightness      float  [0, 1]
            contrast        float  [0, 1]
            obstacle_center float  red-pixel fraction in central column
            looming_center  float  frame-to-frame growth of obstacle_center
        """
        ...

    @abc.abstractmethod
    def step(self, action: str) -> dict:
        """
        Execute one control step.

        Args:
            action: One of "forward" | "slow" | "brake" | "left" | "right"

        Returns:
            Same dict as observe() after the step.
        """
        ...

    @abc.abstractmethod
    def evaluator(self) -> dict:
        """
        Return ground-truth evaluator state (sim) or best available estimate (real).

        Required keys:
            x_m               float   position x [m]
            y_m               float   position y [m]
            yaw_rad           float   heading [rad]
            speed_m_s         float   scalar ground speed [m/s]
            body_contact      bool    any body-level contact right now
            body_contact_events int   total body contact events this episode
        """
        ...

    @abc.abstractmethod
    def close(self) -> None:
        """Release resources."""
        ...

    # ------------------------------------------------------------------
    # Context manager support (optional override)
    # ------------------------------------------------------------------

    def __enter__(self) -> "HardwareInterface":
        return self

    def __exit__(self, *args) -> None:
        self.close()


# ---------------------------------------------------------------------------
# SimInterface — wraps RealWorldEnv
# ---------------------------------------------------------------------------

class SimInterface(HardwareInterface):
    """
    Wraps RealWorldEnv as a HardwareInterface.

    This is the production simulator backend.  All other code should use
    this class rather than instantiating RealWorldEnv directly, so that
    switching to a real robot only requires swapping SimInterface for
    ROSInterface or SerialInterface.
    """

    def __init__(self, scene: str = "open", seed: int = 7,
                 fault: str = "none", fault_start_s: float = 0.2,
                 fault_end_s: float = 0.5, delay_s: float = 0.03,
                 sensor_noise: bool = True):
        from real_world_env import RealWorldEnv
        self._env = RealWorldEnv(
            scene=scene, seed=seed,
            fault=fault, fault_start_s=fault_start_s, fault_end_s=fault_end_s,
            delay_s=delay_s, sensor_noise=sensor_noise,
        )

    @property
    def solids(self):
        return self._env.solids

    def observe(self) -> dict:
        return self._env.observe()

    def step(self, action: str) -> dict:
        return self._env.step(action)

    def evaluator(self) -> dict:
        return self._env.evaluator()

    def close(self) -> None:
        self._env.close()

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# ROSInterface — ROS 2 topic bridge  [STUB — NOT IMPLEMENTED]
# ---------------------------------------------------------------------------

class ROSInterface(HardwareInterface):
    """
    ROS 2 bridge for a physical rover.

    Status: NOT_IMPLEMENTED.
    
    What is needed before this can be activated:
      1. ROS 2 Humble (or later) installed on the rover's on-board computer.
      2. Camera topic:   /camera/image_raw   (sensor_msgs/Image, 10 Hz, 96×128)
      3. IMU topic:      /imu/data           (sensor_msgs/Imu, 100 Hz)
      4. Command topic:  /cmd_vel            (geometry_msgs/Twist)
      5. Contact topic:  /whisker            (std_msgs/Bool[2]) or GPIO read
      6. Network link between rover and workstation (or on-board inference).
      7. Camera intrinsic calibration file (fx, fy, cx, cy, k1, k2).
      8. Physical rover dimensions confirmed (wheelbase, chassis size).

    Implementation plan (once hardware is available):
      - Use rclpy to subscribe to camera/IMU/contact topics.
      - Publish Twist messages on /cmd_vel for forward/left/right/brake.
      - Use an elapsed-time counter as time_s.
      - Decode sensor_msgs/Image to uint8 NumPy array.
      - Apply RealSensorModel.camera() for distortion correction (not addition).
      - Measure body contact via bumper topic or IMU shock threshold.
    """

    _NOT_IMPLEMENTED_MSG = (
        "ROSInterface is a stub. See hardware_interface.py docstring for the "
        "checklist of hardware parameters required before this can be activated. "
        "Use SimInterface for simulator-based development."
    )

    def __init__(self, **kwargs):
        raise NotImplementedError(self._NOT_IMPLEMENTED_MSG)

    def observe(self) -> dict: raise NotImplementedError(self._NOT_IMPLEMENTED_MSG)
    def step(self, action: str) -> dict: raise NotImplementedError(self._NOT_IMPLEMENTED_MSG)
    def evaluator(self) -> dict: raise NotImplementedError(self._NOT_IMPLEMENTED_MSG)
    def close(self) -> None: pass


# ---------------------------------------------------------------------------
# SerialInterface — Arduino / ESP32 motor controller  [STUB — NOT IMPLEMENTED]
# ---------------------------------------------------------------------------

class SerialInterface(HardwareInterface):
    """
    Serial bridge for a rover with an Arduino/ESP32 motor controller.

    Status: NOT_IMPLEMENTED.

    What is needed before this can be activated:
      1. Serial protocol specification:
           - Baud rate (e.g. 115200)
           - Packet format for action commands: e.g. b'CMD:forward\\n'
           - Packet format for sensor responses: e.g. JSON line per step
      2. Camera: USB webcam or separate Pi stream feeding RGB frames.
      3. IMU: MPU-6050 on the Arduino, readings included in serial response.
      4. Encoder: quadrature interrupts on Arduino, tick counts in response.
      5. Whisker: digital inputs on Arduino, state in response packet.
      6. Latency characterisation: measure round-trip serial delay.

    Implementation plan (once hardware is available):
      - Open pyserial port at specified baud.
      - On each step(), write action command; read JSON response.
      - Parse response into the standard obs dict.
      - Run camera capture in a background thread at 10 Hz.
    """

    _NOT_IMPLEMENTED_MSG = (
        "SerialInterface is a stub. See hardware_interface.py docstring for the "
        "serial protocol and hardware requirements. "
        "Use SimInterface for simulator-based development."
    )

    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 115200, **kwargs):
        raise NotImplementedError(self._NOT_IMPLEMENTED_MSG)

    def observe(self) -> dict: raise NotImplementedError(self._NOT_IMPLEMENTED_MSG)
    def step(self, action: str) -> dict: raise NotImplementedError(self._NOT_IMPLEMENTED_MSG)
    def evaluator(self) -> dict: raise NotImplementedError(self._NOT_IMPLEMENTED_MSG)
    def close(self) -> None: pass


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def make_interface(backend: str = "sim", **kwargs) -> HardwareInterface:
    """
    Create a HardwareInterface by name.

    Args:
        backend: "sim" | "ros" | "serial"
        **kwargs: Passed to the chosen backend constructor.

    Returns:
        A HardwareInterface instance.
    """
    if backend == "sim":
        return SimInterface(**kwargs)
    elif backend == "ros":
        return ROSInterface(**kwargs)      # will raise NotImplementedError
    elif backend == "serial":
        return SerialInterface(**kwargs)   # will raise NotImplementedError
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Choose: sim | ros | serial")
