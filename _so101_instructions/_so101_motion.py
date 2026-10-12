"""SO-101 home lifecycle with a paced, bounded joint ramp.

The upstream plugin retains motor connection, pacing, limits, settling, and
disconnect. This adapter checks measured position after the ramp and corrects lag.
"""

from __future__ import annotations

import logging
import math
import sys
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from inspect_robots_so101 import SOArmEmbodiment, packing

logger = logging.getLogger(__name__)


def _read_motor_status(driver: Any, motor: str) -> str:
    """Best-effort register snapshot before a failed home disconnects torque."""
    bus = getattr(driver, "bus", None)
    if bus is None:
        return ""
    readings = {}
    for register in (
        "Operating_Mode", "Min_Position_Limit", "Max_Position_Limit",
        "Min_Voltage_Limit", "Max_Voltage_Limit",
        "P_Coefficient", "I_Coefficient", "D_Coefficient",
        "Minimum_Startup_Force", "CW_Dead_Zone", "CCW_Dead_Zone",
        "Max_Torque_Limit", "Torque_Limit", "Goal_Velocity",
        "Goal_Position", "Present_Position", "Torque_Enable", "Present_Voltage",
        "Present_Temperature", "Present_Load", "Present_Current", "Status", "Moving",
    ):
        try:
            readings[register] = bus.read(register, motor, normalize=False, num_retry=2)
        except Exception as exc:
            readings[register] = f"{type(exc).__name__}: {exc}"
    return f"; live motor registers={readings}"


def _close_partial_robot(robot: Any) -> None:
    """Release components even when SOFollower.is_connected is still false."""
    # SOFollower.disconnect requires all cameras to be connected, so it cannot
    # clean up a front camera whose read thread started before head failed.
    for name, camera in robot.cameras.items():
        try:
            if camera.is_connected or getattr(camera, "thread", None) is not None:
                camera.disconnect()
        except Exception:
            logger.warning("Failed to close SO-101 camera %s", name, exc_info=True)
    try:
        if robot.bus.is_connected:
            robot.bus.disconnect(robot.config.disable_torque_on_disconnect)
    except Exception:
        logger.warning("Failed to close SO-101 motor bus", exc_info=True)


def _configure_follower_with_arm_p(bus: Any, arm_p: int) -> None:
    """Match LeRobot 0.5.1 setup while selecting the five arm motors' P gain."""
    from lerobot.motors.feetech import OperatingMode

    with bus.torque_disabled():
        bus.configure_motors()
        for motor in bus.motors:
            bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            bus.write("P_Coefficient", motor, arm_p if motor != "gripper" else 16)
            bus.write("I_Coefficient", motor, 0)
            bus.write("D_Coefficient", motor, 32)
            if motor == "gripper":
                bus.write("Max_Torque_Limit", motor, 500)
                bus.write("Protection_Current", motor, 250)
                bus.write("Overload_Torque", motor, 25)


def _connected_driver(cfg: Any, *, arm_position_p_coefficient: int | None = None) -> Any:
    """Use the upstream follower configuration, with cleanup on connect failure."""
    from inspect_robots_so101.embodiment import _check_calibrated
    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    if arm_position_p_coefficient is None:
        follower_type = SOFollower
    else:
        class FollowerWithArmP(SOFollower):
            def configure(self) -> None:
                _configure_follower_with_arm_p(self.bus, arm_position_p_coefficient)

        follower_type = FollowerWithArmP

    robot = follower_type(SOFollowerRobotConfig(
        id=cfg.robot_id,
        calibration_dir=Path(cfg.calibration_dir) if cfg.calibration_dir else None,
        port=cfg.port,
        cameras=dict(cfg.camera_configs or {}),
        max_relative_target=cfg.max_relative_target,
        use_degrees=cfg.use_degrees,
        disable_torque_on_disconnect=cfg.disable_torque_on_disconnect,
    ))
    try:
        robot.connect(calibrate=False)
        _check_calibrated(robot, cfg)
    except BaseException as exc:
        _close_partial_robot(robot)
        if isinstance(exc, ConnectionError) and exc.__cause__ is not None:
            # Keep SDK errors such as a busy device visible in the CLI/run log.
            raise ConnectionError(f"{exc} Underlying error: {exc.__cause__}") from exc
        raise
    return robot


def _home_status(arm: Any, message: str) -> None:
    """Show homing through the upstream operator's injectable output surface."""
    output = getattr(arm._operator, "output_fn", logger.info)
    output(message)


def _read_leader_pose(port: str, robot_id: str, calibration_dir: str | None,
                      use_degrees: bool) -> tuple[float, ...]:
    """Read the leader's current teleoperation target without recalibrating it."""
    from lerobot.teleoperators.so_leader import SO101LeaderConfig, SOLeader

    leader = SOLeader(SO101LeaderConfig(
        port=port, id=robot_id,
        calibration_dir=Path(calibration_dir) if calibration_dir else None,
        use_degrees=use_degrees,
    ))
    if not leader.calibration_fpath.is_file():
        raise RuntimeError(f"SO-101 leader calibration is missing: {leader.calibration_fpath}")
    try:
        leader.connect(calibrate=False)
        if not leader.is_calibrated:
            raise RuntimeError(
                f"SO-101 leader calibration does not match the motors: {leader.calibration_fpath}"
            )
        pose = packing.from_obs_dict(leader.get_action())
        if not np.all(np.isfinite(pose)):
            raise RuntimeError(f"SO-101 leader returned a non-finite home pose: {pose}")
        return tuple(float(value) for value in pose)
    finally:
        # connect() may fail after opening the bus.
        if leader.bus.is_connected:
            leader.bus.disconnect()


class HomeSession:
    """Capture a configured source pose and return home after completion or Ctrl-C.

    Upstream currently exposes home on reset, but no public connect-only or
    return-home API. This uses its pinned lifecycle helpers rather than copying
    its motor code. Hardware faults still suppress shutdown motion.
    """

    def __init__(self, arm: Any, *, capture_home: bool = True,
                 return_home: bool = True, home_pose: Any = None,
                 home_max_relative_target: float = 5.0,
                 home_ramp_secs: float = 3.0, home_tolerance: float = 1.0,
                 leader_port: str | None = None, leader_id: str = "leader_arm",
                 leader_calibration_dir: str | None = None,
                 leader_reader: Callable[[str, str, str | None, bool], tuple[float, ...]] =
                 _read_leader_pose) -> None:
        self.arm = arm
        self.capture_home = capture_home
        self.return_home = return_home
        self._configured_home_pose = home_pose if home_pose is not None else arm._cfg.home_pose
        self.home_pose = self._configured_home_pose
        self.home_max_relative_target = home_max_relative_target
        self.home_ramp_secs = home_ramp_secs
        self.home_tolerance = home_tolerance
        self.leader_port = leader_port
        self.leader_id = leader_id
        self.leader_calibration_dir = leader_calibration_dir
        self.leader_reader = leader_reader
        self.completed = False

    def __enter__(self) -> HomeSession:
        try:
            if self.arm._driver is None:
                # Later trials on this connection keep the same home.
                self.home_pose = self._configured_home_pose
                if self.leader_port is not None:
                    self.home_pose = self.leader_reader(
                        self.leader_port, self.leader_id, self.leader_calibration_dir,
                        self.arm._cfg.use_degrees,
                    )
                    # Reject an unreachable leader pose before sending motor commands.
                    replace(self.arm._cfg, home_pose=self.home_pose,
                            max_relative_target=self.home_max_relative_target)
                    logger.info("SO-101 home captured from leader %s: %s",
                                self.leader_id, self.home_pose)
                self.arm._driver = self.arm._driver_factory(self.arm._cfg)
            if self.capture_home and self.home_pose is None:
                measured = packing.from_obs_dict(self.arm._driver.get_observation())
                self.home_pose = tuple(measured)
                # Let the official config validate measured bounds and finite values.
                replace(self.arm._cfg, home_pose=self.home_pose,
                        max_relative_target=self.home_max_relative_target)
                logger.info("SO-101 home captured from starting joint_pos: %s", self.home_pose)
            return self
        except BaseException:
            self.arm.close()
            raise

    @contextmanager
    def homing(self):
        """Apply the driver home cap and disable evaluation settling during home."""
        if self.home_pose is None:
            yield
            return
        evaluation_config = self.arm._cfg
        home_config = replace(evaluation_config, home_pose=self.home_pose,
                              max_relative_target=self.home_max_relative_target,
                              settle_tolerance=None)
        driver = self.arm._require_driver()
        driver_config = getattr(driver, "config", None)
        try:
            self.arm._cfg = home_config
            if driver_config is not None:
                driver.config = replace(driver_config,
                                        max_relative_target=self.home_max_relative_target)
            yield
        finally:
            self.arm._cfg = evaluation_config
            if driver_config is not None:
                driver.config = driver_config

    def ramp_home(self) -> None:
        """Return using the measured-position ramp; another Ctrl-C disconnects."""
        _home_status(self.arm, "homing: ramping SO-101 back to home position "
                     + str(list(self.home_pose)))
        self.arm.settle_timeouts = 0
        self.arm._settle_disabled = False
        with self.homing():
            self.arm._home()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        interrupted = exc_type is not None and issubclass(exc_type, KeyboardInterrupt)
        try:
            if (((exc_type is None and self.completed) or interrupted) and self.return_home
                    and not getattr(self.arm, "_motion_failed", False)
                    and getattr(self.arm, "_home_gate_confirmed", True)
                    and self.arm._driver is not None and self.home_pose is not None):
                self.ramp_home()
        finally:
            SOArmEmbodiment.close(self.arm)


class CLIArm(SOArmEmbodiment):
    """CLI lifecycle adapter with feedback home and upstream joint control."""

    def __init__(self, *args: Any, capture_home: bool = True,
                 return_home: bool = True, home_pose: Any = None,
                 home_max_relative_target: float = 5.0,
                 home_ramp_secs: float = 3.0, home_tolerance: float = 1.0,
                 arm_position_p_coefficient: int | None = None,
                 leader_port: str | None = None, leader_id: str = "leader_arm",
                 leader_calibration_dir: str | None = None,
                 leader_reader: Callable[[str, str, str | None, bool], tuple[float, ...]] =
                 _read_leader_pose,
                 **kwargs: Any) -> None:
        if kwargs.get("driver_factory") is None:
            if arm_position_p_coefficient is None:
                kwargs["driver_factory"] = _connected_driver
            else:
                kwargs["driver_factory"] = (
                    lambda cfg: _connected_driver(
                        cfg, arm_position_p_coefficient=arm_position_p_coefficient
                    )
                )
        super().__init__(*args, **kwargs)
        self._home_session = HomeSession(self, capture_home=capture_home, return_home=return_home,
                                         home_pose=home_pose,
                                         home_max_relative_target=home_max_relative_target,
                                         home_ramp_secs=home_ramp_secs,
                                         home_tolerance=home_tolerance,
                                         leader_port=leader_port, leader_id=leader_id,
                                         leader_calibration_dir=leader_calibration_dir,
                                         leader_reader=leader_reader)
        self._session_started = False
        self._home_gate_confirmed = False
        self._motion_failed = False

    def _home(self) -> None:
        """Interpolate from the measured start, then check and correct residual."""
        if self._cfg.home_pose is None:
            return
        target = np.asarray(self._cfg.home_pose, dtype=np.float64)
        step_limit = self._cfg.max_relative_target
        if step_limit is None:
            raise RuntimeError("home_pose is set but max_relative_target is None")
        driver = self._require_driver()
        current = packing.from_obs_dict(driver.get_observation())
        if not np.all(np.isfinite(current)):
            raise RuntimeError(f"cannot home: initial motor observation is invalid: {current}")
        if (np.any(current < self._cfg.low - step_limit)
                or np.any(current > self._cfg.high + step_limit)):
            raise RuntimeError(
                f"cannot home: observed start pose {current} is outside joint limits "
                f"[{self._cfg.low}, {self._cfg.high}] by more than max_relative_target "
                f"({step_limit}); check calibration"
            )
        tolerance = self._home_session.home_tolerance
        max_dist = float(np.max(np.abs(target - current)))
        hz = self._cfg.control_hz
        ramp_steps = max(1, math.ceil(max_dist / step_limit),
                         math.ceil(self._home_session.home_ramp_secs * hz))
        start = current.copy()
        self._t_last = self._clock()
        for step in range(1, ramp_steps + 1):
            # The driver also bounds each command relative to actual motor
            # position, so a lagging joint cannot jump to a later waypoint.
            self._send(start + (target - start) * (step / ramp_steps))
            self._pace()

        current = self._home_observation()
        # Allow the full bounded correction window if a motor trails the ramp.
        # Encoder readings can stay unchanged for several polls while a loaded
        # servo starts moving; two stagnant 100 ms samples do not prove a stall.
        correction_steps = max(1, math.ceil(2 * hz))
        for _ in range(correction_steps):
            if (np.max(np.abs(target[:5] - current[:5])) <= tolerance
                    and abs(target[5] - current[5]) <= tolerance):
                break
            self._send(current + np.clip(target - current, -step_limit, step_limit))
            self._pace()
            current = self._home_observation()

        arm_errors = np.abs(target[:5] - current[:5])
        gripper_error = abs(float(target[5] - current[5]))
        if np.max(arm_errors) <= tolerance and gripper_error <= tolerance:
            return
        if np.max(arm_errors) <= tolerance:
            self._motion_failed = True
            raise RuntimeError(
                "SO-101 home pose not reached: "
                f"gripper target={target[5]:.2f}, measured={current[5]:.2f}, "
                f"error={gripper_error:.2f} after the home ramp"
                + _read_motor_status(driver, "gripper")
            )
        worst = int(np.argmax(arm_errors))
        self._motion_failed = True
        raise RuntimeError(
            "SO-101 home pose not reached: "
            f"{packing.MOTORS[worst]} target={target[worst]:.2f}, "
            f"measured={current[worst]:.2f}, error={arm_errors[worst]:.2f} degrees "
            f"after the home ramp (limit={step_limit:g} degrees per step, "
            f"tolerance={tolerance:g} degrees). "
            "Check the motor, power, and linkage before running EEF actions."
            + _read_motor_status(driver, packing.MOTORS[worst])
        )

    def _home_observation(self) -> np.ndarray:
        measured = packing.from_obs_dict(self._require_driver().get_observation())
        if not np.all(np.isfinite(measured)):
            raise RuntimeError(f"cannot home: motor observation is invalid: {measured}")
        return measured

    def reset(self, scene: Any, *, seed: int | None = None) -> Any:
        try:
            if not self._session_started:
                self._motion_failed = False
                self._home_session.__enter__()
                self._session_started = True
            if self._home_session.home_pose is not None:
                # Match YAM's first-connection motion gate, followed by the
                # feedback home ramp and the existing scene readiness gate.
                if not self._home_gate_confirmed:
                    self._operator.wait_ready(
                        "Arm will move to the home pose - stand clear, then press Enter..."
                    )
                    self._home_gate_confirmed = True
                _home_status(self, "homing: ramping SO-101 to home position "
                             + str(list(self._home_session.home_pose)))
            with self._home_session.homing():
                return super().reset(scene, seed=seed)
        except BaseException as exc:
            if not isinstance(exc, KeyboardInterrupt):
                self._motion_failed = True
            raise

    def step(self, action: Any) -> Any:
        try:
            return super().step(action)
        except BaseException as exc:
            if not isinstance(exc, KeyboardInterrupt):
                self._motion_failed = True
            raise

    def close(self) -> None:
        try:
            if self._session_started:
                self._session_started = False
                exc_type, exc, tb = sys.exc_info()
                self._home_session.completed = not self._motion_failed
                self._home_session.__exit__(exc_type, exc, tb)
            else:
                super().close()
        finally:
            self._home_gate_confirmed = False

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is not None and not issubclass(exc_type, KeyboardInterrupt):
            self._motion_failed = True
        self.close()
