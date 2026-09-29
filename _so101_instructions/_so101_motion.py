"""Home lifecycle for direct use of the pinned official SO-101 embodiment.

No motor driver or interpolation is implemented here. The upstream plugin
handles connection/calibration, ramping, pacing, limits, settling and disconnect.
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from inspect_robots_so101 import SOArmEmbodiment, packing

logger = logging.getLogger(__name__)


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


def _connected_driver(cfg: Any) -> Any:
    """Use the upstream follower configuration, with cleanup on connect failure."""
    from inspect_robots_so101.embodiment import _check_calibrated
    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    robot = SOFollower(SOFollowerRobotConfig(
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


class HomeSession:
    """Capture the starting pose and return home after completion or Ctrl-C.

    Upstream currently exposes home on reset, but no public connect-only or
    return-home API. This uses its pinned lifecycle helpers rather than copying
    its motor code. Hardware faults still suppress shutdown motion.
    """

    def __init__(self, arm: Any, *, capture_home: bool = True,
                 return_home: bool = True, home_pose: Any = None,
                 home_max_relative_target: float = 1.0) -> None:
        self.arm = arm
        self.capture_home = capture_home
        self.return_home = return_home
        self._configured_home_pose = home_pose if home_pose is not None else arm._cfg.home_pose
        self.home_pose = self._configured_home_pose
        self.home_max_relative_target = home_max_relative_target
        self.completed = False

    def __enter__(self) -> HomeSession:
        try:
            if self.arm._driver is None:
                # Like RoboColosseum, each connection captures its own initial
                # pose; later trials on that connection keep the same home.
                self.home_pose = self._configured_home_pose
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
        """Apply the upstream home cap only while its home ramp is running."""
        if self.home_pose is None:
            yield
            return
        evaluation_config = self.arm._cfg
        home_config = replace(evaluation_config, home_pose=self.home_pose,
                              max_relative_target=self.home_max_relative_target)
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
        """Return using the upstream ramp; another Ctrl-C aborts and disconnects."""
        _home_status(self.arm, "homing: ramping SO-101 back to home position "
                     + str(list(self.home_pose)))
        self.arm.settle_timeouts = 0
        self.arm._settle_disabled = False
        with self.homing():
            self.arm._home()
            self.arm._settle(np.asarray(self.home_pose, dtype=np.float64))

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
    """CLI lifecycle adapter; inherit upstream motor control and home ramp."""

    def __init__(self, *args: Any, capture_home: bool = True,
                 return_home: bool = True, home_pose: Any = None,
                 home_max_relative_target: float = 1.0, **kwargs: Any) -> None:
        if kwargs.get("driver_factory") is None:
            kwargs["driver_factory"] = _connected_driver
        super().__init__(*args, **kwargs)
        self._home_session = HomeSession(self, capture_home=capture_home, return_home=return_home,
                                         home_pose=home_pose,
                                         home_max_relative_target=home_max_relative_target)
        self._session_started = False
        self._home_gate_confirmed = False
        self._motion_failed = False

    def reset(self, scene: Any, *, seed: int | None = None) -> Any:
        try:
            if not self._session_started:
                self._motion_failed = False
                self._home_session.__enter__()
                self._session_started = True
            if self._home_session.home_pose is not None:
                # Match YAM's first-connection motion gate, followed by the
                # upstream SO-101 ramp and its existing scene readiness gate.
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
