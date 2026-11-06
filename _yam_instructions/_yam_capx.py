"""YAM single-arm joint profile for the CaP-X code-as-policy plugin.

The wrapped ``YAMEmbodiment`` keeps CAN driving, cameras, homing, operator
gates, the thermal guardrail, and parking; actions stay absolute joint radians
with normalized grippers, so its recording contract is unchanged. ``capx``
drives one arm at a time, so this adapter presents either the left or the
right arm as the 7-D single-arm profile, holds the other arm at its start
pose, and adds what ``capx`` needs on top through the shared
``inspect_robots_capx.embodiment.CapxArmEmbodiment`` base: depth and
intrinsics from the YAM RealSense reader, base-frame tool state, an optional
camera-to-base transform, and an in-process IK solver (``-P ik=embodiment``)
over i2rt's MuJoCo model of the arm with the configured joint limits.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import numpy.typing as npt
from inspect_robots_capx.embodiment import (
    DEFAULT_IK_ORIENTATION_WEIGHT,
    DEFAULT_IK_POSITION_TOLERANCE_M,
    IK_CONVERGED_M,
    IK_MAX_ITERATIONS,
    ORIENTATION_NOTE_DEG,
    CapxArmEmbodiment,
    load_extrinsics,
)
from inspect_robots_yam.packing import ARM_DOF, ARM_WIDTH, STATE_KEY, TOTAL_DIM

from inspect_robots.approver import DeltaLimitApprover, GuardrailContribution
from inspect_robots.spaces import Box, ObservationSpace, StateField, StateSpec
from inspect_robots.types import Action, Observation

SIDES = ("left", "right")
GRASP_SITE = "grasp_site"
#: Full 14-D proprioception, kept next to the 7-D driven-arm ``joint_pos``.
BIMANUAL_STATE_KEY = "bimanual_joint_pos"
_IK_DT = 0.01
_IK_DAMPING = 1e-4
_IK_ORIENTATION_CONVERGED_RAD = 1e-3


def arm_slice(side: str) -> slice:
    """Return the packed 14-D slice ``[j0..j5, gripper]`` of one arm."""
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    start = 0 if side == "left" else ARM_WIDTH
    return slice(start, start + ARM_WIDTH)


def _load_mink() -> tuple[Any, Any]:
    try:
        import mink
        import mujoco  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "YAM CaP-X IK needs mink and mujoco: uv pip install mink 'mujoco>=3.3.1'",
            name=exc.name,
        ) from exc
    return mujoco, mink


def default_model_path(gripper_type: str) -> str:
    """Return i2rt's combined MJCF of a YAM arm with the configured gripper."""
    try:
        from i2rt.robots.utils import (  # type: ignore[import-untyped]
            ArmType,
            GripperType,
            combine_arm_and_gripper_xml,
        )
    except ModuleNotFoundError as exc:
        from inspect_robots_yam._i2rt import I2RT_INSTALL_COMMAND

        raise ModuleNotFoundError(
            "i2rt provides the YAM robot model used for IK. Install it with: "
            + I2RT_INSTALL_COMMAND,
            name=exc.name,
        ) from exc
    path: str = combine_arm_and_gripper_xml(ArmType.YAM, GripperType[gripper_type])
    return path


class YamKinematics:
    """FK and position-weighted IK of one YAM arm over its MuJoCo model (mink).

    The model's joint ranges are intersected with the configured limits and
    enforced inside the QP, so solutions respect both. Gripper joints are
    pinned at mid-range for FK and IK; ``low``/``high`` are the effective six
    arm limits in radians.
    """

    def __init__(
        self,
        gripper_type: str,
        *,
        joint_low: Any,
        joint_high: Any,
        model_path: str | None = None,
        site: str = GRASP_SITE,
    ) -> None:
        mujoco, mink = _load_mink()
        path = model_path if model_path is not None else default_model_path(gripper_type)
        model = mujoco.MjModel.from_xml_path(str(path))
        ranges = np.asarray(model.jnt_range, dtype=np.float64)
        if ranges.ndim != 2 or not (ARM_DOF <= ranges.shape[0] <= ARM_DOF + 2):
            raise ValueError(
                f"YAM model must expose {ARM_DOF} arm joints and 0..2 gripper joints, "
                f"got range shape {ranges.shape}"
            )
        low = np.asarray(joint_low, dtype=np.float64).reshape(-1)[:ARM_DOF]
        high = np.asarray(joint_high, dtype=np.float64).reshape(-1)[:ARM_DOF]
        if (
            low.shape != (ARM_DOF,)
            or high.shape != (ARM_DOF,)
            or not np.all(np.isfinite(low))
            or not np.all(np.isfinite(high))
        ):
            raise ValueError("YAM IK joint limits must be six finite radians per bound")
        effective_low = np.maximum(ranges[:ARM_DOF, 0], low)
        effective_high = np.minimum(ranges[:ARM_DOF, 1], high)
        if np.any(effective_low >= effective_high):
            raise ValueError(
                "configured joint limits do not overlap the YAM model ranges: "
                f"low {effective_low.tolist()} high {effective_high.tolist()}"
            )
        model.jnt_range[:ARM_DOF, 0] = effective_low
        model.jnt_range[:ARM_DOF, 1] = effective_high
        self.low = effective_low.copy()
        self.high = effective_high.copy()
        self._gripper_pin = np.mean(ranges[ARM_DOF:], axis=1)
        self._site = site
        self._mink = mink
        self._configuration = mink.Configuration(model)
        self._limits = [mink.ConfigurationLimit(model)]
        if model.site(site).id < 0:  # pragma: no cover - mujoco raises first
            raise ValueError(f"YAM model has no site {site!r}")

    def _full_q(self, joints: Any) -> npt.NDArray[np.float64]:
        q = np.asarray(joints, dtype=np.float64).reshape(-1)
        if q.shape != (ARM_DOF,) or not np.all(np.isfinite(q)):
            raise ValueError(f"expected {ARM_DOF} finite arm joints, got {q.tolist()}")
        return np.concatenate((q, self._gripper_pin))

    def fk(self, joints: Any) -> npt.NDArray[np.float64]:
        """Return the 4x4 base-frame pose of the grasp site for six arm joints."""
        self._configuration.update(self._full_q(joints))
        pose = self._configuration.get_transform_frame_to_world(self._site, "site").as_matrix()
        return np.asarray(pose, dtype=np.float64)

    def ik(
        self,
        seed: Any,
        target: npt.NDArray[np.float64],
        orientation_weight: float,
        max_iters: int = IK_MAX_ITERATIONS,
        position_tolerance_m: float = IK_CONVERGED_M,
    ) -> npt.NDArray[np.float64]:
        """Return six arm joints after differential IK from ``seed`` toward ``target``.

        Position is weighted 1.0 and orientation ``orientation_weight`` (0 is
        position-only). Iteration stops once the position is within
        ``position_tolerance_m`` and the weighted orientation has converged,
        or after ``max_iters`` passes; the caller judges the residual with FK.
        """
        mink = self._mink
        task = mink.FrameTask(
            frame_name=self._site,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=max(float(orientation_weight), 0.0),
            lm_damping=1.0,
        )
        task.set_target(mink.SE3.from_matrix(np.asarray(target, dtype=np.float64)))
        self._configuration.update(self._full_q(seed))
        for _ in range(max_iters):
            try:
                velocity = mink.solve_ik(
                    self._configuration,
                    [task],
                    _IK_DT,
                    "quadprog",
                    damping=_IK_DAMPING,
                    limits=self._limits,
                )
            except mink.NoSolutionFound as exc:
                raise ValueError("YAM IK quadratic program is infeasible for this target") from exc
            self._configuration.integrate_inplace(velocity, _IK_DT)
            error = task.compute_error(self._configuration)
            position_done = float(np.linalg.norm(error[:3])) <= position_tolerance_m
            orientation_done = (
                orientation_weight <= 0.0
                or float(np.linalg.norm(error[3:])) <= _IK_ORIENTATION_CONVERGED_RAD
            )
            if position_done and orientation_done:
                break
        return np.asarray(self._configuration.q[:ARM_DOF], dtype=np.float64).copy()


class ObservationDepth:
    """Depth and intrinsics the YAM RealSense reader publishes in ``Observation.extra``.

    ``inspect-robots-yam`` serves ``{camera}_depth`` (meters, lazily resolved)
    and ``{camera}_intrinsics`` for every ``*_depth_serial`` camera. ``bind()``
    is called with each observation's extras, and ``depth()`` resolves the
    bound thunk when model code asks, so it reflects the scene at that time.
    """

    def __init__(self, camera: str) -> None:
        self._camera = camera
        self._depth_key = f"{camera}_depth"
        self._intrinsics_key = f"{camera}_intrinsics"
        self._extra: Mapping[str, Any] | None = None

    def bind(self, extra: Mapping[str, Any]) -> None:
        """Adopt the newest observation's extras."""
        self._extra = extra

    def _require(self, key: str) -> Any:
        if self._extra is None:
            raise RuntimeError("depth is unavailable before the embodiment is reset")
        if key not in self._extra:
            slot = self._camera.removesuffix("_cam")
            raise RuntimeError(
                f"camera {self._camera!r} publishes no {key!r}; configure {slot}_depth_serial "
                "(RealSense colour+depth) for that slot"
            )
        return self._extra[key]

    def install(self) -> None:
        """Check the bound observation carries depth and intrinsics for the camera."""
        self._require(self._depth_key)
        self._require(self._intrinsics_key)

    def depth(self) -> npt.NDArray[np.float32]:
        """Return the (H, W) float32 depth image in meters; 0 marks missing depth."""
        value = self._require(self._depth_key)
        depth: npt.NDArray[np.float32] = np.asarray(
            value() if callable(value) else value, dtype=np.float32
        )
        if depth.ndim != 2 or depth.size == 0:
            raise ValueError(f"{self._depth_key} must be an (H, W) array, got {depth.shape}")
        return depth

    def intrinsics(self) -> npt.NDArray[np.float64]:
        """Return the 3x3 camera matrix valid at the published image resolution."""
        matrix = np.asarray(self._require(self._intrinsics_key), dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"{self._intrinsics_key} must be a finite 3x3 matrix")
        return matrix.copy()


class _ArmSliceApprover:
    """Run a two-arm approver on the full command behind a one-arm action space.

    The idle arm is filled from the hold pose, the inner approver sees a
    private store (its 14-D references must not leak into the 7-D chain), and
    a substituted target is projected back and rewinds the delta limiter so
    later deltas are measured from the pose that ran.
    """

    def __init__(self, inner: Any, embodiment: YamCapxEmbodiment, store: dict[str, Any]) -> None:
        self._inner = inner
        self._embodiment = embodiment
        self._store = store

    def review(self, action: Action, store: dict[str, Any]) -> Action:
        """Approve unchanged by identity or return the projected substitute."""
        expanded = self._embodiment._expand(action)
        reviewed = self._inner.review(expanded, self._store)
        if reviewed is expanded:
            return action
        data = np.asarray(reviewed.data, dtype=np.float64)[self._embodiment._active].copy()
        DeltaLimitApprover.rewind_reference(store, data)
        return replace(action, data=data, meta=dict(reviewed.meta))


class YamCapxEmbodiment(CapxArmEmbodiment):
    """Drive one YAM arm through the CaP-X single-arm profile; hold the other."""

    def __init__(
        self,
        arm: Any,
        kinematics: Any,
        *,
        side: str = "left",
        depth_camera: str = "top_cam",
        depth_source: Any = None,
        extrinsics: npt.NDArray[np.float64] | None = None,
        ik_position_tolerance_m: float = DEFAULT_IK_POSITION_TOLERANCE_M,
        ik_orientation_weight: float = DEFAULT_IK_ORIENTATION_WEIGHT,
        log: Callable[[str], None] | None = None,
    ) -> None:
        config = arm._cfg
        if config.control_interface != "joints" or config.joints_are_delta:
            raise ValueError(
                "YAM CaP-X control requires control_interface='joints' with absolute targets"
            )
        self._side = side
        self._active = arm_slice(side)
        self._idle = arm_slice("right" if side == "left" else "left")
        self._kinematics = kinematics
        self._hold: npt.NDArray[np.float64] | None = None
        self._inner_store: dict[str, Any] = {}
        depth = depth_source if depth_source is not None else ObservationDepth(depth_camera)
        self._observation_depth = depth if isinstance(depth, ObservationDepth) else None
        super().__init__(
            arm,
            name=f"yam_{side}_capx",
            label=f"YAM {side} arm",
            arm_dof=ARM_DOF,
            joint_low=kinematics.low,
            joint_high=kinematics.high,
            limits_label="configured joint limits",
            depth_camera=depth_camera,
            depth_source=depth,
            extrinsics=extrinsics,
            ik_position_tolerance_m=ik_position_tolerance_m,
            ik_orientation_weight=ik_orientation_weight,
            log=log,
        )

    # -- single-arm profile ---------------------------------------------------------

    def _base_info(self) -> Any:
        info = self._arm.info
        space = info.action_space
        semantics = space.semantics
        labels = (*(f"{self._side}_j{index}" for index in range(ARM_DOF)), "gripper")
        max_step = None if semantics.max_step is None else tuple(semantics.max_step[self._active])
        box = Box(
            shape=(ARM_WIDTH,),
            low=np.asarray(space.low, dtype=np.float64)[self._active].copy(),
            high=np.asarray(space.high, dtype=np.float64)[self._active].copy(),
            semantics=replace(semantics, dim_labels=labels, max_step=max_step),
        )
        declared = info.observation_space.state.fields if info.observation_space.state else ()
        state = StateSpec(
            fields=(
                StateField(key=STATE_KEY, shape=(ARM_WIDTH,), unit="rad+normalized"),
                StateField(key=BIMANUAL_STATE_KEY, shape=(TOTAL_DIM,), unit="rad+normalized"),
                *(field for field in declared if field.key != STATE_KEY),
            )
        )
        return replace(
            info,
            action_space=box,
            observation_space=ObservationSpace(
                cameras=info.observation_space.cameras, state_keys=state.keys, state=state
            ),
        )

    def _kinematics_docs(self) -> str:
        side = self._side
        return (
            f"YAM {side} arm for code-as-policy; the other arm holds its start pose. "
            f"move_to_joints takes the six {side} arm joints [{side}_j0 .. {side}_j5] in "
            "radians; open_gripper/close_gripper drive the 0..1 gripper (1 = open, about "
            "9.5 cm between the jaws). solve_ik(position, quaternion_wxyz) solves the "
            f"grasp_site pose (the point between the fingertips) in the {side} arm's base "
            "frame (meters; x forward out of the base, y left, z up) within the joint "
            "limits. It returns those six radians, raises when the position residual "
            f"exceeds {self._ik_position_tolerance_m * 1000:.0f} mm, and prints a note to "
            f"stderr when orientation misses by more than {ORIENTATION_NOTE_DEG:.0f} degrees "
            "(position is prioritized; a six-joint arm reaches most orientations inside "
            "its limits). The gripper approach axis is the tool frame's local +z (wrist "
            "toward fingertips), the same convention as Contact-GraspNet poses, and the "
            "jaws open and close along the tool's local +y. At all-zero joints the tool "
            "+z points along base +x (horizontal, forward); a straight-down approach is "
            "close to quaternion_wxyz [0, 0, 1, 0], and rotating about the approach axis "
            "(a yaw of the whole quaternion) selects the jaw direction; the wrist roll "
            "cannot turn a full half turn, so when IK notes a large orientation residual "
            "try the jaw direction rotated by 180 degrees. Contact-GraspNet places its pose "
            "origin roughly 0.10 m behind the fingertip contact along +z, while grasp_site "
            "sits between "
            "the fingertips, so shift a grasp position forward along its +z before IK. "
            "state joint_pos is the driven arm [j0 .. j5, gripper] and "
            f"state {BIMANUAL_STATE_KEY} the full 14-D two-arm vector; state eef_pos and "
            "eef_quat_wxyz report the measured tool pose at the start of each turn, and "
            "obs['eef_rotation'] is the same orientation as a matrix. segment() and "
            f"plan_grasp() use camera {self._depth_camera!r}; obs['depth'] is aligned to "
            "that image in meters (0 = no reading) and obs['intrinsics'] is its 3x3 "
            "matrix. "
        )

    def _project(self, observation: Observation) -> Observation:
        full = np.asarray(observation.state[STATE_KEY], dtype=np.float64)
        if full.shape != (TOTAL_DIM,) or not np.all(np.isfinite(full)):
            raise ValueError(f"YAM joint observation is invalid: {full.tolist()}")
        if self._hold is None:
            self._hold = full[self._idle].copy()
        if self._observation_depth is not None:
            self._observation_depth.bind(observation.extra)
        return replace(
            observation,
            state={
                **observation.state,
                STATE_KEY: full[self._active].copy(),
                BIMANUAL_STATE_KEY: full,
            },
        )

    def _expand(self, action: Action) -> Action:
        if self._hold is None:
            raise RuntimeError("YAM CaP-X step before reset: no hold pose for the idle arm")
        data = np.asarray(action.data, dtype=np.float64).reshape(-1)
        if data.shape != (ARM_WIDTH,):
            raise ValueError(f"expected a {ARM_WIDTH}-D single-arm action, got shape {data.shape}")
        full = np.empty(TOTAL_DIM, dtype=np.float64)
        full[self._active] = data
        full[self._idle] = self._hold
        return replace(action, data=full)

    # -- kinematics hooks -----------------------------------------------------------

    def _fk(self, joints: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        pose = np.asarray(self._kinematics.fk(joints), dtype=np.float64)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ValueError("forward kinematics returned an invalid pose")
        return pose

    def _iterate_ik(
        self,
        seed: npt.NDArray[np.float64],
        target: npt.NDArray[np.float64],
        orientation_weight: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        solved = np.asarray(
            self._kinematics.ik(seed, target, orientation_weight, IK_MAX_ITERATIONS),
            dtype=np.float64,
        ).reshape(-1)
        if solved.shape != (ARM_DOF,) or not np.all(np.isfinite(solved)):
            raise ValueError("YAM IK returned invalid joint values")
        achieved = self._fk(solved)
        return solved, achieved, float(np.linalg.norm(achieved[:3, 3] - target[:3, 3]))

    # -- lifecycle --------------------------------------------------------------------

    def contribute_guardrails(self, action_space: Box) -> GuardrailContribution:
        """Forward YAM's guardrails (collision holds) with the idle arm held."""
        contribution: GuardrailContribution = self._arm.contribute_guardrails(
            self._arm.info.action_space
        )
        approvers = tuple(
            (name, _ArmSliceApprover(approver, self, self._inner_store))
            for name, approver in contribution.approvers
        )
        warnings = contribution.warnings
        if approvers:
            warnings = (
                *warnings,
                f"guardrails review the full two-arm command with the idle arm held "
                f"at its start pose (driving the {self._side} arm)",
            )
        return GuardrailContribution(approvers=approvers, warnings=warnings)

    def reset(self, scene: Any, *, seed: int | None = None) -> Observation:
        """Home both arms, then hold the idle arm where the ramp left it."""
        self._hold = None
        self._inner_store.clear()
        return super().reset(scene, seed=seed)

    def observe_parked(self) -> Observation | None:
        """Project YAM's final grader view onto the single-arm state layout."""
        parked: Observation | None = self._arm.observe_parked()
        return None if parked is None else self._project(parked)


__all__ = [
    "BIMANUAL_STATE_KEY",
    "DEFAULT_IK_ORIENTATION_WEIGHT",
    "DEFAULT_IK_POSITION_TOLERANCE_M",
    "SIDES",
    "ObservationDepth",
    "YamCapxEmbodiment",
    "YamKinematics",
    "arm_slice",
    "default_model_path",
    "load_extrinsics",
]
