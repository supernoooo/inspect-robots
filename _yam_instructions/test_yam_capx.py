"""Offline YAM CaP-X adapter tests; the CAN driver, cameras, depth, and the LLM are fakes."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest
from _run_yam import configured_yam_capx
from _yam_capx import (
    BIMANUAL_STATE_KEY,
    ObservationDepth,
    YamCapxEmbodiment,
    YamKinematics,
    arm_slice,
)
from inspect_robots_capx import CapxPolicy
from inspect_robots_capx.embodiment import quat_wxyz_from_matrix
from inspect_robots_yam.config import DEFAULT_EEF_HOME_POSE, DEFAULT_JOINT_HOME_POSE, YamConfig
from inspect_robots_yam.embodiment import YAMEmbodiment

from inspect_robots import Action, Scene
from inspect_robots.approver import _LAST_APPROVED_KEY, GuardrailContribution
from inspect_robots.errors import ConfigError
from inspect_robots.types import Observation

HAS_KINEMATICS = all(
    importlib.util.find_spec(name) is not None for name in ("i2rt", "mink", "mujoco")
)
needs_kinematics = pytest.mark.skipif(
    not HAS_KINEMATICS, reason="i2rt, mink, and mujoco are required for real-kinematics tests"
)
HOME = np.asarray(DEFAULT_JOINT_HOME_POSE)
IMAGE = (4, 6)
INTRINSICS = np.array([[600.0, 0.0, 3.0], [0.0, 600.0, 2.0], [0.0, 0.0, 1.0]])
EXTRINSICS = np.array(
    [
        [0.0, -1.0, 0.0, 0.30],
        [-1.0, 0.0, 0.0, 0.00],
        [0.0, 0.0, -1.0, 0.60],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


class EchoDriver:
    """Two-arm driver whose measured state echoes the last command."""

    def __init__(self) -> None:
        self.state = HOME.copy()
        self.commands: list[np.ndarray] = []
        self.closed = False

    def get_joint_pos(self) -> np.ndarray:
        return self.state.copy()

    def get_joint_eff(self) -> np.ndarray:
        return np.zeros(14)

    def get_motor_temps(self) -> np.ndarray:
        return np.full(14, 30.0)

    def command_joint_pos(self, target: np.ndarray) -> None:
        self.state = np.asarray(target, dtype=np.float64).copy()
        self.commands.append(self.state.copy())

    def close(self) -> None:
        self.closed = True


def _cameras(cfg: YamConfig) -> dict[str, np.ndarray]:
    image = np.zeros((cfg.cam_height, cfg.cam_width, 3), dtype=np.uint8)
    return {"top_cam": image, "left_cam": image, "right_cam": image}


def _depth(cfg: YamConfig) -> dict[str, Any]:
    return {
        "top_cam_depth": lambda: np.full(IMAGE, 0.5, dtype=np.float32),
        "top_cam_intrinsics": INTRINSICS,
    }


class FakeKinematics:
    """Tool position = joints[:3] (meters), tool yaw = joints[3]; IK is exact and recorded."""

    def __init__(self) -> None:
        self.low = np.full(6, -2.0)
        self.high = np.full(6, 2.0)
        self.ik_calls: list[tuple[np.ndarray, float, int]] = []
        self.results: list[np.ndarray] = []

    def fk(self, joints: np.ndarray) -> np.ndarray:
        q = np.asarray(joints, dtype=np.float64)
        pose = np.eye(4)
        yaw = float(q[3])
        pose[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
        pose[:3, 3] = q[:3]
        return pose

    def ik(self, seed: np.ndarray, target: np.ndarray, weight: float, max_iters: int) -> np.ndarray:
        self.ik_calls.append((np.asarray(seed, dtype=np.float64).copy(), weight, max_iters))
        if self.results:
            return self.results.pop(0)
        q = np.asarray(seed, dtype=np.float64).copy()
        q[:3] = target[:3, 3]
        q[3] = np.arctan2(target[1, 0], target[0, 0])
        return q


def _arm(driver: EchoDriver, **overrides: Any) -> YAMEmbodiment:
    config = YamConfig(
        cam_height=IMAGE[0],
        cam_width=IMAGE[1],
        unattended=True,
        rest_secs=0.5,
        collision_guardrail=False,
        **overrides,
    )
    return YAMEmbodiment(
        config,
        driver_factory=lambda _cfg: driver,
        camera_reader=_cameras,
        depth_reader=_depth,
        sleep_fn=lambda _seconds: None,
        clock=lambda: 0.0,
    )


def _build(side: str = "left", *, extrinsics: np.ndarray | None = None, **overrides: Any):
    driver = EchoDriver()
    kinematics = FakeKinematics()
    notes: list[str] = []
    wrapped = YamCapxEmbodiment(
        _arm(driver, **overrides),
        kinematics,
        side=side,
        depth_camera="top_cam",
        extrinsics=extrinsics,
        log=notes.append,
    )
    return SimpleNamespace(embodiment=wrapped, driver=driver, kinematics=kinematics, notes=notes)


@pytest.fixture
def rig():
    built = _build()
    yield built
    built.embodiment.close()


# --- single-arm profile -------------------------------------------------------------


@pytest.mark.parametrize("side", ["left", "right"])
def test_info_presents_one_arm_to_the_capx_profile(side):
    built = _build(side)
    info = built.embodiment.info
    active = arm_slice(side)
    full = built.embodiment._arm.info.action_space

    assert info.name == f"yam_{side}_capx"
    assert info.action_space.shape == (7,)
    assert info.action_space.semantics.dim_labels == (
        *(f"{side}_j{index}" for index in range(6)),
        "gripper",
    )
    assert info.action_space.semantics.control_mode == "joint_pos"
    assert np.array_equal(info.action_space.low, full.low[active])
    assert np.array_equal(info.action_space.high, full.high[active])
    assert info.action_space.semantics.max_step == tuple(full.semantics.max_step[active])
    assert info.observation_space.state_keys == {
        "joint_pos",
        BIMANUAL_STATE_KEY,
        "eef_pos",
        "eef_quat_wxyz",
    }
    assert info.observation_space.camera_names == {"top_cam", "left_cam", "right_cam"}
    assert info.control_hz == 10.0
    docs = info.docs
    assert docs.startswith(f"YAM {side} arm for code-as-policy")
    assert "CLOSED-LOOP MODE" in docs and "belong to camera 'top_cam'" in docs
    assert "segment(..., camera='left_cam')" in docs
    assert "Two identical 6-DoF arms" in docs  # the upstream joint guide stays appended
    built.embodiment.close()


def test_reset_projects_state_and_publishes_depth(rig):
    observation = rig.embodiment.reset(Scene(id="s", instruction="pick"))

    assert np.array_equal(observation.state["joint_pos"], HOME[:7])
    assert np.array_equal(observation.state[BIMANUAL_STATE_KEY], HOME)
    assert np.allclose(observation.state["eef_pos"], HOME[:3])
    assert np.allclose(observation.state["eef_quat_wxyz"], [1.0, 0.0, 0.0, 0.0])
    depth = observation.extra["depth"]()
    assert depth.dtype == np.float32 and depth.shape == IMAGE
    assert np.array_equal(observation.extra["intrinsics"](), INTRINSICS)
    assert "extrinsics" not in observation.extra
    assert observation.extra["ik_solver"] == rig.embodiment.solve_ik
    assert np.array_equal(rig.embodiment._hold, HOME[7:])


def test_calibrated_mode_publishes_extrinsics():
    built = _build(extrinsics=EXTRINSICS)
    observation = built.embodiment.reset(Scene(id="s", instruction="pick"))
    assert np.array_equal(observation.extra["extrinsics"], EXTRINSICS)
    assert "camera-to-base" in built.embodiment.info.docs
    built.embodiment.close()


@pytest.mark.parametrize("side", ["left", "right"])
def test_step_drives_one_arm_and_holds_the_other(side):
    built = _build(side)
    built.embodiment.reset(Scene(id="s", instruction="pick"))
    action = np.array([0.1, 0.5, 0.4, -0.2, 0.0, 0.3, 0.0])
    active = arm_slice(side)
    idle = arm_slice("right" if side == "left" else "left")

    result = built.embodiment.step(Action(data=action, meta={"code": "x"}))

    command = built.driver.commands[-1]
    assert np.array_equal(command[active], action)
    assert np.array_equal(command[idle], HOME[idle])
    assert np.array_equal(result.observation.state["joint_pos"], action)
    assert np.allclose(result.observation.state["eef_pos"], action[:3])
    assert result.observation.state[BIMANUAL_STATE_KEY].shape == (14,)
    built.embodiment.close()


def test_step_before_reset_and_bad_shapes_raise(rig):
    with pytest.raises(RuntimeError, match="before reset"):
        rig.embodiment.step(Action(data=np.zeros(7), meta={}))
    rig.embodiment.reset(Scene(id="s", instruction="pick"))
    with pytest.raises(ValueError, match="7-D single-arm action"):
        rig.embodiment.step(Action(data=np.zeros(14), meta={}))
    with pytest.raises(ValueError, match="joint observation is invalid"):
        rig.embodiment._project(Observation(state={"joint_pos": np.zeros(13)}))


def test_rejects_eef_mode_delta_mode_and_bad_side():
    with pytest.raises(ValueError, match="control_interface='joints'"):
        YamCapxEmbodiment(_arm(EchoDriver(), control_interface="eef_pos"), FakeKinematics())
    with pytest.raises(ValueError, match="control_interface='joints'"):
        YamCapxEmbodiment(_arm(EchoDriver(), joints_are_delta=True), FakeKinematics())
    with pytest.raises(ValueError, match="side must be one of"):
        YamCapxEmbodiment(_arm(EchoDriver()), FakeKinematics(), side="middle")


# --- IK through the kinematics hook -------------------------------------------------


def test_solve_ik_seeds_the_kinematics_and_checks_limits(rig):
    rig.embodiment.reset(Scene(id="s", instruction="pick"))
    quaternion = quat_wxyz_from_matrix(
        np.array([[np.cos(0.3), -np.sin(0.3), 0], [np.sin(0.3), np.cos(0.3), 0], [0, 0, 1]])
    )
    seed = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    joints = rig.embodiment.solve_ik(np.array([0.3, 0.1, 0.2]), quaternion, seed)

    assert np.allclose(joints, [0.3, 0.1, 0.2, 0.3, 0.4, 0.5])
    recorded_seed, weight, max_iters = rig.kinematics.ik_calls[0]
    assert np.array_equal(recorded_seed, seed) and weight == 0.1 and max_iters == 100
    assert rig.notes == []
    with pytest.raises(ValueError, match="exceeds the configured joint limits"):
        rig.embodiment.solve_ik(np.array([2.5, 0.0, 0.0]), quaternion, seed)
    rig.kinematics.results = [np.full(6, np.nan)]
    with pytest.raises(ValueError, match="YAM IK returned invalid joint values"):
        rig.embodiment.solve_ik(np.zeros(3), quaternion, seed)


# --- observation-borne depth ----------------------------------------------------------


def test_observation_depth_reports_missing_bindings_and_bad_shapes():
    source = ObservationDepth("left_cam")
    with pytest.raises(RuntimeError, match="before the embodiment is reset"):
        source.depth()
    source.bind({"top_cam_depth": np.zeros(IMAGE)})
    with pytest.raises(RuntimeError, match="configure left_depth_serial"):
        source.install()
    source.bind({"left_cam_depth": np.zeros(3), "left_cam_intrinsics": np.eye(2)})
    source.install()
    with pytest.raises(ValueError, match=r"\(H, W\) array"):
        source.depth()
    with pytest.raises(ValueError, match="3x3"):
        source.intrinsics()
    source.bind(
        {"left_cam_depth": np.ones(IMAGE, dtype=np.uint16), "left_cam_intrinsics": INTRINSICS}
    )
    assert source.depth().dtype == np.float32
    matrix = source.intrinsics()
    matrix[0, 0] = 0.0
    assert INTRINSICS[0, 0] == 600.0


# --- guardrails, parking, factory -------------------------------------------------------


class HoldApprover:
    """Two-arm approver that optionally substitutes a safe pose, like YAM's collision gate."""

    def __init__(self) -> None:
        self.seen: list[np.ndarray] = []
        self.hold = False
        self.safe = np.linspace(0.0, 1.3, 14)

    def review(self, action: Action, store: dict[str, Any]) -> Action:
        self.seen.append(np.asarray(action.data, dtype=np.float64).copy())
        if not self.hold:
            store["safe"] = action.data
            return action
        from inspect_robots.approver import DeltaLimitApprover

        DeltaLimitApprover.rewind_reference(store, self.safe)
        return replace(action, data=self.safe.copy(), meta={"collision_blocked": True})


def test_contribute_guardrails_wraps_two_arm_approvers(rig):
    inner = HoldApprover()
    rig.embodiment._arm.contribute_guardrails = lambda space: GuardrailContribution(
        approvers=(("yam-collision", inner),), warnings=("geometry defaulted",)
    )
    contribution = rig.embodiment.contribute_guardrails(rig.embodiment.info.action_space)
    assert [name for name, _ in contribution.approvers] == ["yam-collision"]
    assert contribution.warnings[0] == "geometry defaulted"
    assert "idle arm held" in contribution.warnings[1]
    approver = contribution.approvers[0][1]
    rig.embodiment.reset(Scene(id="s", instruction="pick"))
    store = {_LAST_APPROVED_KEY: np.zeros(7)}
    action = Action(data=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]), meta={})

    assert approver.review(action, store) is action
    assert np.array_equal(inner.seen[0][:7], action.data)
    assert np.array_equal(inner.seen[0][7:], HOME[7:])
    assert "safe" not in store  # the inner approver writes to its private store
    assert np.array_equal(store[_LAST_APPROVED_KEY], np.zeros(7))

    inner.hold = True
    held = approver.review(action, store)
    assert np.array_equal(held.data, inner.safe[:7])
    assert held.meta == {"collision_blocked": True}
    assert np.array_equal(store[_LAST_APPROVED_KEY], inner.safe[:7])


def test_contribute_guardrails_passes_through_warning_only_contributions(rig):
    contribution = rig.embodiment.contribute_guardrails(rig.embodiment.info.action_space)
    assert contribution.approvers == ()
    assert any("disabled by config" in warning for warning in contribution.warnings)
    assert not any("idle arm" in warning for warning in contribution.warnings)


def test_observe_parked_projects_the_final_view(rig):
    rig.embodiment._arm.observe_parked = lambda: None
    assert rig.embodiment.observe_parked() is None
    parked = Observation(images={}, state={"joint_pos": HOME + 0.1})
    rig.embodiment._arm.observe_parked = lambda: parked
    projected = rig.embodiment.observe_parked()
    assert np.allclose(projected.state["joint_pos"], HOME[:7] + 0.1)
    assert np.allclose(projected.state[BIMANUAL_STATE_KEY], HOME + 0.1)


def test_capx_policy_binds_and_moves_one_arm(rig):
    code = (
        "import numpy as np\n"
        "q = solve_ik(np.array([0.3, 0.1, 0.2]), obs['state']['eef_quat_wxyz'])\n"
        "move_to_joints(q)\n"
        "close_gripper()\n"
        "print(obs['depth'].shape, obs['intrinsics'][0, 0], "
        "obs['state']['bimanual_joint_pos'].shape)"
    )
    replies = [{"choices": [{"message": {"role": "assistant", "content": code}}]}]

    def transport(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions", request.url
        return httpx.Response(200, json=replies.pop(0))

    policy = CapxPolicy(
        model="test/model",
        base_url="http://llm.test/v1",
        ik="embodiment",
        camera="top_cam",
        transport=httpx.MockTransport(transport),
        env={},
    )
    try:
        policy.bind(rig.embodiment.info)
        policy.reset(Scene(id="s", instruction="pick"))
        observation = rig.embodiment.reset(Scene(id="s", instruction="pick"))
        chunk = policy.act(observation)
    finally:
        policy.close()

    final = chunk.actions[-1].data
    assert final.shape == (7,)
    assert np.allclose(final[:3], [0.3, 0.1, 0.2])
    assert final[6] == pytest.approx(0.0)  # gripper closed (low)
    assert "(4, 6) 600.0 (14,)" in policy.transcript()[-1]["content"]
    result = rig.embodiment.step(chunk.actions[-1])
    assert np.array_equal(rig.driver.commands[-1][7:], HOME[7:])
    assert np.allclose(result.observation.state["eef_pos"], [0.3, 0.1, 0.2])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"arm": "middle"}, "arm must be one of"),
        ({"depth_camera": "side_cam"}, "depth_camera must be one of"),
        ({"control_interface": "eef_pos"}, "remove control_interface"),
        ({"joints_are_delta": True}, "absolute joint targets"),
        ({"camera_extrinsics": "/nonexistent/x.json"}, "camera_extrinsics"),
        ({"depth_camera": "left_cam"}, "left_depth_serial"),
        ({"no_such_option": 1}, "Invalid YAM configuration"),
    ],
)
def test_factory_config_errors_are_guided(overrides, message):
    with pytest.raises(ConfigError, match=message):
        configured_yam_capx(
            top_depth_serial="TOP-SERIAL",
            left_cam_device="/dev/video1",
            right_cam_device="/dev/video2",
            unattended=True,
            collision_guardrail=False,
            **overrides,
        )


def test_factory_exposes_setup_metadata():
    assert configured_yam_capx.DEVICE_SLOTS == YAMEmbodiment.DEVICE_SLOTS
    assert "eef_orientation" not in {slot.arg for slot in configured_yam_capx.OPTION_SLOTS}
    assert {"i2rt", "mink", "mujoco"} <= set(configured_yam_capx.RUNTIME_REQUIREMENTS)


# --- real kinematics (i2rt model, mink solver) ---------------------------------------------


@needs_kinematics
def test_real_kinematics_matches_the_i2rt_model():
    config = YamConfig()
    kinematics = YamKinematics("LINEAR_4310", joint_low=config.low[:6], joint_high=config.high[:6])
    zero = kinematics.fk(np.zeros(6))
    assert zero[:3, 3] == pytest.approx((0.248, 0.004, 0.172), abs=0.01)
    assert zero[:3, 2] == pytest.approx((1.0, 0.0, 0.0), abs=0.01)  # tool +z along base +x
    home = kinematics.fk(np.asarray(DEFAULT_EEF_HOME_POSE[:6]))
    assert home[:3, 3] == pytest.approx((0.30, 0.0, 0.20), abs=0.005)
    assert kinematics.low[1] == pytest.approx(0.0)  # model range narrower than the +/-pi config
    assert kinematics.high[0] == pytest.approx(3.05433, abs=1e-4)
    with pytest.raises(ValueError, match="do not overlap"):
        YamKinematics("LINEAR_4310", joint_low=np.full(6, 5.0), joint_high=np.full(6, 6.0))
    with pytest.raises(ValueError, match="six finite radians"):
        YamKinematics("LINEAR_4310", joint_low=np.zeros(3), joint_high=np.ones(6))


@pytest.fixture
def real_rig():
    if not HAS_KINEMATICS:
        pytest.skip("i2rt, mink, and mujoco are required for real-kinematics tests")
    config = YamConfig()
    kinematics = YamKinematics("LINEAR_4310", joint_low=config.low[:6], joint_high=config.high[:6])
    notes: list[str] = []
    wrapped = YamCapxEmbodiment(_arm(EchoDriver()), kinematics, log=notes.append)
    yield SimpleNamespace(embodiment=wrapped, kinematics=kinematics, notes=notes)
    wrapped.close()


def test_real_solve_ik_reaches_fk_target_from_a_perturbed_seed(real_rig):
    goal = np.asarray(DEFAULT_EEF_HOME_POSE[:6]) + np.array([0.2, 0.1, -0.1, 0.1, 0.1, 0.3])
    target = real_rig.kinematics.fk(goal)
    seed = np.asarray(DEFAULT_EEF_HOME_POSE[:6])

    joints = real_rig.embodiment.solve_ik(
        target[:3, 3], quat_wxyz_from_matrix(target[:3, :3]), seed
    )

    achieved = real_rig.kinematics.fk(joints)
    assert np.linalg.norm(achieved[:3, 3] - target[:3, 3]) < 1e-3
    assert real_rig.notes == []
    assert np.all(joints >= real_rig.kinematics.low) and np.all(joints <= real_rig.kinematics.high)


def test_real_straight_down_grasp_pose_is_reachable(real_rig):
    seed = np.asarray(DEFAULT_EEF_HOME_POSE[:6])
    joints = real_rig.embodiment.solve_ik(np.array([0.30, 0.0, 0.10]), [0.0, 0.0, 1.0, 0.0], seed)
    achieved = real_rig.kinematics.fk(joints)
    assert achieved[:3, 3] == pytest.approx((0.30, 0.0, 0.10), abs=0.005)
    assert achieved[:3, 2] == pytest.approx((0.0, 0.0, -1.0), abs=0.02)  # approach axis down
    assert real_rig.notes == []


def test_real_unreachable_orientation_relaxes_to_position_with_a_note(real_rig):
    # Jaws rolled a half turn about the approach axis exceed the wrist roll range,
    # so the position is still reached and the orientation residual is reported.
    seed = np.asarray(DEFAULT_EEF_HOME_POSE[:6])
    joints = real_rig.embodiment.solve_ik(np.array([0.30, 0.0, 0.10]), [0.0, 1.0, 0.0, 0.0], seed)
    achieved = real_rig.kinematics.fk(joints)
    assert achieved[:3, 3] == pytest.approx((0.30, 0.0, 0.10), abs=0.005)
    assert len(real_rig.notes) == 1 and "orientation residual" in real_rig.notes[0]


def test_real_solve_ik_rejects_unreachable_targets(real_rig):
    seed = np.asarray(DEFAULT_EEF_HOME_POSE[:6])
    with pytest.raises(ValueError, match="YAM left arm IK cannot reach"):
        real_rig.embodiment.solve_ik(np.array([1.5, 0.0, 0.1]), [1.0, 0.0, 0.0, 0.0], seed)


@needs_kinematics
@pytest.mark.parametrize("side", ["left", "right"])
def test_factory_builds_a_single_arm_embodiment(side, tmp_path):
    path = tmp_path / "extrinsics.json"
    path.write_text(json.dumps(EXTRINSICS.tolist()))
    wrapped = configured_yam_capx(
        arm=side,
        top_depth_serial="TOP-SERIAL",
        left_cam_device="/dev/video1",
        right_cam_device="/dev/video2",
        unattended=True,
        collision_guardrail=False,
        camera_extrinsics=str(path),
        ik_position_tolerance_m=0.004,
    )
    assert wrapped.info.name == f"yam_{side}_capx"
    assert wrapped.info.action_space.shape == (7,)
    assert wrapped._arm._cfg.top_depth_serial == "TOP-SERIAL"
    assert np.array_equal(wrapped._extrinsics, EXTRINSICS)
    assert wrapped._ik_position_tolerance_m == pytest.approx(0.004)
    assert wrapped.info.action_space.semantics.dim_labels[-1] == "gripper"
