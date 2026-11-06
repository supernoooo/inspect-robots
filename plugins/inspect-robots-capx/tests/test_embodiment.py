"""CapxArmEmbodiment base: extrinsics, quaternions, IK relaxation, lifecycle, docs."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest

from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.scene import Scene
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)
from inspect_robots.types import Action, Observation, StepResult
from inspect_robots_capx import CapxArmEmbodiment, CapxPolicy, load_extrinsics
from inspect_robots_capx.embodiment import (
    ORIENTATION_NOTE_DEG,
    pose_from_quat_wxyz,
    quat_wxyz_from_matrix,
    rotation_angle_deg,
)

EXTRINSICS = np.array(
    [
        [0.0, -1.0, 0.0, 0.30],
        [-1.0, 0.0, 0.0, 0.00],
        [0.0, 0.0, -1.0, 0.60],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
START = np.array([0.1, 0.2, 0.3, 0.0, 1.0])
LOW = np.full(4, -1.0)
HIGH = np.full(4, 1.0)
IMAGE = np.zeros((2, 3, 3), dtype=np.uint8)


class FakeArm:
    """Four arm joints plus a gripper; measured state echoes the last command."""

    def __init__(self, cameras: tuple[str, ...] = ("front", "wrist")) -> None:
        self.state = START.copy()
        self.commands: list[np.ndarray] = []
        self.closed = False
        self.resets = 0
        self.info = EmbodimentInfo(
            name="fake",
            action_space=Box(
                shape=(5,),
                low=np.r_[LOW, 0.0],
                high=np.r_[HIGH, 1.0],
                semantics=ActionSemantics(
                    "joint_pos",
                    gripper="continuous",
                    dim_labels=("j0", "j1", "j2", "j3", "gripper"),
                ),
            ),
            observation_space=ObservationSpace(
                cameras=tuple(CameraSpec(name, 2, 3) for name in cameras),
                state=StateSpec(fields=(StateField("joint_pos", (5,)),)),
            ),
            control_hz=10.0,
            docs="Fake arm notes.",
        )

    def _observation(self) -> Observation:
        images = {camera.name: IMAGE for camera in self.info.observation_space.cameras}
        return Observation(images=images, state={"joint_pos": self.state.copy()}, extra={"tag": 1})

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        self.resets += 1
        return self._observation()

    def step(self, action: Action) -> StepResult:
        self.state = np.asarray(action.data, dtype=np.float64).copy()
        self.commands.append(self.state.copy())
        return StepResult(observation=self._observation(), terminated=False)

    def close(self) -> None:
        self.closed = True

    def connect_operator_session(self, session: Any) -> str:
        return f"connected {session}"


class FakeDepth:
    def __init__(self) -> None:
        self.installs = 0

    def install(self) -> None:
        self.installs += 1

    def depth(self) -> np.ndarray:
        return np.full((2, 3), 0.5, dtype=np.float32)

    def intrinsics(self) -> np.ndarray:
        return np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]])


class LinearArm(CapxArmEmbodiment):
    """Tool position = joints[:3] (meters), tool yaw = joint 3: IK is exact.

    ``ik_results`` optionally scripts ``(joints, error)`` per solver pass so
    the relaxation loop is observable; ``weights`` records the passes.
    """

    def __init__(self, arm: FakeArm, **kwargs: Any) -> None:
        self.ik_results: list[tuple[np.ndarray, float]] = []
        self.weights: list[float] = []
        self.notes: list[str] = []
        kwargs.setdefault("depth_source", FakeDepth())
        kwargs.setdefault("depth_camera", "front")
        kwargs.setdefault("log", self.notes.append)
        super().__init__(
            arm,
            name="linear_capx",
            label="Linear",
            arm_dof=4,
            joint_low=kwargs.pop("joint_low", LOW),
            joint_high=kwargs.pop("joint_high", HIGH),
            **kwargs,
        )

    def _kinematics_docs(self) -> str:
        return "Linear arm: joints are meters. "

    def _fk(self, joints: np.ndarray) -> np.ndarray:
        pose = np.eye(4)
        yaw = float(joints[3])
        pose[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
        pose[:3, 3] = joints[:3]
        return pose

    def _iterate_ik(
        self, seed: np.ndarray, target: np.ndarray, orientation_weight: float
    ) -> tuple[np.ndarray, np.ndarray, float]:
        self.weights.append(orientation_weight)
        if self.ik_results:
            joints, error = self.ik_results.pop(0)
            if not np.all(np.isfinite(joints)):
                raise ValueError("Linear IK returned invalid joint values")
            return joints, self._fk(joints), error
        joints = np.r_[target[:3, 3], np.arctan2(target[1, 0], target[0, 0])]
        return joints, self._fk(joints), 0.0


@pytest.fixture
def rig() -> SimpleNamespace:
    arm = FakeArm()
    depth = FakeDepth()
    wrapped = LinearArm(arm, depth_source=depth, extrinsics=EXTRINSICS)
    return SimpleNamespace(arm=arm, depth=depth, embodiment=wrapped)


# --- extrinsics and rotation helpers --------------------------------------------------


def test_extrinsics_accept_json_list_dict_and_npy(tmp_path):
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps(EXTRINSICS.tolist()))
    keyed = tmp_path / "keyed.json"
    keyed.write_text(json.dumps({"camera_to_base": EXTRINSICS.tolist(), "rms_m": 0.002}))
    array = tmp_path / "pose.npy"
    np.save(array, EXTRINSICS)
    for path in (bare, keyed, array):
        assert np.array_equal(load_extrinsics(path), EXTRINSICS)


@pytest.mark.parametrize(
    "matrix",
    [
        np.eye(3).tolist(),
        (np.diag([2.0, 1.0, 1.0, 1.0])).tolist(),
        np.diag([-1.0, 1.0, 1.0, 1.0]).tolist(),
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]],
        [[float("nan")] * 4] * 4,
    ],
    ids=["shape", "scaled", "reflection", "bottom_row", "non_finite"],
)
def test_extrinsics_reject_non_rigid_transforms(tmp_path, matrix):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(matrix))
    with pytest.raises(ValueError, match="camera_extrinsics"):
        load_extrinsics(path)


def test_extrinsics_missing_file_is_named(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"missing\.json"):
        load_extrinsics(tmp_path / "missing.json")


def _rotation(axis: int, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@pytest.mark.parametrize(
    "rotation",
    [
        np.eye(3),
        _rotation(0, np.pi),
        _rotation(1, np.pi),
        _rotation(2, np.pi),
        _rotation(0, 0.4) @ _rotation(2, -1.3),
        _rotation(1, 2.9) @ _rotation(0, 0.1),
    ],
    ids=["identity", "pi_x", "pi_y", "pi_z", "generic", "near_pi_y"],
)
def test_quaternion_round_trips_every_branch(rotation):
    quaternion = quat_wxyz_from_matrix(rotation)
    assert np.linalg.norm(quaternion) == pytest.approx(1.0)
    assert quaternion[0] >= 0.0
    assert np.allclose(pose_from_quat_wxyz(np.zeros(3), quaternion)[:3, :3], rotation, atol=1e-9)
    assert rotation_angle_deg(rotation, rotation) == pytest.approx(0.0, abs=1e-6)


def test_quaternion_helpers_reject_bad_inputs():
    with pytest.raises(ValueError, match="3x3"):
        quat_wxyz_from_matrix(np.eye(4))
    with pytest.raises(ValueError, match="unit quaternion"):
        pose_from_quat_wxyz(np.zeros(3), np.array([0.0, 0.0, 2.0, 0.0]))
    assert rotation_angle_deg(np.eye(3), _rotation(2, np.pi / 2)) == pytest.approx(90.0)


# --- construction and info ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"depth_camera": "top"}, "not a configured camera"),
        ({"ik_position_tolerance_m": 0}, "ik_position_tolerance_m"),
        ({"ik_orientation_weight": float("inf")}, "ik_orientation_weight"),
        ({"ik_orientation_weight": True}, "ik_orientation_weight"),
        ({"joint_low": np.zeros(4), "joint_high": np.zeros(4)}, "limits are invalid"),
        ({"joint_low": np.zeros(2)}, "limits are invalid"),
    ],
)
def test_constructor_validates_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        LinearArm(FakeArm(), **kwargs)


def test_info_adds_tool_state_and_docs_in_calibrated_mode(rig):
    info = rig.embodiment.info
    assert info.name == "linear_capx"
    assert info.action_space is rig.arm.info.action_space
    assert {"joint_pos", "eef_pos", "eef_quat_wxyz"} == set(info.observation_space.state_keys)
    assert {field.key for field in info.observation_space.state.fields} == {
        "joint_pos",
        "eef_pos",
        "eef_quat_wxyz",
    }
    docs = info.docs
    assert docs.startswith("Linear arm: joints are meters. ")
    assert "camera-to-base" in docs and docs.endswith("Fake arm notes.")
    assert "CLOSED-LOOP MODE" not in docs
    assert rig.embodiment.arm_dof == 4


def test_closed_loop_docs_name_the_other_camera_when_present():
    with_wrist = LinearArm(FakeArm())
    assert "CLOSED-LOOP MODE" in with_wrist.info.docs
    assert "segment(..., camera='wrist')" in with_wrist.info.docs
    assert "belong to camera 'front'" in with_wrist.info.docs
    front_only = LinearArm(FakeArm(cameras=("front",)))
    assert "segment(..., camera=" not in front_only.info.docs
    assert "confirm alignment; after lifting" in front_only.info.docs


# --- lifecycle -------------------------------------------------------------------------


def test_reset_installs_depth_and_augments_observation(rig):
    observation = rig.embodiment.reset(Scene(id="s", instruction="pick"))

    assert rig.arm.resets == 1 and rig.depth.installs == 1
    assert np.allclose(observation.state["joint_pos"], START)
    assert np.allclose(observation.state["eef_pos"], START[:3])
    expected = quat_wxyz_from_matrix(_rotation(2, START[3]))
    assert np.allclose(observation.state["eef_quat_wxyz"], expected)
    assert observation.extra["tag"] == 1
    assert observation.extra["depth"]().dtype == np.float32
    assert observation.extra["intrinsics"]()[0, 0] == 600.0
    assert observation.extra["ik_solver"] == rig.embodiment.solve_ik
    assert np.allclose(observation.extra["eef_rotation"], _rotation(2, START[3]))
    assert np.array_equal(observation.extra["extrinsics"], EXTRINSICS)
    observation.extra["extrinsics"][0, 0] = 9.0
    assert rig.embodiment._extrinsics[0, 0] == 0.0


def test_step_forwards_action_and_augments_result(rig):
    rig.embodiment.reset(Scene(id="s", instruction="pick"))
    target = np.array([0.4, -0.2, 0.1, 0.3, 0.0])

    result = rig.embodiment.step(Action(data=target, meta={}))

    assert np.allclose(rig.arm.commands[-1], target)
    assert np.allclose(result.observation.state["eef_pos"], target[:3])
    assert "ik_solver" in result.observation.extra


def test_closed_loop_mode_omits_extrinsics():
    wrapped = LinearArm(FakeArm())
    observation = wrapped.reset(Scene(id="s", instruction="pick"))
    assert "extrinsics" not in observation.extra


def test_close_and_attribute_passthrough(rig):
    assert rig.embodiment.connect_operator_session("s") == "connected s"
    with pytest.raises(AttributeError):
        _ = rig.embodiment.no_such_hook
    rig.embodiment.close()
    assert rig.arm.closed


def test_invalid_joint_observation_is_rejected(rig):
    rig.arm.state = np.array([0.0, np.nan, 0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="Linear joint observation is invalid"):
        rig.embodiment.reset(Scene(id="s", instruction="pick"))
    rig.arm.state = np.zeros(3)
    with pytest.raises(ValueError, match="joint observation is invalid"):
        rig.embodiment.reset(Scene(id="s", instruction="pick"))


# --- solve_ik ---------------------------------------------------------------------------


def test_solve_ik_returns_exact_solution_without_notes(rig):
    target = np.array([0.5, -0.3, 0.2])
    quaternion = quat_wxyz_from_matrix(_rotation(2, 0.5))

    joints = rig.embodiment.solve_ik(target, quaternion, START[:4])

    assert np.allclose(joints, [*target, 0.5])
    assert rig.embodiment.weights == [0.1]
    assert rig.embodiment.notes == []


def test_solve_ik_relaxes_orientation_until_the_position_fits(rig):
    far = np.array([0.5, 0.5, 0.5, 0.0])
    near = np.array([0.5, -0.3, 0.2, 0.0])
    rig.embodiment.ik_results = [(far, 0.02), (far, 0.01), (near, 0.001)]

    joints = rig.embodiment.solve_ik(np.array([0.5, -0.3, 0.2]), [1.0, 0, 0, 0], START[:4])

    assert np.allclose(joints, near)
    assert rig.embodiment.weights == pytest.approx([0.1, 0.01, 0.001])


def test_solve_ik_reports_unreachable_positions(rig):
    rig.embodiment.ik_results = [(START[:4], 0.02)] * 4
    with pytest.raises(ValueError, match=r"Linear IK cannot reach position .* residual 20\.0 mm"):
        rig.embodiment.solve_ik(np.array([0.5, 0.5, 0.5]), [1.0, 0, 0, 0], START[:4])
    assert rig.embodiment.weights == pytest.approx([0.1, 0.01, 0.001, 0.0])


def test_solve_ik_rejects_bad_inputs_and_limit_violations(rig):
    with pytest.raises(ValueError, match=r"finite position \(3,\), quaternion \(4,\), seed \(4,\)"):
        rig.embodiment.solve_ik(np.zeros(2), [1.0, 0, 0, 0], START[:4])
    with pytest.raises(ValueError, match="seed"):
        rig.embodiment.solve_ik(np.zeros(3), [1.0, 0, 0, 0], np.zeros(2))
    with pytest.raises(ValueError, match="unit quaternion"):
        rig.embodiment.solve_ik(np.zeros(3), [0.0, 0, 2.0, 0], START[:4])
    with pytest.raises(ValueError, match="Linear IK solution exceeds the configured joint limits"):
        rig.embodiment.solve_ik(np.array([1.5, 0.0, 0.0]), [1.0, 0, 0, 0], START[:4])
    rig.embodiment.ik_results = [(np.full(4, np.nan), 0.0)]
    with pytest.raises(ValueError, match="invalid joint values"):
        rig.embodiment.solve_ik(np.zeros(3), [1.0, 0, 0, 0], START[:4])


def test_solve_ik_notes_large_orientation_residuals(rig):
    # The fake arm only yaws, so a 90 degree roll cannot be met.
    sideways = quat_wxyz_from_matrix(_rotation(0, np.pi / 2))

    rig.embodiment.solve_ik(np.array([0.2, 0.1, 0.0]), sideways, START[:4])

    assert len(rig.embodiment.notes) == 1
    note = rig.embodiment.notes[0]
    assert note.startswith("solve_ik note: orientation residual 90 deg")
    assert "achieved quaternion_wxyz" in note
    assert ORIENTATION_NOTE_DEG < 90


def test_custom_limits_label_appears_in_errors():
    wrapped = LinearArm(FakeArm(), limits_label="calibrated joint limits")
    with pytest.raises(ValueError, match="exceeds the calibrated joint limits"):
        wrapped.solve_ik(np.array([2.0, 0.0, 0.0]), [1.0, 0, 0, 0], START[:4])


# --- capx policy end to end ---------------------------------------------------------------


def test_capx_policy_binds_and_moves_through_embodiment_ik(rig):
    code = (
        "import numpy as np\n"
        "q = solve_ik(np.array([0.3, 0.1, 0.2]), obs['state']['eef_quat_wxyz'])\n"
        "move_to_joints(q)\n"
        "close_gripper()\n"
        "print(obs['depth'].shape, obs['intrinsics'][0, 0], obs['extrinsics'][0, 3])"
    )
    replies = [{"choices": [{"message": {"role": "assistant", "content": code}}]}]

    def transport(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions", request.url
        return httpx.Response(200, json=replies.pop(0))

    policy = CapxPolicy(
        model="test/model",
        base_url="http://llm.test/v1",
        ik="embodiment",
        camera="front",
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
    assert np.allclose(final[:4], [0.3, 0.1, 0.2, 0.0])
    assert final[4] == pytest.approx(0.0)
    assert "(2, 3) 600.0 0.3" in policy.transcript()[-1]["content"]
    assert "Linear arm: joints are meters." in policy.transcript()[0]["content"]


def test_subclass_hooks_can_reshape_actions_and_observations():
    class Doubling(LinearArm):
        def _expand(self, action: Action) -> Action:
            return replace(action, data=np.asarray(action.data) * 2.0)

        def _project(self, observation: Observation) -> Observation:
            state = {
                **observation.state,
                "joint_pos": np.asarray(observation.state["joint_pos"]) / 2.0,
            }
            return replace(observation, state=state)

    arm = FakeArm()
    wrapped = Doubling(arm)
    wrapped.reset(Scene(id="s", instruction="pick"))
    result = wrapped.step(Action(data=np.array([0.1, 0.2, 0.3, 0.0, 0.5]), meta={}))
    assert np.allclose(arm.commands[-1], [0.2, 0.4, 0.6, 0.0, 1.0])
    assert np.allclose(result.observation.state["joint_pos"], [0.1, 0.2, 0.3, 0.0, 0.5])
    assert np.allclose(result.observation.state["eef_pos"], [0.1, 0.2, 0.3])
