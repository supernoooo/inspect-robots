"""Offline SO-101 CaP-X adapter tests; motors, cameras and the LLM are all fakes."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest
from _run_so101 import configured_so101_capx
from _so101_capx import (
    RealSenseDepth,
    SO101CapxEmbodiment,
    load_extrinsics,
    load_kinematics,
)
from _so101_motion import CLIArm
from inspect_robots_capx import CapxPolicy
from inspect_robots_so101 import SOArmConfig, packing
from lerobot.utils.rotation import Rotation

from inspect_robots import Action, Scene
from inspect_robots.errors import ConfigError


@pytest.fixture(autouse=True)
def _realsense_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test a placeholder RealSense serial; no real device is opened."""
    monkeypatch.setenv("SO101_HEAD_CAMERA_SERIAL", "TEST-SERIAL")


MODEL = Path(__file__).resolve().parent.parent / ".so101/models/SO101/so101_new_calib.urdf"
START = np.array([0.0, -50.0, 60.0, 20.0, 0.0, 50.0])
LOW = (-110.0, -105.0, -100.0, -100.0, -160.0, 0.0)
HIGH = (110.0, 105.0, 100.0, 100.0, 160.0, 100.0)
CAMERA = (480, 640, 3)
EXTRINSICS = np.array(
    [
        [0.0, -1.0, 0.0, 0.30],
        [-1.0, 0.0, 0.0, 0.00],
        [0.0, 0.0, -1.0, 0.60],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

needs_model = pytest.mark.skipif(
    not MODEL.is_file(), reason="run _so101_instructions/_fetch_so101_model.py first"
)


class FakeDriver:
    def __init__(self, pose: np.ndarray):
        self.pose = pose.copy()
        self.commands: list[np.ndarray] = []

    def get_observation(self):
        return {**packing.to_action_dict(self.pose), "front": np.zeros(CAMERA, dtype=np.uint8)}

    def send_action(self, action):
        self.pose = packing.from_obs_dict(action)
        self.commands.append(self.pose.copy())
        return packing.to_action_dict(self.pose)

    def disconnect(self):
        pass


class FakeDepth:
    def __init__(self):
        self.installs = 0

    def install(self):
        self.installs += 1

    def depth(self):
        return np.full(CAMERA[:2], 0.5, dtype=np.float32)

    def intrinsics(self):
        return np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]])


def _arm(driver: FakeDriver) -> CLIArm:
    config = SOArmConfig(
        cameras=("front",),
        camera_configs={},
        control_hz=10.0,
        settle_tolerance=None,
        joint_low=LOW,
        joint_high=HIGH,
    )
    return CLIArm(
        config,
        capture_home=False,
        return_home=True,
        driver_factory=lambda _: driver,
        operator=SimpleNamespace(wait_ready=lambda *args: None),
        poll_end=lambda: False,
        sleep_fn=lambda _: None,
    )


@pytest.fixture
def rig():
    if not MODEL.is_file():
        pytest.skip("run _so101_instructions/_fetch_so101_model.py first")
    driver = FakeDriver(START)
    depth = FakeDepth()
    notes: list[str] = []
    arm = _arm(driver)
    wrapped = SO101CapxEmbodiment(
        arm,
        load_kinematics(MODEL, joint_low=LOW, joint_high=HIGH),
        depth_camera="front",
        depth_source=depth,
        extrinsics=EXTRINSICS,
        log=notes.append,
    )
    yield SimpleNamespace(embodiment=wrapped, driver=driver, depth=depth, notes=notes)
    wrapped.close()


# --- extrinsics -----------------------------------------------------------------


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


# --- IK ---------------------------------------------------------------------------


@needs_model
def test_solve_ik_reaches_fk_target_from_a_distant_seed(rig):
    kinematics = rig.embodiment._kinematics
    goal = np.array([20.0, -20.0, 30.0, 50.0, 10.0])
    target = kinematics.forward_kinematics(goal)
    xyzw = Rotation.from_matrix(target[:3, :3]).as_quat()

    joints = rig.embodiment.solve_ik(target[:3, 3], np.r_[xyzw[3], xyzw[:3]], START[:5])

    achieved = kinematics.forward_kinematics(joints)
    assert np.linalg.norm(achieved[:3, 3] - target[:3, 3]) < 1e-3
    assert rig.notes == []


@needs_model
def test_solve_ik_rejects_unreachable_positions_and_bad_inputs(rig):
    with pytest.raises(ValueError, match="cannot reach"):
        rig.embodiment.solve_ik(np.array([1.5, 0.0, 0.1]), np.array([0, 0, 1.0, 0]), START[:5])
    with pytest.raises(ValueError, match="unit quaternion"):
        rig.embodiment.solve_ik(np.array([0.2, 0, 0.1]), np.array([0, 0, 2.0, 0]), START[:5])
    with pytest.raises(ValueError, match="finite position"):
        rig.embodiment.solve_ik(np.zeros(2), np.array([1.0, 0, 0, 0]), START[:5])
    assert rig.driver.commands == []


@needs_model
def test_solve_ik_notes_large_orientation_residuals(rig):
    # Straight down is reachable here, but rolling 90 degrees about base x is not.
    position = rig.embodiment._kinematics.forward_kinematics(START[:5])[:3, 3]
    sideways = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])

    rig.embodiment.solve_ik(position, sideways, START[:5])

    assert rig.notes and "orientation residual" in rig.notes[0]


@needs_model
def test_solve_ik_rejects_solutions_outside_calibrated_limits(rig):
    narrow_low = (-5.0, -55.0, 55.0, 15.0, -5.0, 0.0)
    narrow_high = (5.0, -45.0, 65.0, 25.0, 5.0, 100.0)
    wrapped = rig.embodiment
    wrapped._low = np.asarray(narrow_low[:5])
    wrapped._high = np.asarray(narrow_high[:5])
    far = wrapped._kinematics.forward_kinematics(np.array([40.0, -20.0, 30.0, 50.0, 0.0]))
    with pytest.raises(ValueError, match="calibrated joint limits"):
        wrapped.solve_ik(far[:3, 3], np.array([1.0, 0, 0, 0]), START[:5])


@needs_model
def test_solve_ik_rejects_invalid_solver_output(rig):
    rig.embodiment._kinematics = SimpleNamespace(
        inverse_kinematics=lambda *a, **k: np.full(6, np.nan),
        forward_kinematics=lambda q: np.eye(4),
    )
    with pytest.raises(ValueError, match="invalid joint values"):
        rig.embodiment.solve_ik(np.zeros(3), np.array([1.0, 0, 0, 0]), START[:5])


# --- embodiment lifecycle ---------------------------------------------------------


@needs_model
def test_reset_installs_depth_and_publishes_capx_extras_and_eef_state(rig):
    observation = rig.embodiment.reset(Scene(id="s", instruction="pick"))

    assert rig.depth.installs == 1
    assert set(observation.extra) >= {
        "depth",
        "intrinsics",
        "extrinsics",
        "ik_solver",
        "eef_rotation",
    }
    pose_now = rig.embodiment._kinematics.forward_kinematics(START[:5])
    assert np.allclose(observation.extra["eef_rotation"], pose_now[:3, :3])
    assert "CLOSED-LOOP MODE" not in (rig.embodiment.info.docs or "")
    assert observation.extra["depth"]().dtype == np.float32
    assert np.array_equal(observation.extra["extrinsics"], EXTRINSICS)
    pose = rig.embodiment._kinematics.forward_kinematics(START[:5])
    assert np.allclose(observation.state["eef_pos"], pose[:3, 3])
    assert np.linalg.norm(observation.state["eef_quat_wxyz"]) == pytest.approx(1.0)
    info = rig.embodiment.info
    assert info.name == "so101_capx"
    assert {"joint_pos", "eef_pos", "eef_quat_wxyz"} <= info.observation_space.state_keys
    assert "camera-to-base" in (info.docs or "")


@needs_model
def test_step_sends_joint_target_unchanged(rig):
    rig.embodiment.reset(Scene(id="s", instruction="pick"))
    target = START + np.array([1.0, -1.0, 0.5, 0.0, 0.0, 2.0])

    result = rig.embodiment.step(Action(data=target, meta={}))

    assert np.allclose(rig.driver.commands[-1], target)
    assert "ik_solver" in result.observation.extra


@needs_model
def test_missing_extrinsics_is_explained_to_the_model():
    wrapped = SO101CapxEmbodiment(
        _arm(FakeDriver(START)),
        load_kinematics(MODEL, joint_low=LOW, joint_high=HIGH),
        depth_camera="front",
        depth_source=FakeDepth(),
    )
    try:
        observation = wrapped.reset(Scene(id="s", instruction="pick"))
        assert "extrinsics" not in observation.extra
        docs = wrapped.info.docs or ""
        assert "CLOSED-LOOP MODE" in docs and "probe" in docs and "eye-in-hand" in docs
        assert "camera='wrist'" not in docs  # this rig fixture has only the front camera
    finally:
        wrapped.close()


@needs_model
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"depth_camera": "wrist"}, "not a configured camera"),
        ({"ik_position_tolerance_m": 0}, "ik_position_tolerance_m"),
        ({"ik_orientation_weight": float("inf")}, "ik_orientation_weight"),
    ],
)
def test_constructor_validates_options(kwargs, message):
    options: dict[str, Any] = {"depth_camera": "front", "depth_source": FakeDepth(), **kwargs}
    with pytest.raises(ValueError, match=message):
        SO101CapxEmbodiment(_arm(FakeDriver(START)), load_kinematics(MODEL), **options)


# --- capx policy end to end -------------------------------------------------------


@needs_model
def test_capx_policy_binds_and_moves_through_embodiment_ik(rig):
    kinematics = rig.embodiment._kinematics
    goal = START[:5] + np.array([10.0, 5.0, -5.0, 5.0, 0.0])
    target = kinematics.forward_kinematics(goal)[:3, 3].round(4).tolist()
    code = (
        "import numpy as np\n"
        f"q = solve_ik(np.array({target}), obs['state']['eef_quat_wxyz'])\n"
        "move_to_joints(q)\n"
        "close_gripper()\n"
        "print(obs['depth'].shape, obs['intrinsics'][0, 0])"
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
    assert np.linalg.norm(kinematics.forward_kinematics(final[:5])[:3, 3] - target) < 0.005
    assert final[5] == pytest.approx(LOW[5])
    report = policy.transcript()[-1]["content"]
    assert "(480, 640) 600.0" in report


# --- RealSense depth ----------------------------------------------------------------


def _camera(**overrides):
    intrinsics = SimpleNamespace(fx=610.0, fy=611.0, ppx=321.0, ppy=239.0)
    stream = SimpleNamespace(
        as_video_stream_profile=lambda: SimpleNamespace(get_intrinsics=lambda: intrinsics)
    )
    sensor = SimpleNamespace(get_depth_scale=lambda: 0.001)
    profile = SimpleNamespace(
        get_device=lambda: SimpleNamespace(first_depth_sensor=lambda: sensor),
        get_stream=lambda _: stream,
    )
    values = {
        "use_depth": True,
        "rotation": None,
        "rs_profile": profile,
        "_read_from_hardware": lambda: "frames",
        "read_depth": lambda timeout_ms: np.full((2, 3), 500, dtype=np.uint16),
    }
    return SimpleNamespace(**{**values, **overrides})


def test_realsense_depth_aligns_once_and_converts_to_meters():
    camera = _camera()
    original = camera._read_from_hardware
    source = RealSenseDepth(
        SimpleNamespace(_driver=SimpleNamespace(cameras={"front": camera})), "front"
    )
    with pytest.raises(RuntimeError, match="before the embodiment is reset"):
        source.depth()
    with pytest.raises(RuntimeError, match="before the embodiment is reset"):
        source.intrinsics()

    source.install()
    patched = camera._read_from_hardware
    source.install()

    assert patched is not original and camera._read_from_hardware is patched
    assert np.allclose(source.depth(), 0.5) and source.depth().dtype == np.float32
    assert source.intrinsics().tolist() == [[610.0, 0, 321.0], [0, 611.0, 239.0], [0, 0, 1.0]]


@pytest.mark.parametrize(
    ("cameras", "message"),
    [
        ({}, "not connected"),
        ({"front": _camera(use_depth=False)}, "without a depth stream"),
        ({"front": _camera(rotation=1)}, "unrotated"),
    ],
)
def test_realsense_depth_rejects_unusable_cameras(cameras, message):
    source = RealSenseDepth(SimpleNamespace(_driver=SimpleNamespace(cameras=cameras)), "front")
    with pytest.raises(RuntimeError, match=message):
        source.install()


# --- registered factory -------------------------------------------------------------


@needs_model
def test_factory_without_extrinsics_points_closed_loop_at_the_wrist_camera(tmp_path):
    wrapped = configured_so101_capx(
        robot_id="missing", calibration_dir=str(tmp_path), urdf_path=MODEL
    )
    try:
        assert wrapped._extrinsics is None
        assert "segment(..., camera='wrist')" in (wrapped.info.docs or "")
        assert "obs['depth'] (meters) and obs['intrinsics'] belong to camera 'front'" in (
            wrapped.info.docs or ""
        )
    finally:
        wrapped.close()


@needs_model
def test_factory_enables_head_depth_and_reads_extrinsics(tmp_path):
    path = tmp_path / "extrinsics.json"
    path.write_text(json.dumps(EXTRINSICS.tolist()))
    wrapped = configured_so101_capx(
        robot_id="missing",
        calibration_dir=str(tmp_path),
        urdf_path=MODEL,
        camera_extrinsics=str(path),
        ik_position_tolerance_m=0.004,
    )
    try:
        assert wrapped._arm._cfg.camera_configs["front"].use_depth is True
        assert np.array_equal(wrapped._extrinsics, EXTRINSICS)
        assert wrapped._ik_position_tolerance_m == pytest.approx(0.004)
        assert wrapped.info.action_space.semantics.control_mode == "joint_pos"
        assert wrapped.info.action_space.semantics.dim_labels[-1] == "gripper"
    finally:
        wrapped.close()


@needs_model
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"camera_mode": "wrist"}, "head camera"),
        ({"camera_configs": {}}, "head camera"),
        ({"camera_extrinsics": "/nonexistent/x.json"}, "camera_extrinsics"),
        ({"ik_orientation_weight": 0}, "ik_orientation_weight"),
    ],
)
def test_factory_config_errors_are_guided(tmp_path, overrides, message):
    with pytest.raises(ConfigError, match=message):
        configured_so101_capx(
            robot_id="missing", calibration_dir=str(tmp_path), urdf_path=MODEL, **overrides
        )


def test_head_depth_must_be_boolean(tmp_path):
    from _run_so101 import rig_config

    with pytest.raises(ConfigError, match="head_depth"):
        rig_config(head_depth="yes", robot_id="missing", calibration_dir=str(tmp_path))


# --- URDF kinematics (moved from the EEF adapter tests) ---------------------------


def test_model_joint_order_and_fk_ik_round_trip():
    if not MODEL.is_file():
        pytest.skip("SO-101 model cache missing")
    joints = {joint.get("name"): joint for joint in ET.parse(MODEL).getroot().findall("joint")}
    assert set(packing.MOTORS) <= joints.keys()
    link = "gripper_frame_link"
    chain = []
    while link != "base_link":
        joint = next(j for j in joints.values() if j.find("child").get("link") == link)
        if joint.get("type") == "revolute":
            chain.append(joint.get("name"))
            assert joint.find("axis").get("xyz") == "0 0 1"
            bounds = joint.find("limit")
            assert -3.2 < float(bounds.get("lower")) < float(bounds.get("upper")) < 3.2
        link = joint.find("parent").get("link")
    assert chain[::-1] == list(packing.MOTORS[:5])
    assert packing.from_obs_dict(packing.to_action_dict(START)) == pytest.approx(START)
    model = load_kinematics(MODEL)
    for q in (np.zeros(5), START[:5], np.array([15.0, -40.0, 50.0, 10.0, 0.0])):
        actual = model.forward_kinematics(q)
        target = actual.copy()
        target[2, 3] += 0.002
        solved = model.inverse_kinematics(q, target, orientation_weight=0.01)
        achieved = model.forward_kinematics(solved)
        assert np.linalg.norm(achieved[:3, 3] - target[:3, 3]) < 0.0008
        assert np.linalg.norm(achieved[:3, :3] - target[:3, :3]) < 0.004


def test_ik_uses_calibrated_limits_instead_of_generic_urdf_limits():
    if not MODEL.is_file():
        pytest.skip("SO-101 model cache missing")
    low = (-108.0, -103.38, -97.32, -101.32, -180.0, 0.0)
    high = (108.0, 103.38, 97.32, 101.32, 180.0, 100.0)
    model = load_kinematics(MODEL, joint_low=low, joint_high=high)
    for index, name in enumerate(packing.MOTORS[:5]):
        assert np.rad2deg(model.robot.get_joint_limits(name)) == pytest.approx(
            [low[index], high[index]]
        )


def test_front_camera_requires_a_configured_serial(monkeypatch, tmp_path):
    from _run_so101 import rig_config

    monkeypatch.delenv("SO101_HEAD_CAMERA_SERIAL")
    with pytest.raises(ConfigError, match="head_camera_serial is required"):
        rig_config(robot_id="missing", calibration_dir=str(tmp_path))
    config, _ = rig_config(robot_id="missing", calibration_dir=str(tmp_path), camera_mode="wrist")
    assert set(config.camera_configs) == {"wrist"}
