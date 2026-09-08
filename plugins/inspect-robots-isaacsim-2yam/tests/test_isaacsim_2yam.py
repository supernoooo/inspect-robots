from __future__ import annotations

import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inspect_robots import Action, Embodiment, PolicyConfig, PolicyInfo, Scene
from inspect_robots.compat import check_compatibility
from inspect_robots.policy import Policy
from inspect_robots.spaces import Box
from inspect_robots.types import ActionChunk, Observation
from inspect_robots_isaacsim_2yam import (
    IsaacSim2YamEmbodiment,
    isaacsim_2yam_embodiment,
    put_everything_in_box,
)
from inspect_robots_isaacsim_2yam import embodiment as module
from inspect_robots_isaacsim_2yam.asset import prepare_isaac_mjcf
from inspect_robots_isaacsim_2yam.contract import (
    ACTION_DIM,
    DIM_LABELS,
    action_space,
    observation_space,
    physical_joint_targets,
    wire_state,
)


class _MatchingPolicy:
    def __init__(self) -> None:
        space = action_space()
        self.info = PolicyInfo(
            name="matching-yam",
            action_space=Box(shape=space.shape, semantics=space.semantics),
            observation_space=observation_space(360, 640),
        )
        self.config = PolicyConfig(action_horizon=1)

    def reset(self, scene: Scene) -> None:
        pass

    def act(self, observation: Observation) -> ActionChunk:
        return ActionChunk(actions=[Action(data=np.zeros(ACTION_DIM))])


class _FakeTorch:
    @staticmethod
    def as_tensor(value: Any, device: str | None = None) -> np.ndarray:
        return np.asarray(value)


class _FakeTensor:
    def __init__(self, value: Any) -> None:
        self.value = np.asarray(value)

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.value


class _FakeEnv:
    def __init__(self) -> None:
        self.closed = False
        self.seed: int | None = None
        self.next_terminated = False
        self.next_truncated = False
        self.next_info: Any = {"success": False}

    @staticmethod
    def observation() -> dict[str, dict[str, Any]]:
        rgba = np.zeros((1, 360, 640, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        return {
            "policy": {
                "top_cam": rgba,
                "left_cam": rgba,
                "right_cam": rgba,
                "joint_pos": _FakeTensor(np.zeros((1, 14), dtype=np.float32)),
            }
        }

    def reset(self, seed: int | None = None) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        self.seed = seed
        return self.observation(), {}

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, Any]:
        assert np.asarray(action).shape == (1, 14)
        return (
            self.observation(),
            np.asarray([0.5]),
            np.asarray([self.next_terminated]),
            np.asarray([self.next_truncated]),
            self.next_info,
        )

    def close(self) -> None:
        self.closed = True


def _inject(embodiment: IsaacSim2YamEmbodiment) -> _FakeEnv:
    env = _FakeEnv()
    embodiment._env = env
    embodiment._torch = _FakeTorch()
    return env


def test_contract_matches_molmoact2_yam() -> None:
    space = action_space()
    assert space.shape == (14,)
    assert space.semantics is not None
    assert space.semantics.control_mode == "joint_pos"
    assert space.semantics.gripper == "continuous"
    assert space.semantics.dim_labels == DIM_LABELS
    assert space.low is not None and space.high is not None
    np.testing.assert_array_equal(space.low[[6, 13]], [0.0, 0.0])
    np.testing.assert_array_equal(space.high[[6, 13]], [1.0, 1.0])
    obs_space = observation_space(12, 16)
    assert obs_space.camera_names == frozenset({"top_cam", "left_cam", "right_cam"})
    assert obs_space.state_keys == frozenset({"joint_pos"})


def test_physical_action_and_wire_state_round_trip() -> None:
    action = np.arange(14, dtype=np.float64) / 13
    arms, left_gripper, right_gripper = physical_joint_targets(action)
    np.testing.assert_allclose(arms, np.r_[action[:6], action[7:13]])
    assert left_gripper == pytest.approx(-0.0475 * action[6])
    assert right_gripper == pytest.approx(-0.0475)
    state = wire_state(arms[:6], [left_gripper], arms[6:], [right_gripper])
    np.testing.assert_allclose(state[:13], action[:13])
    assert state[13] == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [np.zeros(13), np.zeros((2, 7)), np.full(14, np.nan)])
def test_physical_action_rejects_invalid_vectors(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        physical_joint_targets(bad)


def test_wire_state_rejects_bad_arm_shapes() -> None:
    with pytest.raises(ValueError, match="six joint"):
        wire_state(np.zeros(5), [0], np.zeros(6), [0])


def test_info_factory_protocol_and_compatibility() -> None:
    embodiment = isaacsim_2yam_embodiment(control_hz=20.0)
    assert isinstance(embodiment, IsaacSim2YamEmbodiment)
    assert isinstance(embodiment, Embodiment)
    assert embodiment.info.name == "isaacsim-2yam"
    assert embodiment.info.control_hz == 20.0
    assert embodiment.info.is_simulated
    assert "privileged_success" in embodiment.info.capabilities
    policy: Policy = _MatchingPolicy()
    report = check_compatibility(policy, IsaacSim2YamEmbodiment())
    assert report.ok, report.errors


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"cam_height": 0}, "cam_height"),
        ({"cam_width": 0}, "cam_height"),
        ({"control_hz": 0.0}, "control_hz"),
        ({"control_hz": float("nan")}, "control_hz"),
        ({"spawn_noise": -1.0}, "spawn_noise"),
        ({"spawn_noise": float("nan")}, "spawn_noise"),
    ],
)
def test_constructor_validation(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        IsaacSim2YamEmbodiment(**kwargs)


def test_task_factory() -> None:
    task = put_everything_in_box(episodes=2, seed=7)
    assert task.name == "isaacsim-2yam-put-everything-in-box"
    assert task.max_steps == 400
    assert [scene.init_seed for scene in task.scenes] == [7, 8]
    assert len(task.scenes) == 2
    with pytest.raises(ValueError, match="episodes"):
        put_everything_in_box(episodes=0)


def test_asset_resolution_explicit_and_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset = tmp_path / "yam.xml"
    asset.write_text("<mujoco/>")
    assert IsaacSim2YamEmbodiment(asset_path=str(asset))._resolve_asset_path() == asset
    monkeypatch.setenv("MOLMOACT2_YAM_MJCF", str(asset))
    assert IsaacSim2YamEmbodiment()._resolve_asset_path() == asset
    monkeypatch.delenv("MOLMOACT2_YAM_MJCF")
    nested = tmp_path / "sim_eval/assets/yam/yam_mujoco"
    nested.mkdir(parents=True)
    nested_asset = nested / "bimanual_yam_linear_flattened.xml"
    nested_asset.write_text("<mujoco/>")
    monkeypatch.setenv("MOLMOACT2_ROOT", str(tmp_path))
    assert IsaacSim2YamEmbodiment()._resolve_asset_path() == nested_asset


def test_missing_asset_has_download_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MOLMOACT2_YAM_MJCF", raising=False)
    monkeypatch.delenv("MOLMOACT2_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="download_assets"):
        IsaacSim2YamEmbodiment(asset_path="/does/not/exist")._resolve_asset_path()


def test_prepare_isaac_mjcf_uses_private_unique_meshes(tmp_path: Path) -> None:
    assets = tmp_path / "meshes"
    assets.mkdir()
    (assets / "arm.obj").write_text("o arm\n")
    source = tmp_path / "yam.xml"
    source.write_text(
        """<mujoco><compiler meshdir="meshes"/><asset>
        <mesh name="left_arm" file="arm.obj"/>
        <mesh name="right_arm" file="arm.obj"/>
        </asset><worldbody><body name="bimanual_base"><body name="left_arm_body">
        <geom class="left_visual" mesh="left_arm"/>
        <geom class="right_collision" size="0.1 0.2"/>
        </body></body></worldbody></mujoco>"""
    )
    prepared = prepare_isaac_mjcf(source)
    prepared_root = prepared.path.parent
    tree = ET.parse(prepared.path)
    meshes = tree.getroot().findall("./asset/mesh")
    assert meshes[1].get("file") == "right_arm.obj"
    assert (prepared_root / "assets/right_arm.obj").is_file()
    assert [geom.get("type") for geom in tree.getroot().iter("geom")] == ["mesh", "capsule"]
    base = tree.getroot().find("./worldbody/body")
    assert base is not None and base.get("name") == "bimanual_base"
    assert base.find("inertial") is not None
    prepared.cleanup()
    assert not prepared_root.exists()


def test_prepare_isaac_mjcf_handles_minimal_explicit_geometry(tmp_path: Path) -> None:
    source = tmp_path / "minimal.xml"
    source.write_text(
        """<mujoco><asset/><worldbody>
        <geom type="plane"/><geom class="visual"/>
        </worldbody></mujoco>"""
    )
    prepared = prepare_isaac_mjcf(source)
    try:
        tree = ET.parse(prepared.path)
        geoms = list(tree.getroot().iter("geom"))
        assert (prepared.path.parent / "assets").is_dir()
        assert tree.getroot().find("compiler") is None
        assert geoms[0].get("type") == "plane"
        assert geoms[1].get("type") is None
    finally:
        prepared.cleanup()


def test_prepare_isaac_mjcf_preserves_inertial_and_cleans_up_on_error(
    tmp_path: Path,
) -> None:
    source = tmp_path / "existing-inertial.xml"
    source.write_text(
        """<mujoco><compiler meshdir="missing"/><asset/>
        <worldbody><body name="bimanual_base"><inertial/></body></worldbody></mujoco>"""
    )
    prepared = prepare_isaac_mjcf(source)
    try:
        base = ET.parse(prepared.path).getroot().find("./worldbody/body")
        assert base is not None and len(base.findall("inertial")) == 1
    finally:
        prepared.cleanup()

    broken = tmp_path / "broken.xml"
    broken.write_text(
        """<mujoco><compiler meshdir="missing"/><asset>
        <mesh name="right_missing" file="missing.obj"/>
        </asset><worldbody/></mujoco>"""
    )
    with pytest.raises(RuntimeError, match="mesh asset was not found"):
        prepare_isaac_mjcf(broken)


def test_reset_and_step_translation() -> None:
    embodiment = IsaacSim2YamEmbodiment()
    env = _inject(embodiment)
    observation = embodiment.reset(Scene(id="s", instruction="move", init_seed=4))
    assert env.seed == 4
    assert observation.instruction == "move"
    assert all(image.shape == (360, 640, 3) for image in observation.images.values())
    assert observation.state["joint_pos"].shape == (14,)
    result = embodiment.step(Action(data=np.zeros(14)))
    assert result.reward == 0.5
    assert not result.terminated
    assert not result.truncated
    env.next_terminated = True
    env.next_info = {"success": True}
    result = embodiment.step(Action(data=np.zeros(14)))
    assert result.termination_reason == "success"
    env.next_info = object()
    result = embodiment.step(Action(data=np.zeros(14)))
    assert result.termination_reason == "failure"
    env.next_terminated = False
    env.next_truncated = True
    result = embodiment.step(Action(data=np.zeros(14)))
    assert result.truncated


@pytest.mark.parametrize("action", [np.zeros(13), np.full(14, np.inf)])
def test_step_rejects_invalid_action(action: np.ndarray) -> None:
    embodiment = IsaacSim2YamEmbodiment()
    _inject(embodiment)
    with pytest.raises(ValueError):
        embodiment.step(Action(data=action))


def test_array_helpers_cover_float_empty_and_unwrapped_observation() -> None:
    image = module._to_image(_FakeTensor(np.asarray([[[[0.0, 0.5, 1.0]]]])))
    np.testing.assert_array_equal(image, np.asarray([[[0, 127, 255]]], dtype=np.uint8))
    empty = module._scalar(np.asarray([]))
    assert empty == 0.0
    observation = IsaacSim2YamEmbodiment()._to_observation(object(), None)
    assert not observation.images and not observation.state
    state_only = IsaacSim2YamEmbodiment()._to_observation(
        {"policy": {"joint_pos": np.zeros(14)}}, None
    )
    assert not state_only.images and "joint_pos" in state_only.state
    camera_only = IsaacSim2YamEmbodiment()._to_observation(
        {"policy": {"top_cam": np.zeros((2, 3, 3), dtype=np.uint8)}}, None
    )
    assert set(camera_only.images) == {"top_cam"} and not camera_only.state
    assert module._to_image(np.zeros((2, 3, 3), dtype=np.uint8)).shape == (2, 3, 3)


def test_ensure_env_wiring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    asset = tmp_path / "yam.xml"
    asset.write_text("<mujoco/>")
    calls: dict[str, Any] = {}
    fake_result = object()

    fake_gym = types.ModuleType("gymnasium")

    def make(task_id: str, *, cfg: Any, render_mode: str) -> object:
        calls["make"] = (task_id, cfg, render_mode)
        return fake_result

    fake_gym.make = make  # type: ignore[attr-defined]
    fake_isaacsim = types.ModuleType("isaacsim")
    fake_isaacsim.__path__ = []  # type: ignore[attr-defined]
    fake_core = types.ModuleType("isaacsim.core")
    fake_core.__path__ = []  # type: ignore[attr-defined]
    fake_utils = types.ModuleType("isaacsim.core.utils")
    fake_utils.__path__ = []  # type: ignore[attr-defined]
    fake_extensions = types.ModuleType("isaacsim.core.utils.extensions")

    def enable_extension(name: str) -> None:
        calls["extension"] = name

    fake_extensions.enable_extension = enable_extension  # type: ignore[attr-defined]
    fake_isaac_env = types.ModuleType("inspect_robots_isaacsim_2yam.isaac_env")

    def register_env() -> None:
        calls["registered"] = True

    def make_env_cfg(**kwargs: Any) -> dict[str, Any]:
        calls["cfg"] = kwargs
        return kwargs

    fake_isaac_env.register_env = register_env  # type: ignore[attr-defined]
    fake_isaac_env.make_env_cfg = make_env_cfg  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gymnasium", fake_gym)
    monkeypatch.setitem(sys.modules, "isaacsim", fake_isaacsim)
    monkeypatch.setitem(sys.modules, "isaacsim.core", fake_core)
    monkeypatch.setitem(sys.modules, "isaacsim.core.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "isaacsim.core.utils.extensions", fake_extensions)
    monkeypatch.setitem(sys.modules, "inspect_robots_isaacsim_2yam.isaac_env", fake_isaac_env)
    embodiment = IsaacSim2YamEmbodiment(asset_path=str(asset))
    monkeypatch.setattr(embodiment, "_ensure_app", lambda: object())
    assert embodiment._ensure_env() is fake_result
    assert embodiment._ensure_env() is fake_result
    assert calls["registered"] is True
    assert calls["extension"] == "isaacsim.asset.importer.mjcf"
    assert calls["cfg"]["height"] == 360


def test_ensure_env_wraps_import_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    asset = tmp_path / "yam.xml"
    asset.write_text("<mujoco/>")
    embodiment = IsaacSim2YamEmbodiment(asset_path=str(asset))
    monkeypatch.setattr(embodiment, "_ensure_app", lambda: object())
    original_import = builtins.__import__

    def fail_gym(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "gymnasium":
            raise ImportError("missing gym")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_gym)
    with pytest.raises(RuntimeError, match="Isaac Sim"):
        embodiment._ensure_env()


def test_ensure_app_launch_reuse_and_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    module._ACTIVE_APP = None
    fake_app = types.SimpleNamespace(close=lambda: None)
    fake_torch = types.ModuleType("torch")
    fake_isaaclab = types.ModuleType("isaaclab")
    fake_app_module = types.ModuleType("isaaclab.app")

    class Launcher:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["enable_cameras"] is True
            self.app = fake_app

    fake_app_module.AppLauncher = Launcher  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "isaaclab", fake_isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.app", fake_app_module)
    first = IsaacSim2YamEmbodiment()
    assert first._ensure_app() is fake_app
    assert first._ensure_app() is fake_app
    second = IsaacSim2YamEmbodiment()
    assert second._ensure_app() is fake_app
    first._app = None
    module._ACTIVE_APP = None

    fake_isaaclab.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.delitem(sys.modules, "isaaclab.app")
    with pytest.raises(RuntimeError, match="Isaac Sim"):
        IsaacSim2YamEmbodiment()._ensure_app()


def test_close_and_context_manager() -> None:
    module._ACTIVE_APP = None
    embodiment = IsaacSim2YamEmbodiment()
    env = _inject(embodiment)

    class App:
        closed = False

        def close(self) -> None:
            self.closed = True

    app = App()
    embodiment._app = app
    module._ACTIVE_APP = app
    prepared = types.SimpleNamespace(cleaned=False)

    def cleanup() -> None:
        prepared.cleaned = True

    prepared.cleanup = cleanup
    embodiment._prepared_asset = prepared  # type: ignore[assignment]
    embodiment.close()
    assert env.closed and app.closed and prepared.cleaned
    assert module._ACTIVE_APP is None
    embodiment.close()
    unrelated_app = App()
    another = IsaacSim2YamEmbodiment()
    another._app = app
    module._ACTIVE_APP = unrelated_app
    another.close()
    assert module._ACTIVE_APP is unrelated_app
    module._ACTIVE_APP = None
    with IsaacSim2YamEmbodiment() as entered:
        assert isinstance(entered, IsaacSim2YamEmbodiment)
