from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from inspect_robots import Action, Embodiment, Policy, Scene
from inspect_robots.compat import check_compatibility
from inspect_robots_libero import (
    LiberoEmbodiment,
    LiberoNoopPolicy,
    MolmoAct2LiberoPolicy,
    libero_10,
    libero_90,
    libero_embodiment,
    libero_goal,
    libero_noop_policy,
    libero_object,
    libero_spatial,
    molmoact2_libero_policy,
)
from inspect_robots_libero import embodiment as embodiment_module
from inspect_robots_libero import task as task_module
from inspect_robots_libero.contract import (
    action_space,
    math_sqrt_clamped,
    observation_space,
    quat_xyzw_to_axis_angle,
    robot_state,
)


class _TaskSpec:
    def __init__(self, index: int) -> None:
        self.name = f"task-{index}"
        self.language = f"do task {index}"
        self.problem_folder = "folder"
        self.bddl_file = f"task-{index}.bddl"
        self.init_states_file = f"task-{index}.init"


class _Suite:
    def __init__(self, count: int = 2) -> None:
        self.tasks = [_TaskSpec(index) for index in range(count)]

    def get_task(self, task_id: int) -> _TaskSpec:
        return self.tasks[task_id]


class _Benchmark:
    def __init__(self, suites: dict[str, Any] | None = None) -> None:
        self.suites = suites or {"libero_goal": _Suite}

    def get_benchmark_dict(self) -> dict[str, Any]:
        return self.suites


class _Controller:
    use_delta = False


class _Robot:
    def __init__(self) -> None:
        self.controller = _Controller()


class _FakeEnv:
    instances: ClassVar[list[_FakeEnv]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.closed = False
        self.seed_value: int | None = None
        self.robots = [_Robot()]
        self.success = False
        self.done = False
        self.reward = 0.25
        self.info: Any = {"raw": True}
        self.step_count = 0
        self.init_state: Any = None
        self.instances.append(self)

    @staticmethod
    def observation() -> dict[str, Any]:
        base = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        return {
            "agentview_image": base,
            "robot0_eye_in_hand_image": base + 20,
            "robot0_eef_pos": np.asarray([1.0, 2.0, 3.0]),
            "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.asarray([0.1, -0.1]),
        }

    def seed(self, value: int | None) -> None:
        self.seed_value = value

    def reset(self) -> dict[str, Any]:
        return self.observation()

    def set_init_state(self, state: Any) -> dict[str, Any]:
        self.init_state = state
        return self.observation()

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, Any]:
        assert action.shape == (7,)
        self.step_count += 1
        return self.observation(), self.reward, self.done, self.info

    def check_success(self) -> bool:
        return self.success

    def close(self) -> None:
        self.closed = True


def _install_fake_api(monkeypatch: pytest.MonkeyPatch, suite: _Suite | None = None) -> None:
    selected = suite or _Suite()
    benchmark = _Benchmark({"libero_goal": lambda: selected})
    monkeypatch.setattr(
        embodiment_module,
        "_libero_api",
        lambda: (benchmark, lambda kind: f"/assets/{kind}", _FakeEnv),
    )


def test_contract_and_state_conversion() -> None:
    space = action_space()
    assert space.shape == (7,)
    assert space.semantics is not None
    assert space.semantics.control_mode == "eef_delta_pose"
    assert space.semantics.rotation_repr == "axis_angle"
    assert observation_space().camera_names == frozenset({"image", "wrist_image"})
    assert observation_space().state_keys == frozenset({"state"})
    np.testing.assert_array_equal(quat_xyzw_to_axis_angle([0, 0, 0, 1]), np.zeros(3))
    angle = quat_xyzw_to_axis_angle([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    np.testing.assert_allclose(angle, [0, 0, np.pi / 2], atol=1e-6)
    assert math_sqrt_clamped(-1e-8) == 0.0
    state = robot_state(_FakeEnv.observation())
    np.testing.assert_allclose(state, [1, 2, 3, 0, 0, 0, 0.1, -0.1])


def test_contract_rejects_bad_state() -> None:
    with pytest.raises(ValueError, match="quaternion"):
        quat_xyzw_to_axis_angle([0, 0, 1])
    with pytest.raises(ValueError, match="missing"):
        robot_state({})
    bad = _FakeEnv.observation()
    bad["robot0_eef_pos"] = np.zeros(2)
    with pytest.raises(ValueError, match="unexpected"):
        robot_state(bad)


def test_load_suite_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    empty_suite = _Suite(0)
    benchmark = _Benchmark({"empty": lambda: empty_suite})
    monkeypatch.setattr(
        embodiment_module,
        "_libero_api",
        lambda: (benchmark, lambda kind: kind, _FakeEnv),
    )
    with pytest.raises(ValueError, match="unknown"):
        embodiment_module.load_suite("missing")
    with pytest.raises(ValueError, match="has no tasks"):
        embodiment_module.load_suite("empty")


def test_missing_libero_error_is_actionable() -> None:
    with pytest.raises(RuntimeError, match="official LIBERO"):
        embodiment_module._libero_api()


def test_libero_api_imports_official_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    root = types.ModuleType("libero")
    package = types.ModuleType("libero.libero")
    envs = types.ModuleType("libero.libero.envs")
    benchmark = object()
    get_path = object()
    env_type = object()
    package.benchmark = benchmark  # type: ignore[attr-defined]
    package.get_libero_path = get_path  # type: ignore[attr-defined]
    envs.OffScreenRenderEnv = env_type  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "libero", root)
    monkeypatch.setitem(sys.modules, "libero.libero", package)
    monkeypatch.setitem(sys.modules, "libero.libero.envs", envs)
    assert embodiment_module._libero_api() == (benchmark, get_path, env_type)


def test_libero_api_pins_paths_to_imported_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root = tmp_path / "libero/libero"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("")
    for child in ("assets", "bddl_files", "init_files"):
        (package_root / child).mkdir()
    root = types.ModuleType("libero")
    package = types.ModuleType("libero.libero")
    package.__file__ = str(package_root / "__init__.py")
    package.get_libero_path = lambda key: f"/wrong/{key}"  # type: ignore[attr-defined]
    benchmark = types.ModuleType("libero.libero.benchmark")
    package.benchmark = benchmark  # type: ignore[attr-defined]
    envs = types.ModuleType("libero.libero.envs")
    env_type = object()
    envs.OffScreenRenderEnv = env_type  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "libero", root)
    monkeypatch.setitem(sys.modules, "libero.libero", package)
    monkeypatch.setitem(sys.modules, "libero.libero.benchmark", benchmark)
    monkeypatch.setitem(sys.modules, "libero.libero.envs", envs)
    monkeypatch.delenv("INSPECT_ROBOTS_LIBERO_ROOT", raising=False)

    loaded_benchmark, get_path, loaded_env = embodiment_module._libero_api()

    assert loaded_benchmark is benchmark
    assert loaded_env is env_type
    assert get_path("benchmark_root") == str(package_root)
    assert get_path("bddl_files") == str(package_root / "bddl_files")
    assert benchmark.get_libero_path is get_path  # type: ignore[attr-defined]


def test_requested_libero_root_validation(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not a LIBERO checkout"):
        embodiment_module._requested_package_root(str(tmp_path))
    package_root = tmp_path / "libero/libero"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("")
    for child in ("assets", "bddl_files", "init_files"):
        (package_root / child).mkdir()
    resolved, import_root = embodiment_module._requested_package_root(str(tmp_path))
    assert resolved == package_root
    assert import_root == tmp_path


def test_headless_runtime_defaults_are_noninteractive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for variable in ("MUJOCO_GL", "NUMBA_CACHE_DIR", "MPLCONFIGDIR"):
        monkeypatch.delenv(variable, raising=False)
    embodiment_module._prepare_headless_runtime()
    assert os.environ["MUJOCO_GL"] == "egl"
    assert Path(os.environ["NUMBA_CACHE_DIR"]).is_dir()
    assert Path(os.environ["MPLCONFIGDIR"]).is_dir()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"task_id": -1},
        {"init_state_id": -1},
        {"cam_height": 0},
        {"cam_width": 0},
        {"control_hz": 0},
        {"control_hz": float("nan")},
        {"num_steps_wait": -1},
    ],
)
def test_embodiment_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        LiberoEmbodiment(**kwargs)


def test_embodiment_info_factory_protocol_and_policy_compatibility() -> None:
    embodiment = libero_embodiment(control_hz=10.0)
    policy = molmoact2_libero_policy(post_fn=lambda url, payload, timeout: {"actions": [[0] * 7]})
    assert isinstance(embodiment, LiberoEmbodiment)
    assert isinstance(embodiment, Embodiment)
    assert isinstance(policy, MolmoAct2LiberoPolicy)
    assert isinstance(policy, Policy)
    assert embodiment.info.control_hz == 10.0
    report = check_compatibility(policy, embodiment)
    assert report.ok, report.errors
    noop = libero_noop_policy()
    assert isinstance(noop, LiberoNoopPolicy)
    assert isinstance(noop, Policy)
    assert check_compatibility(noop, embodiment).ok


def test_reset_preprocessing_and_step(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeEnv.instances.clear()
    _install_fake_api(monkeypatch)
    embodiment = LiberoEmbodiment(cam_height=2, cam_width=3, num_steps_wait=2)
    monkeypatch.setattr(
        embodiment, "_load_init_states", lambda suite, task: np.asarray([[10], [20]])
    )
    scene = Scene(
        id="s",
        instruction="do it",
        init_seed=5,
        metadata={
            "libero_suite": "libero_goal",
            "libero_task_id": 1,
            "libero_init_state_id": 3,
        },
    )
    observation = embodiment.reset(scene)
    env = _FakeEnv.instances[-1]
    assert env.seed_value == 5
    np.testing.assert_array_equal(env.init_state, [20])
    assert env.step_count == 2
    assert env.robots[0].controller.use_delta is True
    raw = _FakeEnv.observation()["agentview_image"]
    np.testing.assert_array_equal(observation.images["image"], raw[::-1, ::-1])
    assert observation.images["image"].flags.c_contiguous
    assert observation.instruction == "do it"
    env.success = True
    result = embodiment.step(Action(data=np.zeros(7)))
    assert result.terminated and result.termination_reason == "success"
    assert result.reward == 0.25
    assert result.info["libero_task_id"] == 1
    env.success = False
    env.done = True
    env.info = object()
    result = embodiment.step(Action(data=np.zeros(7)))
    assert result.termination_reason == "environment_done"


def test_reset_without_init_state_and_explicit_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeEnv.instances.clear()
    _install_fake_api(monkeypatch)
    embodiment = LiberoEmbodiment(
        cam_height=2, cam_width=3, num_steps_wait=0, use_init_states=False
    )
    embodiment.reset(Scene(id="s", instruction="x", init_seed=1), seed=9)
    assert _FakeEnv.instances[-1].seed_value == 9
    assert _FakeEnv.instances[-1].init_state is None
    bad_scene = Scene(id="bad", instruction="x", metadata={"libero_task_id": -1})
    with pytest.raises(ValueError, match="must be >= 0"):
        embodiment.reset(bad_scene)


def test_ensure_env_cache_switch_and_range(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeEnv.instances.clear()
    _install_fake_api(monkeypatch)
    embodiment = LiberoEmbodiment(cam_height=2, cam_width=3)
    first = embodiment._ensure_env("libero_goal", 0)
    assert embodiment._ensure_env("libero_goal", 0) is first
    second = embodiment._ensure_env("libero_goal", 1)
    assert second is not first and first.closed
    with pytest.raises(ValueError, match="out of range"):
        embodiment._ensure_env("libero_goal", 2)


def test_init_state_loading_cache_and_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = _Suite()
    benchmark = _Benchmark({"libero_goal": lambda: suite})
    loaded = np.asarray([[1], [2]])
    fake_torch = types.ModuleType("torch")
    calls = {"count": 0}

    def load(path: Path, *, weights_only: bool) -> np.ndarray:
        assert weights_only is False
        calls["count"] += 1
        return loaded

    fake_torch.load = load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        embodiment_module,
        "_libero_api",
        lambda: (benchmark, lambda kind: str(tmp_path / kind), _FakeEnv),
    )
    embodiment = LiberoEmbodiment()
    embodiment._task_suite = suite
    init_dir = tmp_path / "init_states/folder"
    init_dir.mkdir(parents=True)
    init_file = init_dir / suite.tasks[0].init_states_file
    init_file.write_text("stub")
    assert embodiment._load_init_states("libero_goal", 0) is loaded
    assert embodiment._load_init_states("libero_goal", 0) is loaded
    assert calls["count"] == 1
    with pytest.raises(RuntimeError, match="does not exist"):
        embodiment._load_init_states("libero_goal", 1)
    second_file = init_dir / suite.tasks[1].init_states_file
    second_file.write_text("stub")
    fake_torch.load = lambda path, weights_only: []  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="empty"):
        embodiment._load_init_states("libero_goal", 1)


def test_step_guards_and_observation_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    embodiment = LiberoEmbodiment(cam_height=2, cam_width=3)
    with pytest.raises(RuntimeError, match="before reset"):
        embodiment.step(Action(data=np.zeros(7)))
    _install_fake_api(monkeypatch)
    embodiment = LiberoEmbodiment(cam_height=2, cam_width=3, use_init_states=False)
    embodiment.reset(Scene(id="s", instruction="x"))
    with pytest.raises(ValueError, match="shape"):
        embodiment.step(Action(data=np.zeros(6)))
    with pytest.raises(ValueError, match="non-finite"):
        embodiment.step(Action(data=np.full(7, np.nan)))
    raw = _FakeEnv.observation()
    del raw["agentview_image"]
    with pytest.raises(ValueError, match="missing camera"):
        embodiment._to_observation(raw, None)
    raw = _FakeEnv.observation()
    raw["agentview_image"] = np.zeros((2, 3))
    with pytest.raises(ValueError, match="has shape"):
        embodiment._to_observation(raw, None)
    raw = _FakeEnv.observation()
    raw["agentview_image"] = np.zeros((2, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="expected uint8"):
        embodiment._to_observation(raw, None)


def test_close_and_context_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_api(monkeypatch)
    embodiment = LiberoEmbodiment(cam_height=2, cam_width=3)
    embodiment.close()
    env = embodiment._ensure_env("libero_goal", 0)
    with embodiment as entered:
        assert entered is embodiment
    assert env.closed
    embodiment.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout_s": 0},
        {"timeout_s": float("nan")},
        {"cam_height": 0},
        {"cam_width": 0},
        {"action_horizon": 0},
        {"n_action_steps": 0},
    ],
)
def test_policy_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        MolmoAct2LiberoPolicy(**kwargs)


def _observation() -> Any:
    from inspect_robots import Observation

    return Observation(
        images={
            "image": np.zeros((2, 3, 3), dtype=np.uint8),
            "wrist_image": np.zeros((2, 3, 3), dtype=np.uint8),
        },
        state={"state": np.zeros(8, dtype=np.float32)},
    )


def test_policy_request_and_response() -> None:
    calls: dict[str, Any] = {}

    def post(url: str, payload: Any, timeout: float) -> dict[str, Any]:
        calls.update(url=url, payload=payload, timeout=timeout)
        return {"actions": np.zeros((1, 3, 7)), "latency_ms": 12}

    policy = MolmoAct2LiberoPolicy(
        server_url="http://host/", endpoint="custom", cam_height=2, cam_width=3, post_fn=post
    )
    policy.reset(Scene(id="s", instruction="stored"))
    chunk = policy.act(_observation())
    assert len(chunk.actions) == 3
    assert calls["url"] == "http://host/custom"
    assert calls["payload"]["instruction"] == "stored"
    assert calls["payload"]["norm_tag"] == "libero"
    assert calls["payload"]["n_action_steps"] == 10
    assert chunk.meta["server_latency_ms"] == 12
    assert policy.num_inferences == 1
    assert policy.server_url == "http://host"
    assert policy.server_metadata_url == "http://host/health"
    assert "serve_policy.py" in policy.remedy
    observation = _observation()
    observation = type(observation)(
        images=observation.images, state=observation.state, instruction="live"
    )
    policy.act(observation)
    assert calls["payload"]["instruction"] == "live"
    policy_without_latency = MolmoAct2LiberoPolicy(
        cam_height=2,
        cam_width=3,
        post_fn=lambda url, payload, timeout: {"actions": np.zeros((1, 7))},
    )
    assert policy_without_latency.act(_observation()).meta == {}


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ({}, "missing"),
        ({"actions": []}, "shape"),
        ({"actions": np.empty((0, 7))}, "empty"),
        ({"actions": np.zeros((2, 6))}, "shape"),
        ({"actions": np.full((1, 7), np.nan)}, "non-finite"),
        ({"actions": np.zeros((1, 7)), "latency_ms": -1}, "negative"),
    ],
)
def test_policy_rejects_bad_responses(response: dict[str, Any], message: str) -> None:
    policy = MolmoAct2LiberoPolicy(
        cam_height=2,
        cam_width=3,
        post_fn=lambda url, payload, timeout: response,
    )
    with pytest.raises(ValueError, match=message):
        policy.act(_observation())


def test_policy_rejects_bad_observations() -> None:
    from inspect_robots import Observation

    policy = MolmoAct2LiberoPolicy(
        cam_height=2,
        cam_width=3,
        post_fn=lambda url, payload, timeout: {"actions": np.zeros((1, 7))},
    )
    valid = _observation()
    with pytest.raises(ValueError, match="missing camera"):
        policy.act(Observation(images={}, state=valid.state))
    with pytest.raises(ValueError, match="has shape"):
        policy.act(
            Observation(images={**valid.images, "image": np.zeros((1, 1, 3))}, state=valid.state)
        )
    with pytest.raises(ValueError, match="missing state"):
        policy.act(Observation(images=valid.images, state={}))
    with pytest.raises(ValueError, match="state has shape"):
        policy.act(Observation(images=valid.images, state={"state": np.zeros(7)}))
    with pytest.raises(ValueError, match="non-finite"):
        policy.act(Observation(images=valid.images, state={"state": np.full(8, np.inf)}))


def test_noop_policy_emits_one_open_gripper_action() -> None:
    policy = LiberoNoopPolicy(cam_height=2, cam_width=3)
    policy.reset(Scene(id="s", instruction="smoke test"))
    chunk = policy.act(_observation())
    assert len(chunk.actions) == 1
    np.testing.assert_array_equal(chunk.actions[0].data, [0, 0, 0, 0, 0, 0, -1])


def test_task_factory_and_registered_suite_wrappers(monkeypatch: pytest.MonkeyPatch) -> None:
    suite = _Suite(3)
    monkeypatch.setattr(task_module, "load_suite", lambda name: suite)
    task = task_module.libero_suite_task(
        "libero_goal", task_ids="2,0,2", init_state_ids="3,1", seed=4, max_steps=9
    )
    assert task.max_steps == 9
    assert len(task.scenes) == 4
    assert task.scenes[0].instruction == "do task 0"
    assert task.scenes[0].metadata["libero_init_state_id"] == 1
    assert task.scenes[-1].init_seed == 7
    for factory, expected in (
        (libero_spatial, "libero_spatial"),
        (libero_object, "libero_object"),
        (libero_goal, "libero_goal"),
        (libero_10, "libero_10"),
        (libero_90, "libero_90"),
    ):
        result = factory(task_ids="0", episodes_per_task=2)
        assert result.metadata["suite"] == expected
        assert len(result.scenes) == 2


def test_task_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(task_module, "load_suite", lambda name: _Suite())
    with pytest.raises(ValueError, match="unsupported"):
        task_module.libero_suite_task("bad")
    with pytest.raises(ValueError, match="episodes_per_task"):
        task_module.libero_suite_task("libero_goal", episodes_per_task=0)
    with pytest.raises(ValueError, match="select"):
        task_module.libero_suite_task("libero_goal", task_ids="")
    with pytest.raises(ValueError, match="out of range"):
        task_module.libero_suite_task("libero_goal", task_ids="3")
    with pytest.raises(ValueError, match="out of range"):
        task_module.libero_suite_task("libero_goal", init_state_ids="-1")
    task = task_module.libero_suite_task("libero_goal", task_ids=[1], episodes_per_task=1)
    assert task.scenes[0].metadata["libero_task_id"] == 1
