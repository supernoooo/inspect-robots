"""SO-101 bridge contract checks that do not initialize physical devices."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from inspect_robots import colosseum as bridge
from inspect_robots.cli import main
from inspect_robots.embodiment import AUTO_RESET, EmbodimentInfo
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.spaces import ActionSemantics, Box, CameraSpec, ObservationSpace
from inspect_robots.types import Action, ActionChunk, Observation, StepResult

bridge.load_dependencies()


ROOT = Path(__file__).resolve().parents[1]
CATALOG = (
    ROOT / "integrations/colosseum-policy-server/configs/"
    "local-runtime-so101-agent.yaml.example"
)


def test_so101_catalog_and_camera_contract() -> None:
    models = bridge.read_catalog(str(CATALOG), control_hz=10, robot_type="so101")
    assert len(models) == 3
    assert all(model["action_dim"] == 6 for model in models)
    with pytest.raises(ValueError, match="14-D"):
        bridge.read_catalog(str(CATALOG), control_hz=10, robot_type="yam")
    config = SimpleNamespace(
        robot_type="so101", cameras={"head_image": "front", "left_image": "wrist"}
    )
    assert bridge.camera_map(config) == {"head_image": "front", "left_image": "wrist"}
    with pytest.raises(ValueError, match="camera"):
        bridge.camera_map(replace_config(config, cameras={"right_image": "unsupported"}))


def replace_config(config: SimpleNamespace, **kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(**{**vars(config), **kwargs})


def test_so101_eef_setup_uses_single_arm_device_slots() -> None:
    from inspect_robots._setup import _read_raw_config, _render_config
    from inspect_robots.cli import build_parser
    from inspect_robots.conformance import device_slots
    from inspect_robots.registry import registered

    args = build_parser().parse_args(["setup", "--config", ".so101/config.ini"])
    assert args.config == ".so101/config.ini"
    slots = device_slots(registered("embodiment")["so101_eef_configured"])
    assert [(slot.arg, slot.kind) for slot in slots] == [
        ("port", "serial"), ("camera_device", "v4l2")
    ]
    carried = _read_raw_config(ROOT / ".so101/config.ini")
    assert isinstance(carried, dict)
    rendered = _render_config(
        carried["defaults"],
        {slot.arg: carried["embodiment.args"][slot.arg] for slot in slots},
        carried,
        managed_args=tuple(slot.arg for slot in slots),
    )
    for key in ("head_camera_serial", "urdf_path", "leader_port"):
        assert f"{key} = " in rendered


def test_so101_native_args_keep_joint_config_and_frequency(tmp_path: Path) -> None:
    ini = tmp_path / "so101.ini"
    ini.write_text(
        "[defaults]\nembodiment=so101_configured\n"
        "[embodiment.args]\nrobot_id=follower_arm\ncontrol_hz=10\n"
        "use_degrees=true\njoints_are_delta=false\n"
    )
    options = SimpleNamespace(inspect_args=["--", "--config", str(ini)])
    task = {"instruction": "lift the block", "max_steps": 1200}
    args = bridge.native_args(options, task, tmp_path, control_hz=10, robot_type="so101")
    assert args.embodiment == "so101_configured"
    assert args.max_action_delta is None
    assert args.max_steps == 1200
    assert "control_hz=10" in args.embodiment_args
    ini.write_text(ini.read_text().replace("use_degrees=true", "use_degrees=false"))
    with pytest.raises(ValueError, match="use_degrees"):
        bridge.native_args(options, task, tmp_path, control_hz=10, robot_type="so101")


def test_so101_eef_native_args_and_recording_contract(tmp_path: Path) -> None:
    ini = tmp_path / "eef.ini"
    ini.write_text(
        "[defaults]\nembodiment=so101_eef_configured\n"
        "[embodiment.args]\ncontrol_hz=10\nuse_degrees=true\n"
        "urdf_path=/tmp/so101.urdf\n"
    )
    options = SimpleNamespace(inspect_args=["--", "--config", str(ini)])
    task = {"instruction": "lift the block", "max_steps": 20}
    args = bridge.native_args(options, task, tmp_path, control_hz=10, robot_type="so101")
    assert args.embodiment == "so101_eef_configured"
    assert "control_hz=10" in args.embodiment_args
    assert "use_degrees=True" in args.embodiment_args
    assert "joints_are_delta=False" in args.embodiment_args
    assigned = bridge.native_args(options, task, tmp_path, control_hz=30, robot_type="so101")
    assert "control_hz=30" in assigned.embodiment_args
    explicit = SimpleNamespace(inspect_args=[*options.inspect_args, "-E", "control_hz=10"])
    with pytest.raises(ValueError, match="assigned frequency"):
        bridge.native_args(explicit, task, tmp_path, control_hz=30, robot_type="so101")

    class Agent:
        def bind(self, info: EmbodimentInfo) -> None:
            self.info = info

    agent = Agent()
    model = {"action_space": "joint_position", "action_dim": 6,
             "control_hz": 10, "max_horizon": 150}
    session = SimpleNamespace(preparation={"model": model}, ready=lambda: None)
    config = SimpleNamespace(robot_type="so101", test=False)
    space = Box((7,), semantics=ActionSemantics(
        "eef_delta_pose", rotation_repr="axis_angle", frame="base"
    ))
    info = EmbodimentInfo(
        "so101_eef", space,
        observation_space=ObservationSpace(cameras=(CameraSpec("front", 16, 16),)),
        control_hz=10, is_simulated=False,
    )
    policy = bridge.ReportingPolicy(
        agent, session, config, {"head_image": "front"}, control_mode="eef_delta_pose"
    )
    policy.bind(info)
    assert agent.info.action_space.dim == 7
    with pytest.raises(ValueError, match="joint execution"):
        policy.bind(replace(info, action_space=Box((6,), semantics=space.semantics)))
    with pytest.raises(ValueError, match="frequency"):
        policy.bind(replace(info, control_hz=30))
    with pytest.raises(ValueError, match="camera"):
        bridge.ReportingPolicy(
            agent, session, config, {"head_image": "missing"},
            control_mode="eef_delta_pose",
        ).bind(info)


def test_so101_records_measured_state_and_accepted_action(tmp_path: Path) -> None:
    config = SimpleNamespace(robot_type="so101")
    events = []
    session = SimpleNamespace(event=lambda *args, **kwargs: events.append((args, kwargs)))
    sink = bridge.RecordingSink(
        tmp_path, config, session, {"head_image": "head", "left_image": "wrist"}
    )
    state = np.array([1, 2, 3, 4, 5, 40], dtype=np.float64)
    native = Observation(
        state={"joint_pos": state},
        images={"head": np.zeros((2, 2, 3), np.uint8), "wrist": np.zeros((2, 2, 3), np.uint8)},
    )
    split = sink.observation(native)
    np.testing.assert_array_equal(split.joints, state[:5])
    np.testing.assert_array_equal(split.gripper, state[5:])
    assert set(split.images) == {"head_image", "left_image"}
    accepted = np.array([1, 2, 3, 4, 5, 35], dtype=np.float64)
    sink.log_step(
        0, native, Action(np.zeros(6)),
        StepResult(native, info={"so101_joint_command": accepted}),
    )
    assert sink.frame_count == 1
    assert events[0][1]["step"] == 0
    with pytest.raises(ValueError, match="accepted motor command"):
        sink.log_step(1, native, Action(np.zeros(6)), StepResult(native))


def test_so101_rejects_mismatched_assignment() -> None:
    assignment = {
        "id": "ev1", "robot_id": "yam", "inference_mode": "local",
        "test": False, "recording": True, "track": "open",
        "task": {"instruction": "lift", "max_steps": 10, "cameras": ["head_image"]},
        "runs": [{"id": "a", "side": "A"}, {"id": "b", "side": "B"}],
    }
    config = SimpleNamespace(
        robot_type="so101", test=False, recording=True, cameras={"head_image": "head"}
    )
    with pytest.raises(ValueError, match="robot"):
        bridge.validate_assignment(config, assignment)
    assignment["robot_id"] = "so101"
    bridge.validate_assignment(config, assignment)


def test_so101_router_frequency_uses_official_keyed_endpoint() -> None:
    models = bridge.read_catalog(str(CATALOG), robot_type="so101")
    local = next(model for model in models if model["name"] == "grok-4.7")
    router = {
        **bridge.public_model(local),
        "url": "https://api.yhlxj.ai/v1",
        "control_hz": 30,
    }
    config = SimpleNamespace(
        robot_type="so101", llm_api_keys={"xai": "test-key"},
        llm_api_base_urls={"xai": "https://api.x.ai/v1"},
        llm_account_logins={},
    )
    selected = bridge.match_runtime_model(router, models, config)
    assert selected["url"] == "https://api.x.ai/v1"
    assert selected["control_hz"] == 30
    assert local["control_hz"] == 10
    for change in ({"revision": "wrong"}, {"action_dim": 7}, {"control_hz": 0}):
        with pytest.raises(ValueError, match="differs"):
            bridge.match_runtime_model({**router, **change}, models, config)
    with pytest.raises(ValueError, match="differs"):
        bridge.match_runtime_model(router, models, replace_config(config, llm_api_keys={}))
    relay_catalog = [{**local, "url": "https://api.yhlxj.ai/v1"}]
    with pytest.raises(ValueError, match="official API URL"):
        bridge.validate_catalog_keys(
            replace_config(config, llm_api_base_urls={"xai": "https://api.yhlxj.ai/v1"}),
            relay_catalog,
        )


def test_so101_router_claude_relay_uses_official_cli_login() -> None:
    models = bridge.read_catalog(str(CATALOG), robot_type="so101")
    local = next(model for model in models if model["name"] == "claude-opus-5-5")
    router = {
        **bridge.public_model(local),
        "url": "https://api.yhlxj.ai", "control_hz": 30,
    }
    config = SimpleNamespace(
        robot_type="so101", llm_api_keys={},
        llm_api_base_urls={"anthropic": "https://api.anthropic.com/v1"},
        llm_account_logins={"anthropic": "claude-code"},
    )
    selected = bridge.match_runtime_model(router, models, config)
    assert selected["url"] == "https://api.anthropic.com/v1"
    assert selected["control_hz"] == 30
    with pytest.raises(ValueError, match="Claude model"):
        bridge.match_runtime_model(
            {**router, "revision": "other"}, models, config
        )
    with pytest.raises(ValueError, match="differs"):
        bridge.match_runtime_model(
            router, models, replace_config(config, llm_account_logins={})
        )


def test_so101_router_frequency_reaches_native_preview_and_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inspect_robots import cli

    models = bridge.read_catalog(str(CATALOG), robot_type="so101")
    local = next(model for model in models if model["name"] == "grok-4.7")
    router = {
        **bridge.public_model(local),
        "url": "https://api.yhlxj.ai/v1", "control_hz": 30,
    }
    config = SimpleNamespace(
        robot_type="so101", control_hz=10, test=True,
        cameras={"head_image": "front", "left_image": "wrist"},
        llm_api_keys={"xai": "test-key"},
        llm_api_base_urls={"xai": "https://api.x.ai/v1"},
        llm_account_logins={}, recording=True,
    )
    assignment = {
        "task": {"instruction": "lift", "max_steps": 2,
                 "cameras": ["head_image", "left_image"]}
    }
    trial = {"id": "run_a", "side": "A"}
    ini = tmp_path / "eef.ini"
    ini.write_text(
        "[defaults]\nembodiment=so101_eef_configured\n"
        "[embodiment.args]\ncontrol_hz=10\nuse_degrees=true\n"
    )
    options = SimpleNamespace(inspect_args=["--", "--config", str(ini)])
    path = tmp_path / "run_a"
    path.mkdir()
    seen = []

    class Session:
        def __init__(self, *_: object) -> None:
            self.preparation = {"model": router}

        def prepare(self) -> dict[str, object]:
            return router

        def close(self) -> None:
            pass

    def fake_run(args: object, **_: object) -> int:
        seen.append(args)
        return 1  # Stop before any embodiment or hardware is constructed.

    monkeypatch.setattr(bridge, "make_agent", lambda *_: SimpleNamespace())
    monkeypatch.setattr(cli, "_cmd_run", fake_run)
    with pytest.raises(RuntimeError, match="Inspect trial failed"):
        bridge.execute_trial(
            config, assignment, trial, path, options, models, session_factory=Session
        )
    assert len(seen) == 1
    assert "control_hz=30" in seen[0].embodiment_args
    context = json.loads((path / "recording-context.json").read_text())
    assert context["fps"] == 30
    assert context["native_llm_url"] == "https://api.x.ai/v1"
    assert context["native_control_mode"] == "eef_delta_pose"


def test_so101_wrapper_captures_driver_limited_target() -> None:
    class Arm:
        info = SimpleNamespace(
            action_space=Box(shape=(6,), semantics=ActionSemantics(control_mode="joint_pos"))
        )

        def _send(self, command: np.ndarray) -> np.ndarray:
            return np.clip(command, -5, 5)

        def step(self, action: Action) -> StepResult:
            self._send(action.data)
            return StepResult(Observation())

    arm = Arm()
    wrapped = bridge.CaptureSO101Embodiment(arm)
    result = wrapped.step(Action(np.array([10, 0, 0, 0, 0, 2], dtype=np.float64)))
    np.testing.assert_array_equal(result.info["so101_joint_command"], [5, 0, 0, 0, 0, 2])
    assert "_send" not in vars(arm)


def test_so101_eef_wrapper_captures_inner_driver_command() -> None:
    class Driver:
        def _send(self, command: np.ndarray) -> np.ndarray:
            return np.clip(command, -5, 5)

        def step(self, action: Action) -> StepResult:
            self._send(action.data)
            return StepResult(Observation())

    class EEF:
        info = SimpleNamespace(action_space=Box(
            shape=(7,), semantics=ActionSemantics("eef_delta_pose")
        ))

        def __init__(self) -> None:
            self._arm = Driver()

        def step(self, action: Action) -> StepResult:
            return self._arm.step(Action(np.array([10, 0, 0, 0, 0, 2], dtype=float)))

    eef = EEF()
    result = bridge.CaptureSO101Embodiment(eef).step(Action(np.zeros(7)))
    np.testing.assert_array_equal(result.info["so101_joint_command"], [5, 0, 0, 0, 0, 2])
    assert "_send" not in vars(eef._arm)


def test_so101_eef_router_horizon_is_exposed_and_never_truncates() -> None:
    class Agent:
        def bind(self, info: EmbodimentInfo) -> None:
            self.bound = info

        def act(self, observation: Observation) -> ActionChunk:
            return ActionChunk([Action(np.zeros(7)) for _ in range(4)], control_hz=30)

    agent = Agent()
    session = SimpleNamespace(
        preparation={"model": {
            "action_space": "joint_position", "action_dim": 6,
            "control_hz": 30, "max_horizon": 3,
        }},
        failure=None,
        ready=lambda: None,
    )
    policy = bridge.ReportingPolicy(
        agent, session, SimpleNamespace(robot_type="so101", test=False), {},
        control_mode="eef_delta_pose",
    )
    info = EmbodimentInfo(
        "so101_eef",
        Box(
            (7,), low=np.full(7, -0.002), high=np.full(7, 0.002),
            semantics=ActionSemantics(
                "eef_delta_pose", rotation_repr="axis_angle", frame="base",
                dim_labels=("x", "y", "z", "rx", "ry", "rz", "gripper"),
                agent_eef_target=True,
            ),
        ),
        observation_space=ObservationSpace(),
        control_hz=30,
        is_simulated=False,
    )
    policy.bind(info)
    assert agent.bound.action_space.semantics.max_chunk_steps == 3
    with pytest.raises(ValueError, match="exceeds the Router max_horizon"):
        policy.act(Observation())


@pytest.mark.parametrize("eef", [False, True])
def test_so101_local_pair_records_and_exports_without_router(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, eef: bool
) -> None:
    if shutil.which("ffmpeg") is None:
        import imageio_ffmpeg

        ffmpeg = tmp_path / "ffmpeg"
        ffmpeg.symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
        monkeypatch.setenv("PATH", str(tmp_path) + ":" + __import__("os").environ["PATH"])
    import yaml

    from inspect_robots import registry

    class Agent:
        def __init__(self) -> None:
            self.config = PolicyConfig()
            self.info = PolicyInfo("agent", Box((6,)))

        def bind(self, info: EmbodimentInfo) -> None:
            self.info = PolicyInfo(
                "agent", info.action_space, info.observation_space, info.control_hz
            )

        def reset(self, scene: object) -> None:
            pass

        def act(self, observation: Observation) -> ActionChunk:
            data = ([0.001, 0, 0, 0, 0, 0, 1] if eef else [1, 2, 3, 4, 5, 70])
            return ActionChunk([Action(np.array(data, dtype=float))], control_hz=10)

    class Arm:
        def __init__(self, **kwargs: object) -> None:
            self.state = np.array([0, 0, 0, 0, 0, 40], dtype=float)
            self.info = EmbodimentInfo(
                "so101_configured",
                Box(
                    (6,), low=np.array([-90] * 5 + [0]), high=np.array([90] * 5 + [100]),
                    semantics=ActionSemantics("joint_pos"),
                ),
                observation_space=ObservationSpace(
                    cameras=(CameraSpec("front", 16, 16), CameraSpec("wrist", 16, 16))
                ),
                control_hz=kwargs.get("control_hz", 10),
                is_simulated=False,
                capabilities=frozenset({AUTO_RESET}),
            )

        def observe(self) -> Observation:
            image = np.full((16, 16, 3), 80, dtype=np.uint8)
            return Observation(
                images={"front": image, "wrist": image.copy()},
                state={"joint_pos": self.state.copy()},
            )

        def reset(self, scene: object, *, seed: int | None = None) -> Observation:
            return self.observe()

        def _send(self, command: np.ndarray) -> np.ndarray:
            return np.minimum(command, np.array([1, 2, 3, 4, 5, 60], dtype=float))

        def step(self, action: Action) -> StepResult:
            self.state = self._send(action.data)
            return StepResult(self.observe())

        def close(self) -> None:
            pass

    class EEF:
        def __init__(self, **kwargs: object) -> None:
            self._arm = Arm(**kwargs)
            self.info = replace(
                self._arm.info, name="so101_eef",
                action_space=Box(
                    (7,), low=np.array([-0.002] * 3 + [-0.01] * 3 + [-5.0]),
                    high=np.array([0.002] * 3 + [0.01] * 3 + [5.0]),
                    semantics=ActionSemantics(
                        "eef_delta_pose", rotation_repr="axis_angle", frame="base"
                    ),
                ),
            )

        def __getattr__(self, name: str) -> object:
            return getattr(self._arm, name)

        def step(self, action: Action) -> StepResult:
            return self._arm.step(Action(np.array([1, 2, 3, 4, 5, 70], dtype=float)))

    client = yaml.safe_load(
        (ROOT / "integrations/colosseum-client/configs/"
         "robot.inspect-so101-account.yaml.example").read_text()
    )
    client["token"] = "test-token"
    client["llm_api_keys"] = {"openai": "test-openai", "xai": "test-xai"}
    client.pop("llm_account_logins")
    client["evaluation_dir"] = str(tmp_path / "runs")
    client["max_trial_steps"] = 2
    client_path = tmp_path / "client.yaml"
    client_path.write_text(yaml.safe_dump(client))
    registry.registered("embodiment")
    name = "so101_eef_configured" if eef else "so101_configured"
    monkeypatch.setitem(registry._FACTORIES["embodiment"], name, EEF if eef else Arm)
    monkeypatch.setattr(bridge, "make_agent", lambda *args: Agent())
    answers = iter(["3", "4", "B", "okay", "clear"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    ini = tmp_path / "native.ini"
    ini.write_text(f"[defaults]\npolicy=agent\nembodiment={name}\n")
    result = main([
        "colosseum", "--client-config", str(client_path), "--runtime-config", str(CATALOG),
        "--local-pair", "--model-a", "gpt-6-astra", "--model-b", "grok-4.7",
        "--defer-submit", "--instruction", "Lift and replace the block.",
        "--scene", "Block on the table", "--evaluator", "tester",
        "--", "--config", str(ini), "--embodiment", name,
        "--no-rerun", "--no-rerun-save", "--no-store-frames",
    ])
    assert result == 0
    manifest_path = next((tmp_path / "runs").glob("local_*/manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    assert manifest["assignment"]["robot_id"] == "so101"
    for run in manifest["assignment"]["runs"]:
        folder = manifest_path.parent / run["id"]
        context = json.loads((folder / "recording-context.json").read_text())
        assert context["robot_type"] == "so101"
        assert context["camera_map"] == {"head_image": "front", "left_image": "wrist"}
        assert context["native_control_mode"] == ("eef_delta_pose" if eef else "joint_pos")
        assert context["joint_command_info_key"] == "so101_joint_command"
        rows = [json.loads(line) for line in (folder / "frames.jsonl").read_text().splitlines()]
        assert rows[0]["action"][5] == 60
        assert len(rows[0]["joints"]) == 5 and len(rows[0]["gripper"]) == 1
        exported = json.loads((folder / "lerobot/meta/colosseum.json").read_text())
        assert exported["robot_type"] == "so101"
    from inspect_robots.colosseum_review import ReviewStore

    store = ReviewStore(str(client_path))
    evaluation_id = manifest["assignment"]["id"]
    assert store.complete(evaluation_id, manifest)
    page = store.evaluation_page(evaluation_id).decode()
    assert page.count("<div>head_image</div>") == 2
    assert page.count("<div>wrist_image</div>") == 2
    assert "<div>left_image</div>" not in page
    for run in manifest["assignment"]["runs"]:
        assert f'/video/{evaluation_id}/{run["id"]}/left_image' in page
        for camera in ("head_image", "left_image"):
            preview = store.preview(evaluation_id, run["id"], camera)
            assert preview is not None and preview.stat().st_size > 0
