"""Native rollout, recording, identity checks and durable Colosseum recovery."""

from __future__ import annotations

import base64
import copy
import dataclasses
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest

from inspect_robots import colosseum as bridge
from inspect_robots.cli import build_parser, main
from inspect_robots.embodiment import AUTO_RESET, EmbodimentInfo
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.spaces import ActionSemantics, Box, CameraSpec, ObservationSpace
from inspect_robots.types import Action, ActionChunk, Observation, StepResult

pytest.importorskip("google.protobuf")
pytest.importorskip("pyarrow")
pytest.importorskip("yaml")
pytest.importorskip("websockets.sync.client")
bridge.load_dependencies()

from colosseum_client.local_protocol import decode_control, encode_control  # noqa: E402
from colosseum_client.robot_config import RobotClientConfig  # noqa: E402

MODEL = {
    "model_type": "llm",
    "name": "gpt-6-astra",
    "url": "https://relay.example/v1",
    "revision": "rig-v1",
    "action_space": "joint_position",
    "action_dim": 14,
    "control_hz": 30,
    "max_horizon": 150,
}


class FakeAgent:
    def __init__(self) -> None:
        self.config = PolicyConfig()
        self.info = PolicyInfo("agent", Box((14,)))
        self._client = SimpleNamespace(close=lambda: None)

    def bind(self, info: Any) -> None:
        self.info = PolicyInfo("agent", info.action_space, info.observation_space, info.control_hz)

    def reset(self, scene: Any) -> None:
        self.scene = scene

    def act(self, observation: Any) -> ActionChunk:
        # A multi-action chunk exercises recording on steps without new inference.
        values = np.zeros(14)
        values[[6, 13]] = [0.25, 0.75]
        return ActionChunk(
            [Action(values), Action(values.copy())], control_hz=self.info.control_hz or 30
        )


class FakeYam:
    def __init__(self, executions: list[str], **kwargs: Any) -> None:
        self.executions = executions
        self.info = EmbodimentInfo(
            "bridge_test_yam",
            Box(
                (14,),
                low=np.full(14, -2),
                high=np.full(14, 2),
                semantics=ActionSemantics("joint_pos"),
            ),
            observation_space=ObservationSpace(
                cameras=tuple(CameraSpec(name, 16, 16) for name in bridge._CAMERAS.values())
            ),
            control_hz=kwargs.get("control_hz", 30),
            is_simulated=True,
            capabilities=frozenset({AUTO_RESET}),
        )
        self.values: np.ndarray[Any, Any] = np.zeros(14)
        self.values[[6, 13]] = [0.4, 0.6]

    def observe(self) -> Observation:
        return Observation(
            images={name: np.full((16, 16, 3), 80, np.uint8) for name in bridge._CAMERAS.values()},
            state={"joint_pos": self.values.copy()},
        )

    def reset(self, scene: Any, *, seed: int | None = None) -> Observation:
        self.executions.append("reset")
        return self.observe()

    def step(self, action: Action) -> StepResult:
        self.executions.append("step")
        self.values = np.array(action.data)
        return StepResult(self.observe())

    def close(self) -> None:
        self.executions.append("close")


class FakeEEF(FakeYam):
    joint_command_info_key = "yam_joint_command"

    def __init__(self, executions: list[str], **kwargs: Any) -> None:
        super().__init__(executions, **kwargs)
        self.info = dataclasses.replace(
            self.info,
            name="yam_eef",
            action_space=dataclasses.replace(
                self.info.action_space, semantics=ActionSemantics("eef_abs_pose")
            ),
        )

    def step(self, action: Action) -> StepResult:
        self.executions.append("step")
        sent = np.full(14, 0.05)
        sent[7:13] = -0.05
        sent[[6, 13]] = action.data[[6, 13]]
        # Measured feedback intentionally differs from the command, as it can
        # on a real arm before it has reached its target.
        self.values = sent.copy()
        self.values[:6] -= 0.01
        return StepResult(self.observe(), info={self.joint_command_info_key: sent.tolist()})


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    if shutil.which("ffmpeg") is None:
        pytest.skip("FFmpeg required for actual dataset export")
    from colosseum_client import evaluation
    from websockets.sync import client as ws_client

    from inspect_robots import registry

    assignment: dict[str, Any] = {
        "id": "ev_native",
        "robot_id": "yam",
        "track": "open",
        "inference_mode": "local",
        "test": True,
        "recording": True,
        "state": "pending",
        "institution": "test",
        "task": {"instruction": "pick the box", "max_steps": 2, "cameras": list(bridge._CAMERAS)},
        "runs": [
            {"id": "run_a", "side": "A", "state": "assigned"},
            {"id": "run_b", "side": "B", "state": "assigned"},
        ],
    }
    events: list[tuple[Any, ...]] = []
    executions: list[str] = []
    uploads: list[str] = []
    failures = {"result": False, "upload": False, "step": False}

    class API:
        def __init__(self, config: Any) -> None:
            self.config = config

        def request(self, method: str, path: str, body: Any = None) -> dict[str, Any]:
            events.append((method, path))
            if path.endswith("/result"):
                assert executions.count("reset") == 2
                assert not self.config.recording or all(
                    (tmp_path / "runs/ev_native" / r["id"] / "lerobot/meta/info.json").exists()
                    for r in assignment["runs"]
                )
                if failures["result"]:
                    failures["result"] = False
                    raise RuntimeError("submission temporarily unavailable")
                assignment["state"] = "completed"
                assignment["result"] = body
                return {"state": "completed"}
            return copy.deepcopy(assignment)

        def upload_dataset(self, run_id: str, root: Path) -> None:
            assert assignment["state"] == "completed", "publish scores before dataset uploads"
            assert executions.count("reset") == 2
            if failures["upload"]:
                failures["upload"] = False
                raise RuntimeError("upload temporarily unavailable")
            uploads.append(run_id)

        def close(self) -> None:
            pass

    class Socket:
        def __init__(self, url: str, **kwargs: Any) -> None:
            self.trial = next(r for r in assignment["runs"] if f"/{r['id']}/local" in url)
            self.pending = encode_control(
                {
                    "type": "prepare",
                    "protocol_version": 1,
                    "run_id": self.trial["id"],
                    "preparation_id": "prep_test",
                    "model": MODEL,
                    "task": assignment["task"],
                    "test": True,
                }
            )

        def send(self, raw: bytes) -> None:
            message = decode_control(raw)
            kind = message["type"]
            events.append((kind, self.trial["id"], message))
            assert "api_key" not in message
            if kind == "ready":
                assert bridge.public_model(message["model"]) == MODEL
                assert message["loaded"] is True
                self.trial["state"] = "running"
            elif kind == "finish":
                assert executions[-1] == "close", "finish follows native hardware cleanup"
                self.trial["state"] = "finished"
            if kind == "step" and failures["step"]:
                raise RuntimeError("Router disconnected")
            reply = {
                "type": {"ready": "started", "finish": "finished"}.get(kind, kind),
                "run_id": self.trial["id"],
            }
            if "step" in message:
                reply["step"] = message["step"]
            self.pending = encode_control(reply)

        def recv(self, *, timeout: float) -> bytes:
            return cast(bytes, self.pending)

        def close(self) -> None:
            pass

    config = tmp_path / "client.yaml"
    import yaml

    config.write_text(
        yaml.safe_dump(
            {
                "url": "wss://router.example/ws",
                "token": "router-secret",
                "institution": "test",
                "policy_server_url": "ws://localhost:8000",
                "robot_type": "yam",
                "test": True,
                "cameras": dict.fromkeys(bridge._CAMERAS, "native"),
                "llm_api_keys": {"openai": "unique-provider-secret"},
                "llm_api_base_urls": {"openai": "https://relay.example/v1"},
                "instruction": "pick the box",
                "scene": "table",
                "max_trial_steps": 2,
                "control_hz": 30,
                "evaluation_dir": str(tmp_path / "runs"),
            }
        )
    )
    catalog = tmp_path / "runtime.yaml"
    catalog.write_text(
        yaml.safe_dump(
            {
                "backend": {"name": "auto"},
                "models": [{**MODEL, "endpoint": "inprocess://inspect-agent"}],
            }
        )
    )
    monkeypatch.setitem(
        registry._FACTORIES["embodiment"],
        "bridge_test_yam",
        lambda **kwargs: FakeYam(executions, **kwargs),
    )
    monkeypatch.setattr(evaluation, "EvalAPI", API)
    monkeypatch.setattr(ws_client, "connect", Socket)
    monkeypatch.setattr(bridge, "make_agent", lambda *args: FakeAgent())
    answers = iter(["75", "100", "B", "looks good"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    argv = [
        "colosseum",
        "--client-config",
        str(config),
        "--runtime-config",
        str(catalog),
        "--",
        "--embodiment",
        "bridge_test_yam",
        "--no-prompt",
        "--no-rerun",
        "--no-rerun-save",
        "--no-store-frames",
    ]
    return SimpleNamespace(
        argv=argv,
        config=config,
        catalog=catalog,
        assignment=assignment,
        events=events,
        executions=executions,
        uploads=uploads,
        failures=failures,
        root=tmp_path / "runs/ev_native",
    )


@pytest.mark.parametrize("failure", [None, "result", "upload"])
def test_native_cli_records_exports_publishes_and_resumes_without_rerunning(
    rig: SimpleNamespace, failure: str | None
) -> None:
    import pyarrow.parquet as pq

    if failure:
        rig.failures[failure] = True
    assert main(rig.argv) == (1 if failure else 0)
    if failure:
        assert main(["colosseum", "--client-config", str(rig.config), "--resume", "ev_native"]) == 0
    assert rig.executions.count("reset") == 2
    assert rig.executions.count("step") == 4
    assert rig.executions.count("close") == 2
    assert rig.uploads == ["run_a", "run_b"]
    assert sum(e[:2] == ("POST", "/next") for e in rig.events) == 1
    manifest = json.loads((rig.root / "manifest.json").read_text())
    assert manifest["outcomes"]["A"] == {"success": False, "partial_success": 0.75}
    assert manifest["submitted"] is True
    for run_id in rig.uploads:
        dataset = rig.root / run_id / "lerobot"
        table = pq.read_table(dataset / "data/chunk-000/file-000.parquet").to_pydict()
        assert table["observation.state"][0][6] == pytest.approx(0.4)
        assert table["observation.state"][0][13] == pytest.approx(0.6)
        assert table["action"][0][6] == pytest.approx(0.25)
        assert table["action"][0][13] == pytest.approx(0.75)
        times = table["observation.capture_timestamp"]
        assert len(times) == 2 and 0 <= times[0] < times[1]
        assert len((rig.root / run_id / "frames.jsonl").read_text().splitlines()) == 3
        metadata = json.loads((dataset / "meta/colosseum.json").read_text())
        assert metadata["run_id"] == run_id
        assert metadata["source"] == "inspect-robots CLI"
        assert "observation.cartesian_position" not in table
    for path in rig.root.rglob("*.json"):
        assert "unique-provider-secret" not in path.read_text()
        assert "router-secret" not in path.read_text()


def test_eef_cli_uploads_post_ik_joint_commands_and_preserves_native_actions(
    rig: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyarrow.parquet as pq

    from inspect_robots import registry

    monkeypatch.setitem(
        registry._FACTORIES["embodiment"],
        "yam_eef",
        lambda **kwargs: FakeEEF(rig.executions, **kwargs),
    )
    selected = rig.argv.index("bridge_test_yam")
    argv = [*rig.argv[:selected], "yam_eef", *rig.argv[selected + 1 :]]
    assert main(argv) == 0
    assert rig.uploads == ["run_a", "run_b"]
    for run_id in rig.uploads:
        path = rig.root / run_id
        table = pq.read_table(path / "lerobot/data/chunk-000/file-000.parquet").to_pydict()
        np.testing.assert_allclose(table["action"][0][:6], 0.05)
        np.testing.assert_allclose(table["action"][0][7:13], -0.05)
        assert table["action"][1][:6] != table["observation.state"][1][:6]
        assert table["action"][0][6] == pytest.approx(0.25)
        metadata = json.loads((path / "lerobot/meta/colosseum.json").read_text())
        assert metadata["action_space"] == "joint_position"
        assert metadata["recording_context"]["native_control_mode"] == "eef_abs_pose"
        native = next((path / "inspect").glob("*/actions/*.jsonl"))
        original = next(
            row["action"]
            for line in native.read_text().splitlines()
            if "action" in (row := json.loads(line))
        )
        np.testing.assert_allclose(original[:6], 0)
        assert original != table["action"][0]


def test_missing_ik_telemetry_cannot_be_uploaded_as_joint_actions(tmp_path: Path) -> None:
    calls: list[Any] = []
    session = SimpleNamespace(event=lambda *args, **kwargs: calls.append(args))
    sink = bridge.RecordingSink(tmp_path, None, cast(Any, session), {}, control_mode="eef_abs_pose")
    observation = Observation()
    with pytest.raises(ValueError, match="post-IK"):
        sink.log_step(0, observation, Action(np.zeros(14)), StepResult(observation))
    assert not calls and sink.frame_count == 0


def test_old_eef_plugin_is_rejected_before_reset_or_execution() -> None:
    embodiment = FakeEEF([])
    embodiment.joint_command_info_key = "unsupported"
    with pytest.raises(ValueError, match="telemetry"):
        bridge.CaptureClockEmbodiment(embodiment)


def test_eef_native_flags_preserve_yam_camera_config_and_explicit_overrides(tmp_path: Path) -> None:
    config = tmp_path / "yam.ini"
    config.write_text(
        "[defaults]\nembodiment=yam_arms\n[embodiment.args]\n"
        "left_channel=calibrated_left\nright_channel=calibrated_right\n"
        "top_cam_device=/dev/camera-top\ncontrol_interface=joints\n"
        "auto_start=true\nzero_gravity_mode=false\neef_orientation=false\n"
    )
    options = build_parser().parse_args(
        [
            "colosseum",
            "--client-config",
            "client.yaml",
            "--",
            "--config",
            str(config),
            "--embodiment",
            "yam_eef",
            "-E",
            "eef_orientation=True",
        ]
    )
    from inspect_robots.cli import _parse_kvs

    native = bridge.native_args(options, {"instruction": "pick", "max_steps": 20}, tmp_path)
    values = _parse_kvs(native.embodiment_args)
    assert values["left_channel"] == "calibrated_left"
    assert values["right_channel"] == "calibrated_right"
    assert values["top_cam_device"] == "/dev/camera-top"
    assert values["control_interface"] == "eef_pos"
    assert values["auto_start"] is False
    assert values["zero_gravity_mode"] is False
    assert values["eef_orientation"] is True
    assert values["control_hz"] == 30


def test_router_disconnect_does_not_submit_failed_trial(rig: SimpleNamespace) -> None:
    rig.failures["step"] = True
    assert main(rig.argv) == 1
    assert not rig.uploads
    assert not any(e[:2] == ("POST", "/assignments/ev_native/result") for e in rig.events)
    assert not (rig.root / "run_a/bridge-trial.json").exists()
    assert json.loads((rig.root / "run_a/recording-status.json").read_text())["complete"] is False
    assert rig.executions[-1] == "close"


def test_deferred_upload_can_be_resumed_without_rerunning(rig: SimpleNamespace) -> None:
    assert main([*rig.argv[:5], "--skip-upload", *rig.argv[5:]]) == 0
    assert not rig.uploads
    assert json.loads((rig.root / "manifest.json").read_text())["submitted"] is True
    assert main(["colosseum", "--client-config", str(rig.config), "--resume", "ev_native"]) == 0
    assert rig.uploads == ["run_a", "run_b"]
    assert rig.executions.count("reset") == 2


def test_local_review_defers_results_and_uploads_until_explicit_submit(
    rig: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert main([*rig.argv[:5], "--defer-submit", *rig.argv[5:]]) == 0
    manifest = rig.root / "manifest.json"
    saved = json.loads(manifest.read_text())
    assert saved["submission_deferred"] is True
    assert not saved.get("submitted")
    assert saved["result"]["preference"] == "B"
    assert rig.assignment["state"] == "pending"
    assert not rig.uploads
    assert not any(e[:2] == ("POST", "/assignments/ev_native/result") for e in rig.events)
    for run in ("run_a", "run_b"):
        assert (rig.root / run / "lerobot/meta/info.json").is_file()

    def unexpected_input(prompt: str) -> str:
        raise AssertionError("Saved evaluation must not prompt during submission")

    monkeypatch.setattr("builtins.input", unexpected_input)
    resume = ["colosseum", "--client-config", str(rig.config), "--resume", "ev_native"]
    # Ordinary recovery must preserve the user's decision to defer publication.
    assert main(resume) == 0
    assert not rig.uploads
    assert not any(e[:2] == ("POST", "/assignments/ev_native/result") for e in rig.events)
    assert main([*resume, "--submit-only"]) == 0
    assert rig.executions.count("reset") == 2
    assert rig.uploads == ["run_a", "run_b"]
    saved = json.loads(manifest.read_text())
    assert saved["submitted"] is True and saved["submission_deferred"] is False


def test_submit_only_rejects_unfinished_runs_without_executing_hardware(
    rig: SimpleNamespace,
) -> None:
    rig.root.mkdir(parents=True)
    (rig.root / "manifest.json").write_text(
        json.dumps({"bridge": "inspect-robots-v1", "assignment": rig.assignment})
    )
    assert (
        main(
            [
                "colosseum",
                "--client-config",
                str(rig.config),
                "--resume",
                "ev_native",
                "--submit-only",
            ]
        )
        == 1
    )
    assert not rig.executions and not rig.uploads
    assert not any(e[:2] == ("POST", "/assignments/ev_native/result") for e in rig.events)


@pytest.mark.parametrize("failure", ["result", "upload"])
def test_review_submission_failure_can_retry_without_rerunning(
    rig: SimpleNamespace, failure: str
) -> None:
    assert main([*rig.argv[:5], "--defer-submit", *rig.argv[5:]]) == 0
    rig.failures[failure] = True
    submit = [
        "colosseum",
        "--client-config",
        str(rig.config),
        "--submit-only",
        "--resume",
        "ev_native",
    ]
    assert main(submit) == 1
    saved = json.loads((rig.root / "manifest.json").read_text())
    assert saved["submission_deferred"] is (failure == "result")
    assert main(submit) == 0
    assert rig.executions.count("reset") == 2
    assert rig.uploads == ["run_a", "run_b"]


def test_excel_distinguishes_local_review_and_published_results(
    rig: SimpleNamespace, tmp_path: Path
) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    target = tmp_path / "review.xlsx"
    assert main([*rig.argv[:5], "--defer-submit", "--excel", str(target), *rig.argv[5:]]) == 0

    def statuses() -> list[tuple[bool, bool]]:
        workbook = openpyxl.load_workbook(target)
        try:
            rows = list(workbook["Colosseum Evals"].values)
            headers = rows[0]
            return [
                (row[headers.index("Result submitted")], row[headers.index("Dataset uploaded")])
                for row in rows[1:]
            ]
        finally:
            workbook.close()

    assert statuses() == [(False, False), (False, False)]
    assert (
        main(
            [
                "colosseum",
                "--client-config",
                str(rig.config),
                "--submit-only",
                "--resume",
                "ev_native",
            ]
        )
        == 0
    )
    assert statuses() == [(True, True), (True, True)]
    assert rig.executions.count("reset") == 2


@pytest.mark.parametrize(
    "flags",
    [
        ["--submit-only"],
        ["--submit-only", "--resume", "ev_native", "--defer-submit"],
        ["--submit-only", "--resume", "ev_native", "--dry-run"],
        ["--submit-only", "--resume", "ev_native", "--export-excel-only"],
    ],
)
def test_invalid_submit_flags_never_contact_router(rig: SimpleNamespace, flags: list[str]) -> None:
    assert main(["colosseum", "--client-config", str(rig.config), *flags]) == 1
    assert not rig.events and not rig.executions and not rig.uploads


def test_recording_disabled_still_submits_scores_without_uploads(rig: SimpleNamespace) -> None:
    import yaml

    config = yaml.safe_load(rig.config.read_text())
    config["recording"] = False
    rig.config.write_text(yaml.safe_dump(config))
    rig.assignment["recording"] = False
    assert main(rig.argv) == 0
    assert not rig.uploads
    assert not list(rig.root.glob("*/lerobot"))
    assert json.loads((rig.root / "manifest.json").read_text())["submitted"] is True


def test_eval_updates_excel_and_resume_reuses_target_without_duplicate_rows(
    rig: SimpleNamespace,
) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    from inspect_robots.colosseum_excel import HEADERS, SHEET

    target = rig.root.parent / "results.xlsx"
    assert main([*rig.argv[:5], "--excel", str(target), *rig.argv[5:]]) == 0
    assert main(["colosseum", "--client-config", str(rig.config), "--resume", "ev_native"]) == 0
    assert rig.executions.count("reset") == 2
    book = openpyxl.load_workbook(target)
    try:
        sheet = book[SHEET]
        assert sheet.max_row == 3
        assert sheet.cell(2, HEADERS.index("Preference") + 1).value == "B"
        assert sheet.cell(2, HEADERS.index("Score (%)") + 1).value == 75
        assert all(sheet.cell(row, HEADERS.index("Dataset uploaded") + 1).value for row in (2, 3))
    finally:
        book.close()


def test_excel_lock_keeps_router_uploads_and_allows_local_only_retry(
    rig: SimpleNamespace,
) -> None:
    pytest.importorskip("openpyxl")
    target = rig.root.parent / "results.xlsx"
    target.parent.mkdir(parents=True)
    lock = target.with_name(".~lock.results.xlsx#")
    lock.write_text("editor open")
    assert main([*rig.argv[:5], "--excel", str(target), *rig.argv[5:]]) == 1
    assert rig.uploads == ["run_a", "run_b"]
    saved = json.loads((rig.root / "manifest.json").read_text())
    assert saved["submitted"] is True and saved["excel_pending"] is True
    assert not target.exists()
    lock.unlink()
    calls = len(rig.events)
    assert main(["colosseum", "--client-config", str(rig.config), "--export-excel-only"]) == 0
    assert len(rig.events) == calls
    assert rig.executions.count("reset") == 2
    assert target.exists()
    saved = json.loads((rig.root / "manifest.json").read_text())
    assert saved["excel_pending"] is False


def test_unknown_assigned_model_rejected_before_hardware(
    rig: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual = bridge.read_catalog
    monkeypatch.setattr(
        bridge, "read_catalog", lambda path: [{**m, "revision": "wrong"} for m in actual(path)]
    )
    assert main(rig.argv) == 1
    assert not rig.executions
    assert not rig.uploads


@pytest.mark.parametrize("explicit_type", [True, False])
def test_router_vla_is_rejected_before_agent_robot_or_readiness(
    rig: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    explicit_type: bool,
) -> None:
    if explicit_type:
        monkeypatch.setitem(MODEL, "model_type", "vla")
    else:
        monkeypatch.delitem(MODEL, "model_type")
    monkeypatch.setitem(MODEL, "url", "https://huggingface.co/NUSMAGIC/GR00T-N1.7-MolmoAct2-YAM")

    def forbidden_agent(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("VLA must never initialize an agent")

    monkeypatch.setattr(bridge, "make_agent", forbidden_agent)
    assert main(rig.argv) == 1
    assert "Router assigned a VLA model" in capsys.readouterr().err
    assert not rig.executions and not rig.uploads
    assert not any(e[0] in {"ready", "start", "step", "finish"} for e in rig.events)
    assert rig.assignment["state"] == "pending"
    saved = json.loads((rig.root / "router-models.json").read_text())
    assert saved["run_a"]["model"].get("model_type", "vla") == "vla"
    assert saved["run_a"]["side"] == "A"
    assert not (rig.root / "run_a").exists()


@pytest.mark.parametrize("change", [{"model_type": "vla"}, {"name": "gpt-unlisted-model"}])
def test_runtime_rejects_models_outside_three_llms_before_assignment(
    rig: SimpleNamespace, change: dict[str, str]
) -> None:
    import yaml

    runtime = yaml.safe_load(rig.catalog.read_text())
    runtime["models"][0].update(change)
    rig.catalog.write_text(yaml.safe_dump(runtime))
    assert main(rig.argv) == 1
    assert not rig.events and not rig.executions and not rig.uploads


def test_dry_run_never_contacts_router_or_hardware(rig: SimpleNamespace) -> None:
    assert main([*rig.argv[:5], "--dry-run", *rig.argv[5:]]) == 0
    assert not rig.events and not rig.executions


def test_assignment_validation_failure_does_not_suggest_resuming_missing_id(
    rig: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from colosseum_client import evaluation

    class RejectingAPI:
        def __init__(self, config: Any) -> None:
            pass

        def request(self, method: str, path: str, body: Any = None) -> Any:
            assert (method, path) == ("POST", "/next")
            raise RuntimeError("Evaluation API 422: Unprocessable Entity")

        def close(self) -> None:
            pass

    monkeypatch.setattr(evaluation, "EvalAPI", RejectingAPI)
    assert main(rig.argv) == 1
    assert not rig.executions and not rig.uploads
    assert not (rig.root / "manifest.json").exists()
    assert "No evaluation assignment was obtained" in capsys.readouterr().err


def test_ee_flags_and_url_mismatch_rejected_before_assignment(rig: SimpleNamespace) -> None:
    assert main([*rig.argv, "-E", "control_interface=eef_pos"]) == 1
    assert not rig.events
    import yaml

    value = yaml.safe_load(rig.config.read_text())
    value.pop("llm_api_base_urls")
    rig.config.write_text(yaml.safe_dump(value))
    assert main(rig.argv) == 1
    assert not rig.events


def test_pending_evaluation_is_not_implicitly_discarded(rig: SimpleNamespace) -> None:
    rig.root.mkdir(parents=True)
    (rig.root / "manifest.json").write_text(json.dumps({"bridge": "inspect-robots-v1"}))
    assert main(rig.argv) == 1
    assert not rig.events


def test_receipt_clock_preserves_existing_sensor_timestamps() -> None:
    observation = Observation(state_time=123, image_times={"top_cam": 122}, extra={"depth": True})
    stamped = bridge.CaptureClockEmbodiment.stamp(observation)
    assert stamped.state_time == 123 and stamped.image_times == {"top_cam": 122}
    assert stamped.extra["depth"] is True
    assert "_colosseum_received_at" not in observation.extra


def test_recording_rejects_missing_state_without_fabrication(tmp_path: Path) -> None:
    sink = bridge.RecordingSink(tmp_path, None, cast(Any, None), {"head_image": "top_cam"})
    with pytest.raises(ValueError, match="measured"):
        sink.observation(Observation(images={"top_cam": np.zeros((2, 2, 3), np.uint8)}))


def test_catalog_key_url_declaration_and_no_credential_leak() -> None:
    config = RobotClientConfig(
        url="wss://router.example",
        token="secret",
        cameras={},
        policy_server_url="ws://localhost:8000",
        llm_api_keys={"openai": "private"},
        llm_api_base_urls={"openai": "https://relay.example/v1/"},
    )
    bridge.validate_catalog_keys(config, [MODEL])
    assert config.llm_api_urls == ["https://api.openai.com/v1"]
    assert "private" not in repr(config)
    with pytest.raises(ValueError, match="configured"):
        bridge.validate_catalog_keys(dataclasses.replace(config, llm_api_keys={}), [MODEL])


@pytest.fixture
def account_cli(tmp_path: Path) -> Path:
    path = tmp_path / "fake-claude"
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "assert sys.argv[-2:] == ['auth', 'status'], 'No inference permitted'\n"
        "print(json.dumps({'loggedIn': True, 'authMethod': 'claude.ai'}))\n"
    )
    path.chmod(0o755)
    return path


def test_claude_account_constructs_real_agent_without_api_keys(
    account_cli: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from inspect_robots import registry

    config = RobotClientConfig(
        url="wss://router.example",
        token="router-private",
        cameras={},
        llm_account_logins={"anthropic": "claude-code"},
    )
    model = {
        **MODEL,
        "name": "claude-opus-5-5",
        "url": "https://api.anthropic.com/v1",
        "bridge_options": {"claude_command": str(account_cli), "effort": "low"},
    }
    monkeypatch.setenv("ANTHROPIC_API_KEY", "wrong-api-key")
    monkeypatch.setenv("OPENAI_API_KEY", "other-provider-key")
    monkeypatch.setenv("UNUSUAL_ROUTER_TOKEN", "router-private")
    original = registry.resolve
    captured: dict[str, Any] = {}

    def resolve(kind: str, name: str, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return original(kind, name, **kwargs)

    monkeypatch.setattr(registry, "resolve", resolve)
    bridge.validate_catalog_keys(config, [model])
    agent = bridge.make_agent(model, config, {})
    try:
        assert captured["model"] == "anthropic/claude-opus-5-5"
        assert captured["wire"] == "claude-code"
        assert not {"base_url", "api_key_env", "transport"} & captured.keys()
        assert (
            not {"ANTHROPIC_API_KEY", "OPENAI_API_KEY", "UNUSUAL_ROUTER_TOKEN"}
            & captured["env"].keys()
        )
        assert agent.config.wire == "claude-code"
        assert "private" not in json.dumps(dataclasses.asdict(agent.config))
    finally:
        agent._client.close()
    with pytest.raises(ValueError, match="official"):
        bridge.make_agent({**model, "url": "https://relay.example"}, config, {})


def test_account_login_pairwise_records_auth_and_defers_submission(
    rig: SimpleNamespace, account_cli: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml

    monkeypatch.setitem(MODEL, "name", "claude-opus-5-5")
    monkeypatch.setitem(MODEL, "url", "https://api.anthropic.com/v1")
    runtime = yaml.safe_load(rig.catalog.read_text())
    runtime["models"] = [
        {
            **MODEL,
            "endpoint": "inprocess://inspect-agent",
            "backend_options": {"claude_command": str(account_cli)},
        }
    ]
    rig.catalog.write_text(yaml.safe_dump(runtime))
    config = yaml.safe_load(rig.config.read_text())
    config["llm_api_keys"] = {}
    config["llm_account_logins"] = {"anthropic": "claude-code"}
    config["llm_api_base_urls"] = {"anthropic": "https://api.anthropic.com/v1"}
    rig.config.write_text(yaml.safe_dump(config))
    assert main([*rig.argv[:5], "--defer-submit", *rig.argv[5:]]) == 0
    assert not rig.uploads
    assert not any(e[:2] == ("POST", "/assignments/ev_native/result") for e in rig.events)
    context = json.loads((rig.root / "run_a/recording-context.json").read_text())
    assert context["native_llm_auth"] == "claude-code"
    assert (
        main(
            [
                "colosseum",
                "--client-config",
                str(rig.config),
                "--submit-only",
                "--resume",
                "ev_native",
            ]
        )
        == 0
    )
    assert rig.executions.count("reset") == 2
    assert rig.uploads == ["run_a", "run_b"]


@pytest.mark.parametrize("status", ["logged-out", "console"])
def test_account_login_preflight_rejects_missing_subscription(
    account_cli: Path, status: str
) -> None:
    from inspect_robots.errors import ConfigError

    script = account_cli.read_text()
    script = (
        script.replace("True", "False")
        if status == "logged-out"
        else script.replace("claude.ai", "api_key")
    )
    account_cli.write_text(script)
    config = RobotClientConfig(
        url="wss://router.example",
        token="test",
        cameras={},
        llm_account_logins={"anthropic": "claude-code"},
    )
    model = {
        **MODEL,
        "name": "claude-opus-5-5",
        "url": "https://api.anthropic.com/v1",
        "bridge_options": {"claude_command": str(account_cli)},
    }
    with pytest.raises(ConfigError, match="claude auth login"):
        bridge.validate_catalog_keys(config, [model])


def test_account_login_rejects_router_relay_identity_before_hardware(
    rig: SimpleNamespace,
    account_cli: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import yaml

    monkeypatch.setitem(MODEL, "name", "claude-opus-5-5")
    runtime = yaml.safe_load(rig.catalog.read_text())
    runtime["models"] = [
        {
            **MODEL,
            "url": "https://api.anthropic.com/v1",
            "endpoint": "inprocess://inspect-agent",
            "backend_options": {"claude_command": str(account_cli)},
        }
    ]
    rig.catalog.write_text(yaml.safe_dump(runtime))
    config = yaml.safe_load(rig.config.read_text())
    config["llm_api_keys"] = {}
    config["llm_account_logins"] = {"anthropic": "claude-code"}
    config["llm_api_base_urls"] = {"anthropic": "https://api.anthropic.com/v1"}
    rig.config.write_text(yaml.safe_dump(config))
    assert main(rig.argv) == 1
    assert "Claude Code login cannot replace a relay key" in capsys.readouterr().err
    assert not rig.executions and not rig.uploads


def test_parser_keeps_native_flags_after_separator() -> None:
    args = build_parser().parse_args(
        [
            "colosseum",
            "--client-config",
            "client.yaml",
            "--runtime-config",
            "runtime.yaml",
            "--",
            "--config",
            "rig.ini",
            "-P",
            "max_speed_frac=0.03",
        ]
    )
    native = bridge.native_args(args, {"instruction": "pick", "max_steps": 20}, Path("logs"))
    assert native.config == "rig.ini"
    assert native.policy_args == ["max_speed_frac=0.03"]
    assert native.max_steps == 20 and native.epochs == 1


def test_native_action_chunks_honor_registered_horizon_and_control_rate() -> None:
    agent = FakeAgent()
    session: Any = SimpleNamespace(preparation={"model": {**MODEL, "max_horizon": 1}}, failure=None)
    policy = bridge.ReportingPolicy(agent, session, None, {})
    assert len(policy.act(Observation()).actions) == 1
    session.preparation["model"]["control_hz"] = 15
    with pytest.raises(ValueError, match="frequency"):
        policy.act(Observation())


@pytest.mark.parametrize(
    "name,provider,wire",
    [
        ("gpt-6-astra", "openai", "responses"),
        ("grok-4.7", "xai", "responses"),
        ("claude-opus-5-5", "anthropic", "messages"),
    ],
)
def test_real_agent_is_constructed_with_assigned_identity_and_isolated_keys(
    monkeypatch: pytest.MonkeyPatch, name: str, provider: str, wire: str
) -> None:
    pytest.importorskip("inspect_robots_agent")
    import httpx

    from inspect_robots import registry

    original = registry.resolve
    captured: dict[str, Any] = {}
    transport = httpx.MockTransport(
        lambda request: pytest.fail("Construction must not call an API")
    )

    def resolve(kind: str, component: str, **kwargs: Any) -> Any:
        captured.update(kwargs)
        kwargs["transport"] = transport
        return original(kind, component, **kwargs)

    monkeypatch.setattr(registry, "resolve", resolve)
    config = SimpleNamespace(llm_api_keys={provider: "session-secret"}, llm_account_logins={})
    model = {**MODEL, "name": name, "bridge_options": {"effort": "low"}}
    agent = bridge.make_agent(model, config, {"max_speed_frac": 0.03})
    try:
        assert agent.config.model == name
        assert agent.config.wire == wire
        assert agent.config.base_url == MODEL["url"]
        assert agent.config.max_speed_frac == 0.03
        assert list(captured["env"].values()) == ["session-secret"]
        assert "session-secret" not in json.dumps(dataclasses.asdict(agent.config))
        assert agent.config.wire_capture is False
    finally:
        agent._client.close()


def test_grok_request_uses_native_model_id_at_explicit_xai_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("inspect_robots_agent")
    import httpx

    from inspect_robots import registry

    original = registry.resolve
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url == httpx.URL("https://api.x.ai/v1/responses")
        assert request.headers["authorization"] == "Bearer session-secret"
        body = json.loads(request.content)
        if body["model"] != "grok-4.7":
            return httpx.Response(404, json={"error": "model not found"})
        return httpx.Response(
            200,
            json={
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ready"}],
                    }
                ]
            },
        )

    def resolve(kind: str, component: str, **kwargs: Any) -> Any:
        kwargs["transport"] = httpx.MockTransport(respond)
        return original(kind, component, **kwargs)

    monkeypatch.setattr(registry, "resolve", resolve)
    config = SimpleNamespace(llm_api_keys={"xai": "session-secret"}, llm_account_logins={})
    model = {
        **MODEL,
        "name": "grok-4.7",
        "url": "https://api.x.ai/v1",
        "bridge_options": {},
    }
    agent = bridge.make_agent(model, config, {})
    try:
        assert not requests
        result = agent._client.complete([{"role": "user", "content": "hello"}], [])
        assert result.content == "ready"
        assert len(requests) == 1
    finally:
        agent._client.close()


@pytest.mark.parametrize(
    "url", ["http://relay.example", "https://u:p@relay.example", "https://relay.example?key=secret"]
)
def test_client_rejects_credential_bearing_api_urls(url: str) -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        RobotClientConfig(
            url="wss://router.example", token="t", cameras={}, llm_api_base_urls={"openai": url}
        )


@pytest.fixture
def local_rig(rig: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    import yaml

    from inspect_robots import registry

    config = yaml.safe_load(rig.config.read_text())
    config.update(
        test=False,
        evaluator="operator",
        llm_api_keys={"openai": "gpt-private", "xai": "grok-private"},
        llm_api_base_urls={"openai": MODEL["url"], "xai": MODEL["url"]},
    )
    rig.config.write_text(yaml.safe_dump(config))
    runtime = yaml.safe_load(rig.catalog.read_text())
    runtime["models"].append(
        {**MODEL, "name": "grok-4.7", "revision": "4.7", "endpoint": "inprocess://inspect-agent"}
    )
    rig.catalog.write_text(yaml.safe_dump(runtime))

    def physical_fake(**kwargs: Any) -> Any:
        robot = FakeYam(rig.executions, **kwargs)
        robot.info = dataclasses.replace(robot.info, is_simulated=False)
        return robot

    monkeypatch.setitem(registry._FACTORIES["embodiment"], "bridge_test_yam", physical_fake)
    chosen = []

    def make_agent(model: dict[str, Any], *args: Any) -> Any:
        chosen.append(model["name"])
        return FakeAgent()

    monkeypatch.setattr(bridge, "make_agent", make_agent)
    answers = iter(["3", "4", "B", "reviewed", "clear"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    rig.local_argv = [
        *rig.argv[:5],
        "--local-pair",
        "--model-a",
        "gpt-6-astra",
        "--model-b",
        "grok-4.7",
        "--defer-submit",
        *[a for a in rig.argv[5:] if a != "--no-prompt"],
    ]
    rig.chosen = chosen
    return rig


def _collect_local(rig: SimpleNamespace) -> tuple[Path, dict[str, Any]]:
    assert main(rig.local_argv) == 0
    manifests = list(rig.root.parent.glob("local_*/manifest.json"))
    manifest = sorted(manifests)[-1]
    return manifest, json.loads(manifest.read_text())


def test_explicit_local_pair_collects_real_data_without_router_assignment(
    local_rig: SimpleNamespace,
    tmp_path: Path,
) -> None:
    local_rig.local_argv = [
        *local_rig.local_argv[:5],
        "--excel",
        str(tmp_path / "local.xlsx"),
        *local_rig.local_argv[5:],
    ]
    manifest, saved = _collect_local(local_rig)
    assert local_rig.chosen == ["gpt-6-astra", "grok-4.7"]
    assert not local_rig.events and not local_rig.uploads
    assert local_rig.executions.count("reset") == 2
    assert saved["bridge"] == "inspect-robots-local-v1"
    assert saved["submission_deferred"] is True and not saved.get("submitted")
    for run in saved["assignment"]["runs"]:
        marker = json.loads((manifest.parent / run["id"] / "bridge-trial.json").read_text())
        assert marker["native_complete"] and marker["local_finished"]
        assert not marker["router_finished"]
        assert (manifest.parent / run["id"] / "lerobot/eval_trials.csv").is_file()
        context = json.loads((manifest.parent / run["id"] / "recording-context.json").read_text())
        assert context["model_selection"] == "local"
    text = manifest.read_text()
    assert "private" not in text and "router-secret" not in text
    resume = [
        "colosseum",
        "--client-config",
        str(local_rig.config),
        "--resume",
        saved["assignment"]["id"],
    ]
    assert main(resume) == 0  # Defer decision persists without constructing a new agent.
    assert local_rig.executions.count("reset") == 2 and not local_rig.events


def _validate_public_upload_contract(value: Any, schema: dict[str, Any]) -> None:
    contract = json.loads(
        (Path(__file__).parent / "fixtures/colosseum_upload_contract.json").read_text()
    )
    if "$ref" in schema:
        _validate_public_upload_contract(
            value, contract["components"]["schemas"][schema["$ref"].split("/")[-1]]
        )
        return
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            try:
                _validate_public_upload_contract(value, option)
            except AssertionError:
                continue
            return
        raise AssertionError("No allowed schema variant")
    kind = schema.get("type")
    if kind == "object":
        assert isinstance(value, dict)
        properties = schema.get("properties", {})
        assert set(schema.get("required", [])) <= set(value)
        if schema.get("additionalProperties") is False:
            assert set(value) <= set(properties)
        assert len(value) >= schema.get("minProperties", 0)
        assert len(value) <= schema.get("maxProperties", 10**9)
        for name, item in value.items():
            if name in properties:
                _validate_public_upload_contract(item, properties[name])
            elif isinstance(schema.get("additionalProperties"), dict):
                _validate_public_upload_contract(item, schema["additionalProperties"])
    elif kind == "array":
        assert isinstance(value, list)
        assert schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 10**9)
        for item in value:
            _validate_public_upload_contract(item, schema.get("items", {}))
    elif kind == "string":
        assert isinstance(value, str)
        assert schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", 10**9)
        if "pattern" in schema:
            assert re.fullmatch(schema["pattern"], value)
    elif kind in {"number", "integer"}:
        assert type(value) in ({int} if kind == "integer" else {int, float})
        assert schema.get("minimum", -(10**9)) <= value <= schema.get("maximum", 10**9)
    elif kind == "boolean":
        assert type(value) is bool
    elif kind == "null":
        assert value is None
    if "enum" in schema:
        assert value in schema["enum"]
    if "const" in schema:
        assert value == schema["const"]


class FakeLocalImport:
    def __init__(self) -> None:
        self.evaluations: list[dict[str, Any]] = []
        self.datasets: dict[str, dict[str, Any]] = {}
        self.files: dict[tuple[str, str], bytes] = {}
        self.videos: list[bytes] = []
        self.events: list[tuple[str, str]] = []
        self.fail_file = False
        self.lose_create_reply = False

    def request(self, method: str, path: str, body: Any = None) -> Any:
        self.events.append((method, path))
        parsed = urlsplit(path)
        route = parsed.path
        if route == "/api/list_ab_evaluations":
            return self.evaluations
        if route == "/api/evaluations":
            _validate_public_upload_contract(
                body, {"$ref": "#/components/schemas/EvaluationCreateRequest"}
            )
            assert method == "POST" and body["policyA"] == "gpt-6-astra-yam@rig-v1"
            assert body["policyB"] == "grok-4.7-yam@4.7"
            assert body["partialSuccessA"] == 0.75 and body["partialSuccessB"] == 1
            record = {**body, "id": "manual_eval_1", "finalized": False}
            self.evaluations.append(record)
            if self.lose_create_reply:
                self.lose_create_reply = False
                raise RuntimeError("POST response lost")
            return record
        if route == "/api/datasets":
            _validate_public_upload_contract(body, {"$ref": "#/components/schemas/ImportManifest"})
            assert body["model"]["name"] in bridge._LLM_MODELS
            assert body["model"]["repo_id"] == "" and body["model"]["revision"] == ""
            assert body["scoring"] == "score-0-4-success-4-v1"
            existing = next(
                (
                    key
                    for key, value in self.datasets.items()
                    if value["submission_id"] == body["submission_id"]
                ),
                None,
            )
            identity = existing or f"ds_{len(self.datasets) + 1}"
            self.datasets[identity] = body
            return {"id": identity}
        if route.startswith("/api/datasets/"):
            identity = route.split("/")[3]
            if route.endswith("/upload-url"):
                file = parse_qs(parsed.query)["path"][0]
                return {
                    "storage": "local",
                    "dataset_id": identity,
                    "path": file,
                    "complete": (identity, file) in self.files,
                }
            if route.endswith("/complete"):
                for file in self.datasets[identity]["files"]:
                    raw = self.files[identity, file["path"]]
                    assert len(raw) == file["size"]
                    assert (
                        base64.b64encode(hashlib.sha256(raw).digest()).decode()
                        == file["checksum_sha256"]
                    )
                return {"state": "ready"}
            if route.endswith("/publish"):
                return {"state": "published"}
        if route.endswith("/videos/upload-urls"):
            _validate_public_upload_contract(
                body, {"$ref": "#/components/schemas/VideoUploadRequest"}
            )
            return {
                "uploads": [
                    {**v, "url": "https://storage.example/" + v["side"] + "/" + v["camera"]}
                    for v in body["videos"]
                ]
            }
        if route.endswith("/result"):
            _validate_public_upload_contract(
                body, {"$ref": "#/components/schemas/EvaluationResultUpdateRequest"}
            )
            assert method == "PATCH" and body["finalize"] is True
            self.evaluations[0].update(finalized=True, notes=body["notes"])
            return self.evaluations[0]
        raise AssertionError(f"Unexpected API route: {method} {route}")

    def put(self, target: dict[str, Any], path: Path, local_route: str | None = None) -> None:
        if target.get("complete"):
            return
        if self.fail_file:
            self.fail_file = False
            raise RuntimeError("File upload interrupted")
        raw = path.read_bytes()
        if target.get("storage") == "local":
            self.files[target["dataset_id"], target["path"]] = raw
        else:
            assert path.suffix == ".mp4"
            self.videos.append(raw)

    def close(self) -> None:
        pass


@pytest.mark.parametrize("failure", [None, "file", "create-reply"])
def test_local_pair_publishes_scores_videos_and_verified_datasets_without_next(
    local_rig: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    from inspect_robots import colosseum_import

    manifest, local = _collect_local(local_rig)
    server = FakeLocalImport()
    monkeypatch.setattr(colosseum_import, "LocalUploadAPI", lambda config: server)
    server.fail_file = failure == "file"
    server.lose_create_reply = failure == "create-reply"
    submit = [
        "colosseum",
        "--client-config",
        str(local_rig.config),
        "--submit-only",
        "--resume",
        local["assignment"]["id"],
    ]
    assert main(submit) == (1 if failure else 0)
    if failure:
        assert main(submit) == 0
    assert local_rig.executions.count("reset") == 2
    assert len(server.evaluations) == 1 and server.evaluations[0]["finalized"]
    assert len(server.datasets) == 2 and len(server.videos) == 6
    assert all(
        "/next" not in path and "/api/eval/assignments" not in path for _, path in server.events
    )
    saved = json.loads(manifest.read_text())
    assert saved["submitted"] and not saved["submission_deferred"]
    assert saved["remote_evaluation_id"] == "manual_eval_1"
    assert all(saved["datasets_uploaded"].values())
    assert "router-secret" not in manifest.read_text() and "private" not in manifest.read_text()


def test_local_model_credentials_checked_before_any_execution(
    local_rig: SimpleNamespace,
) -> None:
    import yaml

    config = yaml.safe_load(local_rig.config.read_text())
    config["llm_api_keys"].pop("xai")
    local_rig.config.write_text(yaml.safe_dump(config))
    assert main(local_rig.local_argv) == 1
    assert not local_rig.executions and not local_rig.events and not local_rig.chosen


def test_local_pair_can_select_claude_account_without_anthropic_key(
    local_rig: SimpleNamespace,
    account_cli: Path,
) -> None:
    import yaml

    config = yaml.safe_load(local_rig.config.read_text())
    config["llm_account_logins"] = {"anthropic": "claude-code"}
    config["llm_api_base_urls"]["anthropic"] = "https://api.anthropic.com/v1"
    local_rig.config.write_text(yaml.safe_dump(config))
    runtime = yaml.safe_load(local_rig.catalog.read_text())
    runtime["models"].append(
        {
            **MODEL,
            "name": "claude-opus-5-5",
            "url": "https://api.anthropic.com/v1",
            "endpoint": "inprocess://inspect-agent",
            "backend_options": {"claude_command": str(account_cli)},
        }
    )
    local_rig.catalog.write_text(yaml.safe_dump(runtime))
    index = local_rig.local_argv.index("--model-b") + 1
    local_rig.local_argv[index] = "claude-opus-5-5"
    manifest, saved = _collect_local(local_rig)
    assert local_rig.chosen == ["gpt-6-astra", "claude-opus-5-5"]
    b = saved["assignment"]["runs"][1]
    context = json.loads((manifest.parent / b["id"] / "recording-context.json").read_text())
    assert context["native_llm_auth"] == "claude-code"
    assert not local_rig.events


def test_rejected_manual_creation_can_retry_after_config_fix(
    local_rig: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from inspect_robots import colosseum_import

    manifest, local = _collect_local(local_rig)
    server = FakeLocalImport()
    original = server.request
    rejected = False

    def request(method: str, path: str, body: Any = None) -> Any:
        nonlocal rejected
        if method == "POST" and path == "/api/evaluations" and not rejected:
            rejected = True
            raise colosseum_import.ImportHTTPError(422, "Unprocessable Entity")
        return original(method, path, body)

    monkeypatch.setattr(server, "request", request)
    monkeypatch.setattr(colosseum_import, "LocalUploadAPI", lambda config: server)
    submit = [
        "colosseum",
        "--client-config",
        str(local_rig.config),
        "--submit-only",
        "--resume",
        local["assignment"]["id"],
    ]
    assert main(submit) == 1
    assert not json.loads(manifest.read_text()).get("evaluation_create_pending")
    assert main(submit) == 0
    assert len(server.evaluations) == 1
    assert local_rig.executions.count("reset") == 2


def test_local_pair_dry_run_does_not_allocate_or_execute(local_rig: SimpleNamespace) -> None:
    argv = [*local_rig.local_argv[:5], "--dry-run", *local_rig.local_argv[5:]]
    assert main(argv) == 0
    assert not local_rig.events and not local_rig.executions and not local_rig.chosen
    assert not list(local_rig.root.parent.glob("local_*/manifest.json"))


def _automatic_local_args(args: list[str]) -> list[str]:
    automatic = list(args)
    for flag in ("--model-a", "--model-b"):
        index = automatic.index(flag)
        del automatic[index : index + 2]
    return automatic


def test_automatic_local_pair_selects_distinct_available_models_and_pins_resume(
    local_rig: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inspect_robots import colosseum_local

    local_rig.local_argv = _automatic_local_args(local_rig.local_argv)
    manifest, saved = _collect_local(local_rig)
    assert set(local_rig.chosen) == {"gpt-6-astra", "grok-4.7"}
    assert saved["model_selection"] == "random"
    models = [run["model"]["name"] for run in saved["assignment"]["runs"]]
    assert models == local_rig.chosen
    assert not local_rig.events

    def forbidden_draw() -> Any:
        pytest.fail("Resume must not change the previously selected models")

    monkeypatch.setattr(colosseum_local.random, "SystemRandom", forbidden_draw)
    assert (
        main(
            [
                "colosseum",
                "--client-config",
                str(local_rig.config),
                "--resume",
                saved["assignment"]["id"],
            ]
        )
        == 0
    )
    assert [
        run["model"]["name"] for run in json.loads(manifest.read_text())["assignment"]["runs"]
    ] == models
    assert local_rig.executions.count("reset") == 2


def test_automatic_pair_can_draw_claude_from_all_three_models(
    local_rig: SimpleNamespace,
    account_cli: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import yaml

    from inspect_robots import colosseum_local

    config = yaml.safe_load(local_rig.config.read_text())
    config["llm_account_logins"] = {"anthropic": "claude-code"}
    config["llm_api_base_urls"]["anthropic"] = "https://api.anthropic.com/v1"
    local_rig.config.write_text(yaml.safe_dump(config))
    runtime = yaml.safe_load(local_rig.catalog.read_text())
    runtime["models"].append(
        {
            **MODEL,
            "name": "claude-opus-5-5",
            "url": "https://api.anthropic.com/v1",
            "endpoint": "inprocess://inspect-agent",
            "backend_options": {"claude_command": str(account_cli)},
        }
    )
    local_rig.catalog.write_text(yaml.safe_dump(runtime))

    class Draw:
        def sample(self, available: list[str], count: int) -> list[str]:
            assert set(available) == bridge._LLM_MODELS and count == 2
            return ["claude-opus-5-5", "grok-4.7"]

    monkeypatch.setattr(colosseum_local.random, "SystemRandom", Draw)
    local_rig.local_argv = _automatic_local_args(local_rig.local_argv)
    _collect_local(local_rig)
    assert local_rig.chosen == ["claude-opus-5-5", "grok-4.7"]
    assert not local_rig.events


def test_automatic_pair_with_one_credential_fails_before_any_execution(
    local_rig: SimpleNamespace,
) -> None:
    import yaml

    config = yaml.safe_load(local_rig.config.read_text())
    config["llm_api_keys"].pop("xai")
    local_rig.config.write_text(yaml.safe_dump(config))
    assert main(_automatic_local_args(local_rig.local_argv)) == 1
    assert not local_rig.executions and not local_rig.events


def test_local_pair_requires_zero_or_two_explicit_model_flags(local_rig: SimpleNamespace) -> None:
    args = list(local_rig.local_argv)
    index = args.index("--model-b")
    del args[index : index + 2]
    assert main(args) == 1
    assert not local_rig.executions and not local_rig.events


def test_multiple_local_pairs_can_wait_for_review_without_router(
    local_rig: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    answers = iter(["3", "4", "B", "reviewed", "clear"] * 2)
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    assert main(local_rig.local_argv) == 0
    assert main(local_rig.local_argv) == 0
    manifests = list(local_rig.root.parent.glob("local_*/manifest.json"))
    assert len(manifests) == 2
    assert local_rig.executions.count("reset") == 4
    assert not local_rig.events
    assert all(not json.loads(path.read_text()).get("submitted") for path in manifests)


def test_local_submit_only_rejects_partial_pair_without_upload_or_robot(
    local_rig: SimpleNamespace,
) -> None:
    folder = local_rig.root.parent / "local_incomplete"
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                "bridge": "inspect-robots-local-v1",
                "assignment": {"test": False},
            }
        )
    )
    assert (
        main(
            [
                "colosseum",
                "--client-config",
                str(local_rig.config),
                "--submit-only",
                "--resume",
                folder.name,
            ]
        )
        == 1
    )
    assert not local_rig.events and not local_rig.executions


def test_local_abort_only_closes_local_pair(local_rig: SimpleNamespace) -> None:
    manifest, saved = _collect_local(local_rig)
    assert (
        main(
            [
                "colosseum",
                "--client-config",
                str(local_rig.config),
                "--abort",
                saved["assignment"]["id"],
                "--reason",
                "invalid local recording",
            ]
        )
        == 0
    )
    assert json.loads(manifest.read_text())["aborted"] is True
    assert not local_rig.events and local_rig.executions.count("reset") == 2


def test_local_test_data_cannot_publish_as_physical_evaluation(
    local_rig: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml

    from inspect_robots import colosseum_import

    manifest, saved = _collect_local(local_rig)
    saved["assignment"]["test"] = True
    manifest.write_text(json.dumps(saved))
    config = yaml.safe_load(local_rig.config.read_text())
    config["test"] = True
    local_rig.config.write_text(yaml.safe_dump(config))

    def forbidden_api(config: Any) -> Any:
        pytest.fail("Synthetic local data must never be submitted")

    monkeypatch.setattr(colosseum_import, "LocalUploadAPI", forbidden_api)
    assert (
        main(
            [
                "colosseum",
                "--client-config",
                str(local_rig.config),
                "--submit-only",
                "--resume",
                saved["assignment"]["id"],
            ]
        )
        == 1
    )
    assert not local_rig.events and local_rig.executions.count("reset") == 2


def test_local_upload_adapter_preserves_router_auth_and_isolates_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx

    from inspect_robots.colosseum_import import LocalUploadAPI

    config = RobotClientConfig(
        url="https://router.example".replace("https", "wss"), token="client-secret", cameras={}
    )
    api = LocalUploadAPI(config)
    api.client.close()
    router_requests = []

    def router(request: Any) -> Any:
        router_requests.append(request)
        assert request.headers["authorization"] == "Bearer client-secret"
        return httpx.Response(200, json={"id": "ds_1"})

    original_client = httpx.Client
    api.client = original_client(
        base_url="https://router.example",
        headers={"Authorization": "Bearer client-secret"},
        transport=httpx.MockTransport(router),
    )
    api.api.client = api.client
    storage_requests = []

    def storage(request: Any) -> Any:
        storage_requests.append(request)
        assert "authorization" not in request.headers and "cookie" not in request.headers
        assert request.content == b"real-recording"
        return httpx.Response(200)

    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: original_client(transport=httpx.MockTransport(storage))
    )
    data = tmp_path / "recording.bin"
    data.write_bytes(b"real-recording")
    try:
        assert api.request("POST", "/api/datasets", {"name": "recording"})["id"] == "ds_1"
        api.put({"storage": "local"}, data, "/api/datasets/ds_1/file?path=recording.bin")
        api.put({"url": "https://storage.example/file?signature=signed-secret"}, data)
        assert len(router_requests) == 2 and len(storage_requests) == 1
    finally:
        api.close()


def _review_form(a: str = "75", b: str = "100") -> dict[str, list[str]]:
    return {
        "score_A": [a],
        "score_B": [b],
        "preference": ["B"],
        "feedback": ["reviewed in browser"],
        "confidence": ["clear"],
    }


def test_record_only_pair_reviews_then_uploads_without_rerunning(
    local_rig: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inspect_robots import colosseum_import
    from inspect_robots.colosseum_review import ReviewStore

    local_rig.local_argv.insert(5, "--record-only")
    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("Recording must not score"))
    manifest, local = _collect_local(local_rig)
    identity = local["assignment"]["id"]
    assert local["review"] == "pending" and not local.get("result") and not local["outcomes"]
    assert not local_rig.events and not local_rig.uploads
    for run in local["assignment"]["runs"]:
        folder = manifest.parent / run["id"]
        assert not (folder / "lerobot").exists()
        assert (folder / "head_image/000000.png").is_file()
        assert local["execution"][run["id"]]["eval"]["max_steps"] == 2
    store = ReviewStore(str(local_rig.config))
    assert store.complete(identity, local)
    html = store.evaluation_page(identity).decode()
    assert "Execution settings" in html and "max_llm_calls" in html
    assert html.count("score (0-100)") == 2 and 'max="100"' in html
    assert "private" not in html and "router-secret" not in html
    preview = store.preview(identity, local["assignment"]["runs"][0]["id"], "head_image")
    assert preview is not None and preview.stat().st_size > 0
    assert not local_rig.events
    store.save(identity, _review_form())
    assert not local_rig.events
    assert not any(manifest.parent.glob("*/lerobot"))
    server = FakeLocalImport()
    monkeypatch.setattr(colosseum_import, "LocalUploadAPI", lambda config: server)
    assert (
        main(
            [
                "colosseum",
                "--client-config",
                str(local_rig.config),
                "--submit-only",
                "--resume",
                identity,
            ]
        )
        == 0
    )
    assert local_rig.executions.count("reset") == 2
    assert len(server.datasets) == 2 and len(server.videos) == 6
    notes = json.loads(server.evaluations[0]["notes"])
    assert set(notes["execution"]) == {run["id"] for run in local["assignment"]["runs"]}
    for run in local["assignment"]["runs"]:
        metadata = json.loads(
            (manifest.parent / run["id"] / "lerobot/meta/colosseum.json").read_text()
        )
        assert metadata["recording_context"]["native_execution"]["eval"]["max_steps"] == 2


def test_record_only_router_pair_scores_in_review_before_explicit_submission(
    rig: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inspect_robots.colosseum_review import ReviewStore

    rig.argv.insert(5, "--record-only")
    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("Recording must not score"))
    assert main(rig.argv) == 0
    local = json.loads((rig.root / "manifest.json").read_text())
    assert all(run["state"] == "finished" for run in local["assignment"]["runs"])
    assert not local["outcomes"] and not local.get("result")
    assert not rig.uploads and not any(event[1].endswith("/result") for event in rig.events)
    assert not any(rig.root.glob("*/lerobot"))
    store = ReviewStore(str(rig.config))
    store.save("ev_native", _review_form("75", "100"))
    assert (
        main(
            [
                "colosseum",
                "--client-config",
                str(rig.config),
                "--submit-only",
                "--resume",
                "ev_native",
            ]
        )
        == 0
    )
    assert rig.uploads == ["run_a", "run_b"]
    assert rig.executions.count("reset") == 2


def test_review_rescores_old_exports_preserving_video_and_robot_evidence(
    local_rig: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inspect_robots import colosseum_import
    from inspect_robots.colosseum_review import ReviewStore

    answers = iter(["1", "2", "B", "old score", "close"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    manifest, local = _collect_local(local_rig)
    files = {
        p: p.read_bytes()
        for p in manifest.parent.rglob("*")
        if p.is_file() and (p.suffix in {".mp4", ".parquet"} or p.name == "frames.jsonl")
    }
    store = ReviewStore(str(local_rig.config))
    run = local["assignment"]["runs"][0]
    preview = store.preview(local["assignment"]["id"], run["id"], "head_image")
    assert preview is not None and "lerobot" in preview.parts
    store.save(local["assignment"]["id"], _review_form())
    dataset = manifest.parent / run["id"] / "lerobot"
    assert json.loads((dataset / "meta/colosseum.json").read_text())["outcome"]["score_0_4"] == 1
    server = FakeLocalImport()
    monkeypatch.setattr(colosseum_import, "LocalUploadAPI", lambda config: server)
    assert (
        main(
            [
                "colosseum",
                "--client-config",
                str(local_rig.config),
                "--submit-only",
                "--resume",
                local["assignment"]["id"],
            ]
        )
        == 0
    )
    assert json.loads((dataset / "meta/colosseum.json").read_text())["outcome"]["score_0_4"] == 3
    assert ",3," in (dataset / "eval_trials.csv").read_text()
    assert all(path.read_bytes() == content for path, content in files.items())
    assert local_rig.executions.count("reset") == 2


@pytest.mark.parametrize("score", ["nan", "inf", "-1", "101", "1.5"])
def test_review_rejects_invalid_scores_without_mutating_manifest(
    local_rig: SimpleNamespace,
    score: str,
) -> None:
    from inspect_robots.colosseum_review import ReviewStore

    manifest, local = _collect_local(local_rig)
    before = manifest.read_bytes()
    with pytest.raises(ValueError, match="score"):
        ReviewStore(str(local_rig.config)).save(local["assignment"]["id"], _review_form(score))
    assert manifest.read_bytes() == before


def test_review_draft_is_local_and_published_scores_are_locked(local_rig: SimpleNamespace) -> None:
    from inspect_robots.colosseum_review import ReviewStore

    manifest, local = _collect_local(local_rig)
    identity = local["assignment"]["id"]
    store = ReviewStore(str(local_rig.config))
    form = _review_form()
    form["preference"] = [""]
    store.save(identity, form)
    saved = store.load(identity)
    assert not saved.get("result") and saved["draft"]["confidence"] == "clear"
    with pytest.raises(ValueError, match="save all"):
        store.start_job(identity, "submit")
    assert not local_rig.events
    saved["remote_evaluation_id"] = "manual_eval_1"
    manifest.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="locked"):
        store.save(identity, _review_form())


def test_review_submission_job_uses_submit_only_and_never_run(
    local_rig: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    from inspect_robots.colosseum_review import ReviewStore

    _, local = _collect_local(local_rig)
    commands: list[list[str]] = []

    def start(argv: list[str], **kwargs: Any) -> Any:
        commands.append(argv)
        assert kwargs["stdin"] == subprocess.DEVNULL
        return SimpleNamespace(poll=lambda: None, wait=lambda: 0)

    monkeypatch.setattr(subprocess, "Popen", start)
    store = ReviewStore(str(local_rig.config))
    store.start_job(local["assignment"]["id"], "submit")
    assert commands[0][:4] == [sys.executable, "-m", "inspect_robots.cli", "colosseum"]
    assert "--submit-only" in commands[0] and "--runtime-config" not in commands[0]
    with pytest.raises(ValueError, match="running"):
        store.start_job(local["assignment"]["id"], "submit")


def test_review_rejects_modified_export_before_upload(
    local_rig: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inspect_robots import colosseum_import
    from inspect_robots.colosseum_review import ReviewStore

    manifest, local = _collect_local(local_rig)
    store = ReviewStore(str(local_rig.config))
    store.save(local["assignment"]["id"], _review_form("25", "50"))
    folder = manifest.parent / local["assignment"]["runs"][0]["id"] / "lerobot"
    (folder / "data/chunk-000/file-000.parquet").write_bytes(b"tampered")
    monkeypatch.setattr(
        colosseum_import,
        "LocalUploadAPI",
        lambda config: pytest.fail("Modified evidence must never upload"),
    )
    assert (
        main(
            [
                "colosseum",
                "--client-config",
                str(local_rig.config),
                "--submit-only",
                "--resume",
                local["assignment"]["id"],
            ]
        )
        == 1
    )


def test_review_dispatch_is_offline_and_uses_project_port(
    local_rig: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inspect_robots import colosseum_review

    calls = []

    def serve(path: str, port: int, **kwargs: Any) -> int:
        calls.append((path, port))
        return 0

    monkeypatch.setattr(colosseum_review, "serve", serve)
    assert main(["colosseum", "--client-config", str(local_rig.config), "--review"]) == 0
    assert calls == [(str(local_rig.config), 8767)]
    assert not local_rig.executions and not local_rig.events and not local_rig.chosen


def test_review_http_views_save_and_seek_are_local_and_protected(
    local_rig: SimpleNamespace,
) -> None:
    import http.client
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.parse import urlencode

    from inspect_robots.colosseum_review import ReviewStore, handler

    manifest, local = _collect_local(local_rig)
    identity = local["assignment"]["id"]
    store = ReviewStore(str(local_rig.config))
    run = local["assignment"]["runs"][0]
    preview = store.preview(identity, run["id"], "head_image")
    assert preview is not None
    before = manifest.read_bytes()
    with ThreadingHTTPServer(("127.0.0.1", 0), handler(store)) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        try:
            client.request("GET", "/")
            response = client.getresponse()
            assert response.status == 200 and identity.encode() in response.read()
            client.request("GET", f"/ev/{identity}")
            response = client.getresponse()
            assert response.status == 200 and b"Save locally" in response.read()
            form = {key: value[0] for key, value in _review_form().items()}
            client.request("POST", f"/ev/{identity}", urlencode(form))
            response = client.getresponse()
            assert response.status == 403
            response.read()
            assert manifest.read_bytes() == before
            form.update(token=store.token, action="save", score_A="25")
            client.request("POST", f"/ev/{identity}", urlencode(form))
            response = client.getresponse()
            assert response.status == 200 and b"No uploads started" in response.read()
            assert store.load(identity)["outcomes"]["A"]["score_0_4"] == 1
            assert store.load(identity)["outcomes"]["A"]["partial_success"] == 0.25
            route = f"/video/{identity}/{run['id']}/head_image"
            client.request("GET", route, headers={"Range": "bytes=0-15"})
            response = client.getresponse()
            assert response.status == 206
            assert response.read() == preview.read_bytes()[:16]
            client.request("GET", route, headers={"Range": "bytes=999999999-"})
            response = client.getresponse()
            assert response.status == 416
            response.read()
            client.request("GET", "/", headers={"Host": "untrusted.example"})
            response = client.getresponse()
            assert response.status == 400
            response.read()
            assert not local_rig.events and not local_rig.uploads
        finally:
            client.close()
            server.shutdown()
            thread.join(timeout=5)


def test_review_cannot_submit_incomplete_runs_or_escaped_evidence(
    local_rig: SimpleNamespace,
) -> None:
    from inspect_robots.colosseum_review import ReviewStore

    manifest, local = _collect_local(local_rig)
    identity = local["assignment"]["id"]
    store = ReviewStore(str(local_rig.config))
    local["assignment"]["runs"][0]["state"] = "assigned"
    manifest.write_text(json.dumps(local))
    with pytest.raises(ValueError, match="Finish"):
        store.start_job(identity, "submit")
    with pytest.raises(ValueError, match="ID"):
        store.load("../outside")
    with pytest.raises(ValueError, match="escapes"):
        store.path(identity, "..", "..", "outside")
    assert not local_rig.events


def test_client_frequency_drives_native_execution_and_dataset_timestamps(
    local_rig: SimpleNamespace,
) -> None:
    import pyarrow.parquet as pq
    import yaml

    config = yaml.safe_load(local_rig.config.read_text())
    config["control_hz"] = 20
    local_rig.config.write_text(yaml.safe_dump(config))
    runtime = yaml.safe_load(local_rig.catalog.read_text())
    for model in runtime["models"]:
        model["control_hz"] = 20
    local_rig.catalog.write_text(yaml.safe_dump(runtime))
    manifest, local = _collect_local(local_rig)
    for run in local["assignment"]["runs"]:
        folder = manifest.parent / run["id"]
        context = json.loads((folder / "recording-context.json").read_text())
        assert context["fps"] == 20
        assert context["native_execution"]["eval"]["embodiment_info"]["control_hz"] == 20
        assert context["native_execution"]["embodiment_args"]["control_hz"] == 20
        dataset = folder / "lerobot"
        info = json.loads((dataset / "meta/info.json").read_text())
        assert info["fps"] == 20
        data = pq.read_table(dataset / "data/chunk-000/file-000.parquet").to_pydict()
        assert data["timestamp"] == pytest.approx([0, 1 / 20])


def test_catalog_frequency_mismatch_fails_before_policy_or_robot(
    local_rig: SimpleNamespace,
) -> None:
    import yaml

    config = yaml.safe_load(local_rig.config.read_text())
    config["control_hz"] = 20
    local_rig.config.write_text(yaml.safe_dump(config))
    assert main(local_rig.local_argv) == 1
    assert not local_rig.chosen and not local_rig.executions and not local_rig.events


def test_native_wire_uses_ini_and_explicit_policy_override(tmp_path: Path) -> None:
    from inspect_robots.cli import _parse_kvs

    ini = tmp_path / "rig.ini"
    ini.write_text("[defaults]\npolicy=agent\n[policy.args]\nwire=responses\n")
    argv = ["colosseum", "--client-config", "client.yaml", "--", "--config", str(ini)]
    args = bridge.native_args(
        build_parser().parse_args(argv),
        {"instruction": "pick", "max_steps": 800},
        tmp_path,
        control_hz=30,
    )
    assert _parse_kvs(args.policy_args)["wire"] == "responses"
    assert args.max_steps == 800
    args = bridge.native_args(
        build_parser().parse_args([*argv, "-P", "wire=chat"]),
        {"instruction": "pick", "max_steps": 800},
        tmp_path,
        control_hz=30,
    )
    assert _parse_kvs(args.policy_args)["wire"] == "chat"


def test_explicit_wire_overrides_model_runtime_wire(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("inspect_robots_agent")
    import httpx

    from inspect_robots import registry

    original = registry.resolve

    def resolve(kind: str, name: str, **kwargs: Any) -> Any:
        kwargs["transport"] = httpx.MockTransport(lambda request: pytest.fail("No inference"))
        return original(kind, name, **kwargs)

    monkeypatch.setattr(registry, "resolve", resolve)
    config = SimpleNamespace(llm_api_keys={"xai": "session-secret"}, llm_account_logins={})
    model = {**MODEL, "name": "grok-4.7", "bridge_options": {"wire": "responses"}}
    agent = bridge.make_agent(model, config, {"wire": "chat"})
    try:
        assert agent.config.wire == "chat" and agent.config.model == "grok-4.7"
    finally:
        agent._client.close()
