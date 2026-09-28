"""Grouped run artifacts and remote checkpoint identity."""

from __future__ import annotations

import importlib
import json
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from inspect_robots import eval, read_eval_log
from inspect_robots.cli import _directory_log_paths, main
from inspect_robots.eval import _allocate_run_dir, _policy_server_metadata
from inspect_robots.logging import JsonLogSink, LiveLogSink
from inspect_robots.logging.rerun_sink import RerunSink
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.registry import resolve


def test_eval_groups_all_builtin_artifacts_by_sequential_run(tmp_path: Path) -> None:
    task = resolve("task", "cubepick-reach")

    first = eval(
        task,
        ScriptedPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        store_frames=True,
    )[0]
    second = eval(
        task,
        ScriptedPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        store_frames=True,
    )[0]

    assert first.eval.run_id is not None
    assert second.eval.run_id is not None
    assert re.fullmatch(r"\d{8}-run001", first.eval.run_id)
    assert second.eval.run_id == first.eval.run_id[:-3] + "002"
    for log in (first, second):
        run_dir = tmp_path / str(log.eval.run_id)
        log_path = run_dir / f"{log.eval.run_id}.json"
        assert log_path.is_file()
        saved = read_eval_log(str(log_path))
        assert saved.eval.run_id == log.eval.run_id
        assert saved.stats.frames_dir == str(run_dir / "frames")
        for sample in saved.samples:
            for metadata in sample.trial_metadata:
                assert (run_dir / metadata["actions"]).is_file()
        assert list((run_dir / "actions").glob("*.jsonl"))
        assert list((run_dir / "frames").glob("*.npy"))
    assert not list(tmp_path.glob("*.json"))
    assert not (tmp_path / "actions").exists()
    assert not (tmp_path / "frames").exists()


def test_live_snapshot_and_final_log_share_the_run_name(tmp_path: Path) -> None:
    final = JsonLogSink(str(tmp_path))
    live = LiveLogSink(str(tmp_path), min_write_interval_s=0)
    log = eval(
        resolve("task", "cubepick-reach"),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        sinks=[final, live],
    )[0]
    run_dir = tmp_path / str(log.eval.run_id)

    assert final.path == run_dir / f"{log.eval.run_id}.json"
    assert final.path.is_file()
    assert live.path == run_dir / f"{log.eval.run_id}.live.json"
    assert not live.path.exists()


def test_explicit_separate_file_sink_roots_keep_their_layout(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    custom = tmp_path / "custom"
    final = JsonLogSink(str(custom))
    live = LiveLogSink(str(custom))
    log = eval(
        resolve("task", "cubepick-reach"),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        log_dir=str(root),
        sinks=[final, live],
    )[0]

    assert final.path is not None and final.path.parent == custom
    assert live.path is not None and live.path.parent == custom
    assert final.path.is_file()
    assert re.fullmatch(r"cubepick-reach_[0-9a-f]{8}\.json", final.path.name)
    assert not live.path.exists()
    assert not list((root / str(log.eval.run_id)).glob("*.json"))


def test_directory_discovery_preserves_new_legacy_and_flat_logs(tmp_path: Path) -> None:
    expected = [tmp_path / "legacy.json"]
    for name in ("20260928-run001", "20260928-RUN002", "20260928_run0003"):
        run_dir = tmp_path / name
        run_dir.mkdir()
        expected.append(run_dir / "run.json")
    for path in expected:
        path.touch()
    unrelated = tmp_path / "other"
    unrelated.mkdir()
    (unrelated / "not-a-run.json").touch()
    (tmp_path / "20260928-run004").touch()

    assert _directory_log_paths(tmp_path) == sorted(expected)


def test_run_directory_allocator_retries_a_concurrent_claim(
    tmp_path: Path, monkeypatch: Any
) -> None:
    real_mkdir = Path.mkdir
    raced = False

    def mkdir(path: Path, *args: Any, **kwargs: Any) -> None:
        nonlocal raced
        if path.parent == tmp_path and path.name.endswith("-run001") and not raced:
            raced = True
            real_mkdir(path, *args, **kwargs)
            raise FileExistsError(path)
        real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mkdir)
    run_id, run_dir = _allocate_run_dir(str(tmp_path))

    assert raced
    assert run_id.endswith("-run002")
    assert run_dir.is_dir()


def test_run_allocator_preserves_legacy_sequences_and_skips_file_collisions(
    tmp_path: Path,
) -> None:
    date = datetime.now().astimezone().strftime("%Y%m%d")
    (tmp_path / f"{date}_run0007").mkdir()
    (tmp_path / f"{date}-RUN008").mkdir()
    (tmp_path / f"{date}-run009").touch()
    (tmp_path / "unrelated").mkdir()

    run_id, run_dir = _allocate_run_dir(str(tmp_path))

    assert run_id == f"{date}-run010"
    assert run_dir.is_dir()


def test_run_numbers_are_per_agent_and_reset_each_day(tmp_path: Path, monkeypatch: Any) -> None:
    current = [datetime(2026, 9, 28, 12)]

    class Clock:
        @staticmethod
        def now() -> datetime:
            return current[0]

    eval_module = importlib.import_module("inspect_robots.eval")
    monkeypatch.setattr(eval_module, "datetime", Clock)
    claude = str(tmp_path / "yam" / "claude")
    gpt = str(tmp_path / "yam" / "gpt")

    assert _allocate_run_dir(claude)[0] == "20260928-run001"
    assert _allocate_run_dir(claude)[0] == "20260928-run002"
    assert _allocate_run_dir(gpt)[0] == "20260928-run001"
    current[0] = datetime(2026, 9, 29, 12)
    assert _allocate_run_dir(claude)[0] == "20260929-run001"


def test_policy_server_metadata_is_best_effort(monkeypatch: Any) -> None:
    class RaisingURL:
        @property
        def server_url(self) -> str:
            raise RuntimeError("broken property")

    class NonHTTP:
        server_url = "ws://model-host"

    class RaisingMetadataURL:
        server_url = "http://model-host"

        @property
        def server_metadata_url(self) -> str:
            raise RuntimeError("broken property")

    class EmptyMetadataURL:
        server_url = "http://model-host"
        server_metadata_url = ""

    eval_module = importlib.import_module("inspect_robots.eval")

    def offline(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("offline")

    monkeypatch.setattr(eval_module.urllib.request, "urlopen", offline)

    assert _policy_server_metadata(cast(Any, RaisingURL())) == {}
    assert _policy_server_metadata(cast(Any, NonHTTP())) == {}
    raised = _policy_server_metadata(cast(Any, RaisingMetadataURL()))
    empty = _policy_server_metadata(cast(Any, EmptyMetadataURL()))
    assert raised["metadata_url"] == "http://model-host/act"
    assert empty["metadata_url"] == "http://model-host/act"
    assert "OSError: offline" in raised["probe_error"]


def test_policy_server_metadata_ignores_non_object_health(monkeypatch: Any) -> None:
    class Policy:
        server_url = "http://model-host"
        server_metadata_url = "http://model-host/metadata"

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"[]"

    eval_module = importlib.import_module("inspect_robots.eval")
    monkeypatch.setattr(eval_module.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())

    assert _policy_server_metadata(cast(Any, Policy())) == {
        "url": "http://model-host",
        "metadata_url": "http://model-host/metadata",
    }


def test_rerun_directory_target_rebinds_only_when_configured(tmp_path: Path) -> None:
    fixed = RerunSink(str(tmp_path / "fixed.rrd"))
    grouped = RerunSink(recording_dir=str(tmp_path))
    separate = RerunSink(recording_dir=str(tmp_path / "separate"))

    fixed.bind_run_dir(str(tmp_path / "run"), "run")
    grouped.bind_run_dir(str(tmp_path / "run"), "run")
    separate.bind_run_dir(str(tmp_path / "run"), "run")

    assert fixed.recording_dir is None
    assert grouped.recording_dir == str(tmp_path / "run")
    assert separate.recording_dir == str(tmp_path / "separate")


def test_eval_records_action_server_checkpoint_and_revision(
    tmp_path: Path, monkeypatch: Any
) -> None:
    class ServerPolicy(ScriptedPolicy):
        server_url = "http://model-host:8202"

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "status": "ok",
                    "checkpoint": "allenai/MolmoAct2-BimanualYAM",
                    "revision": "8dcbed66c7c0",
                }
            ).encode()

    eval_module = importlib.import_module("inspect_robots.eval")
    seen: list[tuple[str, float]] = []

    def urlopen(url: str, timeout: float) -> Response:
        seen.append((url, timeout))
        return Response()

    monkeypatch.setattr(eval_module.urllib.request, "urlopen", urlopen)
    log = eval(
        resolve("task", "cubepick-reach"),
        ServerPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
    )[0]

    assert seen == [("http://model-host:8202/act", 3.0)]
    assert log.eval.policy_server["checkpoint"] == "allenai/MolmoAct2-BimanualYAM"
    assert log.eval.policy_server["revision"] == "8dcbed66c7c0"
    (path,) = (tmp_path / str(log.eval.run_id)).glob("*.json")
    assert read_eval_log(str(path)).eval.policy_server == log.eval.policy_server


def test_view_places_nested_report_in_the_run_directory(tmp_path: Path) -> None:
    log = eval(
        resolve("task", "cubepick-reach"),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
    )[0]
    run_dir = tmp_path / str(log.eval.run_id)
    (log_path,) = run_dir.glob("*.json")

    assert main(["view", str(log_path), "--no-video"]) == 0
    report = run_dir / "html" / f"{log_path.stem}.html"
    assert report.is_file()
    assert str(log.eval.run_id) in report.read_text(encoding="utf-8")

    assert main(["view", str(tmp_path), "--no-video"]) == 0
    index = tmp_path / "html" / "index.html"
    assert index.is_file()
    assert f"../{run_dir.name}/html/{report.name}" in index.read_text(encoding="utf-8")


def test_cli_prints_server_checkpoint_identity(tmp_path: Path, capsys: Any) -> None:
    log = eval(
        resolve("task", "cubepick-reach"),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
    )[0]
    run_dir = tmp_path / str(log.eval.run_id)
    (log_path,) = run_dir.glob("*.json")
    server = replace(
        log,
        eval=replace(
            log.eval,
            policy_server={"checkpoint": "org/model", "revision": "abc123"},
        ),
    )
    cli = importlib.import_module("inspect_robots.cli")

    cli._print_run_summary(server, str(log_path), False)
    assert "server checkpoint: org/model@abc123" in capsys.readouterr().out

    log_path.write_text(json.dumps(server.to_dict()), encoding="utf-8")
    assert main(["inspect", str(log_path)]) == 0
    assert "checkpoint:  org/model@abc123" in capsys.readouterr().out

    without_revision = replace(
        server,
        eval=replace(server.eval, policy_server={"repo_id": "org/legacy"}),
    )
    cli._print_run_summary(without_revision, str(log_path), False)
    assert "server checkpoint: org/legacy" in capsys.readouterr().out

    log_path.write_text(json.dumps(without_revision.to_dict()), encoding="utf-8")
    assert main(["inspect", str(log_path)]) == 0
    inspect_output = capsys.readouterr().out
    assert "checkpoint:  org/legacy" in inspect_output
    assert "org/legacy@" not in inspect_output
