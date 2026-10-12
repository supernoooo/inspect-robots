"""Excel exports preserve old statistics, upsert runs and survive interrupted saves."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from inspect_robots.colosseum_excel import (
    HEADERS,
    SHEET,
    WorkbookBusy,
    evaluation_rows,
    update_workbook,
    validate_target,
)

openpyxl = pytest.importorskip("openpyxl")


def manifest(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Any] = {
        "assignment": {
            "id": "ev_test",
            "track": "open",
            "scene": "red block on table",
            "institution": "NUS",
            "evaluator": "operator_01",
            "test": False,
            "recording": True,
            "task": {"instruction": "Pick up the red block"},
            "runs": [{"id": "run_a", "side": "A"}, {"id": "run_b", "side": "B"}],
        },
        "outcomes": {
            "A": {"success": False, "partial_success": 0.75},
            "B": {"success": True, "partial_success": 1},
        },
        "result": {"preference": "B", "feedback": "B placed the block correctly"},
        "submitted": True,
        "datasets_uploaded": {"run_a": True},
        "excel_router": "https://router.example",
        "token": "never-export-router-secret",
    }
    for run in saved["assignment"]["runs"]:
        path = root / run["id"]
        path.mkdir()
        (path / "policy.json").write_text(
            json.dumps(
                {
                    "model": {
                        "name": "gpt-6-astra",
                        "url": "https://api.example/v1",
                        "revision": "rig-v1",
                    },
                    "api_key": "never-export-provider-secret",
                }
            )
        )
        (path / "recording-context.json").write_text(
            json.dumps({"native_control_mode": "eef_abs_pose"})
        )
        (path / "inspect-result.json").write_text(
            json.dumps(
                {
                    "stats": {
                        "total_steps": 25,
                        "duration_s": 12.5,
                        "started_at": "2026-10-11T10:00:00Z",
                    },
                }
            )
        )
        video = path / "lerobot/videos/observation.images.head_image/chunk-000/file-000.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"test video placeholder")
    result = root / "manifest.json"
    result.write_text(json.dumps(saved))
    return result


def test_preserves_existing_sheets_and_adds_scores_with_video_links(tmp_path: Path) -> None:
    target = tmp_path / "results.xlsx"
    book = openpyxl.Workbook()
    book.active.title = "Leaderboard"
    book.active["A1"] = "Existing VLA results"
    book.active["B2"] = "=SUM(1,2)"
    book.active["A1"].font = openpyxl.styles.Font(bold=True, color="123456")
    book.create_sheet("Task x VLA")["C3"] = 0.8
    book.create_sheet("Subtasks")
    book.create_sheet("Episodes")["A4"] = "Legacy episode"
    book.save(target)
    book.close()
    original = target.read_bytes()
    source = manifest(tmp_path / "evals/ev_test")
    assert update_workbook(target, [source]) == 2
    assert target.with_name("results.before-colosseum.xlsx").read_bytes() == original
    loaded = openpyxl.load_workbook(target)
    try:
        assert loaded.sheetnames == ["Leaderboard", "Task x VLA", "Subtasks", "Episodes", SHEET]
        assert loaded["Leaderboard"]["A1"].value == "Existing VLA results"
        assert loaded["Leaderboard"]["A1"].font.color.rgb == "00123456"
        assert loaded["Leaderboard"]["B2"].value == "=SUM(1,2)"
        assert loaded["Leaderboard"]["B2"].data_type == "f"
        assert loaded["Task x VLA"]["C3"].value == 0.8
        sheet = loaded[SHEET]
        assert sheet.max_row == 3
        values = dict(zip(HEADERS, (cell.value for cell in sheet[2]), strict=True))
        assert values["Score (0-1)"] == 0.75 and values["Score (%)"] == 75
        assert values["Evaluator"] == "operator_01" and values["Preference"] == "B"
        assert values["Difficulty"] is None
        assert values["Native control"] == "EEF / IK"
        assert values["Dataset uploaded"] is True
        assert sheet.cell(3, HEADERS.index("Dataset uploaded") + 1).value is False
        cell = sheet.cell(2, HEADERS.index("Head video") + 1)
        assert (
            cell.hyperlink.target
            == "evals/ev_test/run_a/lerobot/videos/observation.images.head_image/"
            "chunk-000/file-000.mp4"
        )
        assert "never-export" not in str(list(sheet.values))
    finally:
        loaded.close()


def test_resume_updates_existing_rows_and_keeps_feedback_as_literal_text(tmp_path: Path) -> None:
    source = manifest(tmp_path / "evals/ev_test")
    target = tmp_path / "results.xlsx"
    update_workbook(target, [source])
    saved = json.loads(source.read_text())
    saved["outcomes"]["A"]["partial_success"] = 0.5
    saved["datasets_uploaded"]["run_b"] = True
    saved["result"]["feedback"] = '=HYPERLINK("https://example.test","text")'
    source.write_text(json.dumps(saved))
    update_workbook(target, [source])
    book = openpyxl.load_workbook(target)
    try:
        sheet = book[SHEET]
        assert sheet.max_row == 3
        assert sheet.cell(2, HEADERS.index("Score (0-1)") + 1).value == 0.5
        assert sheet.cell(3, HEADERS.index("Dataset uploaded") + 1).value is True
        cell = sheet.cell(2, HEADERS.index("Feedback") + 1)
        assert cell.data_type == "s" and cell.value.startswith("=HYPERLINK")
    finally:
        book.close()


def test_editor_lock_defers_export_without_changing_workbook(tmp_path: Path) -> None:
    source = manifest(tmp_path / "evals/ev_test")
    target = tmp_path / "results.xlsx"
    book = openpyxl.Workbook()
    book.save(target)
    book.close()
    original = target.read_bytes()
    (tmp_path / ".~lock.results.xlsx#").write_text("editor lock")
    with pytest.raises(WorkbookBusy, match="Close"):
        update_workbook(target, [source])
    assert target.read_bytes() == original
    assert not (tmp_path / "results.before-colosseum.xlsx").exists()


def test_failed_save_is_atomic_and_does_not_lose_existing_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = manifest(tmp_path / "evals/ev_test")
    target = tmp_path / "results.xlsx"
    book = openpyxl.Workbook()
    book.active["A1"] = "keep this"
    book.save(target)
    book.close()
    original = target.read_bytes()

    def fail(self: Any, filename: Path) -> None:
        filename.write_bytes(b"incomplete archive")
        raise OSError("disk error")

    monkeypatch.setattr(openpyxl.Workbook, "save", fail)
    with pytest.raises(OSError, match="disk error"):
        update_workbook(target, [source])
    assert target.read_bytes() == original
    assert not list(tmp_path.glob(".colosseum-excel-*.xlsx"))


def test_unscored_runs_are_not_fabricated_as_failures(tmp_path: Path) -> None:
    source = manifest(tmp_path / "evals/ev_test")
    saved = json.loads(source.read_text())
    saved["outcomes"].pop("B")
    source.write_text(json.dumps(saved))
    assert [row["Side"] for row in evaluation_rows(source)] == ["A"]


def test_existing_unrelated_sheet_with_same_name_is_not_overwritten(tmp_path: Path) -> None:
    source = manifest(tmp_path / "evals/ev_test")
    target = tmp_path / "results.xlsx"
    book = openpyxl.Workbook()
    book.active.title = SHEET
    book.active["A1"] = "My own calculations"
    book.save(target)
    book.close()
    original = target.read_bytes()
    with pytest.raises(ValueError, match="headers"):
        update_workbook(target, [source])
    assert target.read_bytes() == original


def test_output_format_is_checked_before_export(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="xlsx"):
        validate_target(tmp_path / "results.xls")
