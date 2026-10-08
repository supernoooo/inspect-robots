from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


def test_fixed_frames_generate_overlay_and_machine_results(tmp_path: Path) -> None:
    frames = []
    for index in range(2):
        frame = tmp_path / f"frame_{index}.png"
        Image.new("RGB", (48, 32), (20 + index, 30, 40)).save(frame)
        frames.append(frame)
    output = tmp_path / "output"
    script = Path(__file__).resolve().parents[2] / "scripts" / "check_vision.py"
    subprocess.run([sys.executable, str(script), *(str(frame) for frame in frames),
                    "--output-dir", str(output), "--fake"], check=True, timeout=10)
    records = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert records["version"] == 1
    assert len(records["frames"]) == 2
    for record in records["frames"]:
        assert record["state"] == "ready"
        assert {item["category"] for item in record["detections"]} == {
            "red_block", "yellow_ball", "box"}
        assert all(item["mask"]["encoding"] == "rle-row-major-v1"
                   for item in record["detections"])
        overlay = Path(record["overlay"])
        assert overlay.exists()
        with Image.open(overlay) as image:
            assert image.size == (48, 32)


def test_tray_manifest_preserves_three_saved_frame_stamps(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[2] / "scripts" / "check_vision.py"
    entries = []
    for index, camera in enumerate(("top_cam", "left_cam", "right_cam")):
        name = f"{camera}.png"
        Image.new("RGB", (48, 32), (20, 30, 40)).save(tmp_path / name)
        entries.append({"path": name, "camera": camera, "captured_at": 100.125 + index,
                        "height": 32, "width": 48})
    manifest = tmp_path / "frames.json"
    manifest.write_text(json.dumps({"frames": entries}), encoding="utf-8")
    output = tmp_path / "tray-output"
    subprocess.run([sys.executable, str(script), "--manifest", str(manifest),
                    "--profile", "right_red_block_tray", "--output-dir", str(output),
                    "--fake"], check=True, timeout=10)
    result = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert result["version"] == 2 and result["mode"] == "protocol_fake"
    assert len(result["frames"]) == 3
    for source, record in zip(entries, result["frames"]):
        assert record["camera"] == source["camera"]
        assert record["captured_at"] == source["captured_at"]
        assert record["human_review"]["status"] == "pending"
        assert {d["category"] for d in record["detections"]} == {"red_block", "tray"}
        assert all(d["captured_at"] == source["captured_at"] and d["mask"]["counts"]
                   for d in record["detections"])
        assert Path(record["overlay"]).exists()

    entries[0]["width"] = 47
    manifest.write_text(json.dumps({"frames": entries}), encoding="utf-8")
    bad = subprocess.run([sys.executable, str(script), "--manifest", str(manifest),
                          "--profile", "right_red_block_tray", "--output-dir", str(output),
                          "--fake"], capture_output=True, text=True, timeout=10)
    assert bad.returncode != 0 and "dimensions differ" in bad.stderr
