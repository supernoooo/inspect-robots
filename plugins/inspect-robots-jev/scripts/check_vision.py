"""Inspect saved RGB frames with the service or a deterministic fake response."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from inspect_robots_jev.vision_client import VisionClient, VisionOutcome
from inspect_robots_jev.vision_protocol import (
    PROFILE_VERSION, PROMPT_VERSIONS, SERVICE_VERSION, SIM_PROFILE, TRAY_PROFILE,
    VERSION, decode_response, encode_mask,
)

COLORS = {"red_block": (255, 50, 50), "yellow_ball": (250, 205, 35),
          "box": (50, 165, 255), "tray": (35, 205, 255)}
CAMERAS = {"top_cam", "left_cam", "right_cam"}


def fake_outcome(height: int, width: int, camera: str, captured_at: float,
                 task_profile: str | None = None) -> VisionOutcome:
    if height < 6 or width < 6:
        raise ValueError("fake overlay needs images at least 6x6")
    request_id = "fixed-fake-response"
    model = {"detector_id": "fake-grounding-dino", "detector_revision": "offline-v1",
             "segmenter_id": "fake-sam2", "segmenter_revision": "offline-v1"}
    if task_profile is not None:
        model = {**model, "prompt_version": PROMPT_VERSIONS[task_profile],
                 "service_version": SERVICE_VERSION}
    rows = []
    categories = ("red_block", "yellow_ball", "box") if task_profile != TRAY_PROFILE else ("red_block", "tray")
    for index, category in enumerate(categories):
        x1 = index * width // len(categories)
        x2 = (index + 1) * width // len(categories)
        y1, y2 = height // 4, 3 * height // 4
        mask = np.zeros((height, width), dtype=np.bool_)
        mask[y1:y2, x1:x2] = True
        rows.append({"category": category, "box": [x1, y1, x2, y2], "mask": encode_mask(mask),
                     "detection_score": 0.9, "segmentation_score": 0.95, "occluded": False,
                     "camera": camera, "request_id": request_id, "captured_at": captured_at,
                     "model_version": model})
    response = {"version": VERSION if task_profile is None else PROFILE_VERSION,
                "request_id": request_id, "camera": camera,
                "captured_at": captured_at, "height": height, "width": width,
                "model_version": model, "detections": rows}
    if task_profile is not None:
        response["task_profile"] = task_profile
    detections = decode_response(response, request_id=request_id, camera=camera,
                                 captured_at=captured_at, height=height, width=width,
                                 task_profile=task_profile)
    return VisionOutcome("ready", None, detections, model, request_id)


def render(image: Image.Image, outcome: VisionOutcome) -> Image.Image:
    result = image.convert("RGBA")
    for detection in outcome.detections or outcome.review_detections:
        color = COLORS[detection.category]
        layer = Image.new("RGBA", result.size, (*color, 0))
        alpha = Image.fromarray(np.where(detection.mask, 90, 0).astype(np.uint8), "L")
        layer.putalpha(alpha)
        result = Image.alpha_composite(result, layer)
        draw = ImageDraw.Draw(result)
        draw.rectangle(detection.box, outline=(*color, 255), width=2)
        label = (f"{detection.category} D:{detection.detection_score:.2f} "
                 f"S:{detection.segmentation_score:.2f} O:{int(detection.occluded)}")
        draw.text((detection.box[0], max(0, detection.box[1] - 12)), label, fill=(*color, 255))
    if outcome.failure:
        ImageDraw.Draw(result).text((2, 2), f"FAIL: {outcome.failure}", fill=(255, 255, 255, 255),
                                    stroke_width=1, stroke_fill=(160, 0, 0, 255))
    return result.convert("RGB")


def outcome_record(path: Path, overlay: Path, outcome: VisionOutcome, *, camera: str,
                   captured_at: float, height: int, width: int, profile: str,
                   mode: str) -> dict[str, Any]:
    return {"source": str(path), "overlay": str(overlay), "mode": mode,
            "task_profile": profile, "camera": camera, "captured_at": captured_at,
            "height": height, "width": width, "request_id": outcome.request_id,
            "model_version": outcome.model_version, "state": outcome.state,
            "failure": outcome.failure,
            "human_review": {"status": "pending", "notes": ""},
            "detections": [{"category": d.category, "box": list(d.box),
                            "mask": encode_mask(d.mask), "detection_score": d.detection_score,
                            "segmentation_score": d.segmentation_score, "occluded": d.occluded,
                            "failure_reason": outcome.failure,
                            "camera": d.camera, "request_id": d.request_id,
                            "captured_at": d.captured_at, "model_version": d.model_version}
                           for d in outcome.detections or outcome.review_detections]}


def load_manifest(path: Path) -> list[tuple[Path, str, float, int, int]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"frames"} or not isinstance(value["frames"], list) or not value["frames"]:
        raise ValueError("manifest must contain a nonempty frames list")
    frames = []
    for row in value["frames"]:
        if not isinstance(row, dict) or set(row) != {"path", "camera", "captured_at", "height", "width"}:
            raise ValueError("each frame needs path, camera, captured_at, height, width")
        if not isinstance(row["path"], str) or not row["path"] or row["camera"] not in CAMERAS:
            raise ValueError("invalid frame path or camera")
        stamp = row["captured_at"]
        if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
            raise ValueError("captured_at must be an original finite numeric timestamp")
        if any(type(row[key]) is not int or row[key] < 1 for key in ("height", "width")):
            raise ValueError("height and width must be positive integers")
        frame_path = Path(row["path"])
        if not frame_path.is_absolute():
            frame_path = path.parent / frame_path
        frames.append((frame_path, row["camera"], stamp, row["height"], row["width"]))
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay Jev vision results on saved frames")
    parser.add_argument("frames", nargs="*", type=Path, help="protocol fake frames only")
    parser.add_argument("--manifest", type=Path, help="saved frame metadata JSON; required for real service")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera", choices=sorted(CAMERAS), default="top_cam")
    parser.add_argument("--profile", choices=[SIM_PROFILE, TRAY_PROFILE], default=SIM_PROFILE)
    parser.add_argument("--url", default="http://127.0.0.1:8765/v1/detect")
    parser.add_argument("--fake", action="store_true", help="protocol/overlay test only; no recognition evidence")
    parser.add_argument("--detector-revision", help="expected resolved 40-hex Grounding DINO commit")
    parser.add_argument("--segmenter-revision", help="expected resolved 40-hex SAM commit")
    args = parser.parse_args()
    if args.frames and (args.manifest or not args.fake):
        parser.error("positional frames are for --fake only; use --manifest for saved real frames")
    if not args.frames and args.manifest is None:
        parser.error("supply --manifest or positional --fake frames")
    if not args.fake and args.profile == TRAY_PROFILE and not all(
            value and re.fullmatch(r"[0-9a-f]{40}", value)
            for value in (args.detector_revision, args.segmenter_revision)):
        parser.error("tray service mode requires both resolved 40-hex model revisions")
    task_profile = None if args.profile == SIM_PROFILE else TRAY_PROFILE
    expected = None
    if not args.fake and task_profile is not None:
        expected = {"detector_id": "IDEA-Research/grounding-dino-tiny",
                    "detector_revision": args.detector_revision,
                    "segmenter_id": "facebook/sam2.1-hiera-tiny",
                    "segmenter_revision": args.segmenter_revision,
                    "prompt_version": PROMPT_VERSIONS[task_profile],
                    "service_version": SERVICE_VERSION}
    client = None if args.fake else VisionClient(args.url, task_profile=task_profile,
                                                  expected_model_version=expected)
    if args.manifest:
        frames = load_manifest(args.manifest)
    else:
        frames = [(path, args.camera, time.monotonic(), -1, -1) for path in args.frames]
    if not args.fake and args.profile == TRAY_PROFILE and {row[1] for row in frames} != CAMERAS:
        parser.error("tray real-service manifest must contain top, left, and right cameras")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (path, camera, captured_at, height, width) in enumerate(frames):
        with Image.open(path) as source:
            if source.mode != "RGB":
                raise ValueError(f"{path}: saved frame must be RGB")
            image = source.copy()
        if height != -1 and (image.height, image.width) != (height, width):
            raise ValueError(f"{path}: image dimensions differ from manifest")
        pixels = np.asarray(image)
        outcome = (fake_outcome(image.height, image.width, camera, captured_at, task_profile)
                   if client is None else client.observe(pixels, camera, captured_at, enforce_age=False))
        overlay = args.output_dir / f"{index:03d}_{camera}_{path.stem}_overlay.png"
        render(image, outcome).save(overlay)
        records.append(outcome_record(path, overlay, outcome, camera=camera,
                                      captured_at=captured_at, height=image.height,
                                      width=image.width, profile=args.profile,
                                      mode="protocol_fake" if args.fake else "real_service"))
    result_path = args.output_dir / "results.json"
    result_path.write_text(json.dumps({"version": VERSION if task_profile is None else PROFILE_VERSION,
                                       "mode": "protocol_fake" if args.fake else "real_service",
                                       "task_profile": args.profile, "frames": records}, indent=2), encoding="utf-8")
    print(result_path)


if __name__ == "__main__":
    main()
