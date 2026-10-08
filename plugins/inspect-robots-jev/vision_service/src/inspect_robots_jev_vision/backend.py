"""Grounding DINO tiny boxes refined with SAM 2.1 tiny masks."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from inspect_robots_jev.vision_protocol import SIM_PROFILE, TRAY_PROFILE, VisionProtocolError

DETECTOR_ID = "IDEA-Research/grounding-dino-tiny"
SEGMENTER_ID = "facebook/sam2.1-hiera-tiny"
PROMPTS = {SIM_PROFILE: "red block. yellow ball. box.",
           TRAY_PROFILE: "red block. tray."}
LABELS = {SIM_PROFILE: {"red block": "red_block", "yellow ball": "yellow_ball", "box": "box"},
          TRAY_PROFILE: {"red block": "red_block", "tray": "tray"}}


@dataclass(frozen=True, eq=False)
class Region:
    category: str
    box: tuple[int, int, int, int]
    mask: npt.NDArray[np.bool_]
    detection_score: float
    segmentation_score: float
    occluded: bool


class Backend(Protocol):
    model_version: dict[str, str]

    def infer(self, image: npt.NDArray[np.uint8], task_profile: str = SIM_PROFILE) -> list[Region]: ...


class PretrainedBackend:
    """Load pinned model snapshots in the service process only."""

    def __init__(self, *, detector_revision: str = "main", segmenter_revision: str = "main",
                 local_files_only: bool = False, device: str | None = None) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                                  Sam2Model, Sam2Processor)

        detector_path = snapshot_download(DETECTOR_ID, revision=detector_revision,
                                          local_files_only=local_files_only,
                                          allow_patterns=["*.json", "*.safetensors", "*.txt"])
        segmenter_path = snapshot_download(SEGMENTER_ID, revision=segmenter_revision,
                                           local_files_only=local_files_only,
                                           allow_patterns=["*.json", "*.safetensors", "*.txt"])
        detector_commit = Path(detector_path).name
        segmenter_commit = Path(segmenter_path).name
        if not all(re.fullmatch(r"[0-9a-f]{40}", commit)
                   for commit in (detector_commit, segmenter_commit)):
            raise ValueError("model snapshots must resolve to 40-character commits")
        self.model_version = {
            "detector_id": DETECTOR_ID, "detector_revision": detector_commit,
            "segmenter_id": SEGMENTER_ID, "segmenter_revision": segmenter_commit,
        }
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detector_processor = AutoProcessor.from_pretrained(detector_path)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(detector_path).to(self.device).eval()
        self.segmenter_processor = Sam2Processor.from_pretrained(segmenter_path)
        self.segmenter = Sam2Model.from_pretrained(segmenter_path).to(self.device).eval()

    def infer(self, image: npt.NDArray[np.uint8], task_profile: str = SIM_PROFILE) -> list[Region]:
        import torch
        from PIL import Image

        picture = Image.fromarray(image, mode="RGB")
        height, width = image.shape[:2]
        inputs = self.detector_processor(images=picture, text=PROMPTS[task_profile], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self.detector(**inputs)
        found = self.detector_processor.post_process_grounded_object_detection(
            output, inputs.input_ids, box_threshold=0.30, text_threshold=0.25,
            target_sizes=[(height, width)])[0]
        regions: list[Region] = []
        for label, score, raw_box in zip(found["labels"], found["scores"], found["boxes"]):
            phrase = str(label).lower().strip(" .")
            category = LABELS[task_profile].get(phrase)
            if category is None:
                continue
            x1, y1, x2, y2 = [float(v) for v in raw_box]
            box = (max(0, int(math.floor(x1))), max(0, int(math.floor(y1))),
                   min(width, int(math.ceil(x2))), min(height, int(math.ceil(y2))))
            if box[0] >= box[2] or box[1] >= box[3]:
                continue
            sam_inputs = self.segmenter_processor(images=picture, input_boxes=[[list(box)]],
                                                   return_tensors="pt").to(self.device)
            with torch.inference_mode():
                sam_output = self.segmenter(**sam_inputs, multimask_output=False)
            raw_mask = self.segmenter_processor.post_process_masks(
                sam_output.pred_masks.cpu(), sam_inputs["original_sizes"])[0][0][0]
            mask = np.asarray(raw_mask, dtype=np.bool_)
            if mask.shape != (height, width):
                raise VisionProtocolError("invalid_mask", "SAM mask dimensions differ")
            clipped = np.zeros_like(mask)
            clipped[box[1]:box[3], box[0]:box[2]] = mask[box[1]:box[3], box[0]:box[2]]
            if not clipped.any():
                raise VisionProtocolError("invalid_mask", "SAM mask is empty")
            segmentation_score = float(sam_output.iou_scores.reshape(-1)[0])
            if not math.isfinite(segmentation_score):
                continue
            occupancy = int(clipped.sum()) / ((box[2] - box[0]) * (box[3] - box[1]))
            occluded = occupancy < 0.35 or box[0] == 0 or box[1] == 0 or box[2] == width or box[3] == height
            regions.append(Region(category, box, clipped, float(score),
                                  min(1.0, max(0.0, segmentation_score)), occluded))
        return regions
