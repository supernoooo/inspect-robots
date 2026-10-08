# Jev vision service

Install this package in a separate Python environment from Isaac Lab:

From the repository root, install the client and service packages in a separate
vision environment. The service does not need the Isaac adapter:

```bash
python -m pip install -e . -e plugins/inspect-robots-jev \
  -e plugins/inspect-robots-jev/vision_service
inspect-robots-jev-vision --host 127.0.0.1 --port 8765
```

The service loads `IDEA-Research/grounding-dino-tiny` and
`facebook/sam2.1-hiera-tiny`. `--detector-revision` and
`--segmenter-revision` accept Hugging Face revisions. On startup it prints the
resolved snapshot commits returned in every response. `--local-files-only`
requires cached weights and does not fetch them.
Startup fails unless both snapshot paths resolve to 40-character commit IDs.
The service startup log and v2 responses include the resolved model IDs and
commits, prompt version, and service version. The service uses
`red block. tray.` for `right_red_block_tray` and the original
`red block. yellow ball. box.` for simulation.

`POST /v1/detect` version 1 accepts only version, request ID, camera name, client
monotonic `captured_at`, dimensions, and base64 row-major RGB bytes. The
implicit v1 profile remains the simulation `red_block/yellow_ball/box` task.
Version 2 requires `task_profile`: either `sim_put_everything_in_box` or
`right_red_block_tray`. Its response echoes that profile and permits only that
profile's categories; the tray profile permits only `red_block` and `tray`.
The response echoes request metadata exactly. Boxes are integer `[x1,y1,x2,y2]`
with exclusive right/bottom edges. Masks use row-major RLE alternating zero
and one runs, starting with zeros. Each detection carries category, two scores,
occlusion flag, and resolved model revisions. V2 also carries
`prompt_version` and `service_version` in `model_version`. The tray client
requires a pinned expected fingerprint and rejects a mismatch. Client age checks use its own
clock; the server never evaluates time freshness. HTTP 400/500 indicate
service failures. Protocol and inference failures have distinct error codes.
No joint state, depth, actions, or simulator data is accepted.

The model may misidentify the simulated objects. `occluded` is a conservative
mask occupancy or image-edge heuristic and needs visual review on actual
saved camera frames.

For a weight-free overlay check:

```bash
python plugins/inspect-robots-jev/scripts/check_vision.py frame.png \
  --output-dir /tmp/jev-vision-check --fake
```

`--fake` checks only protocol and drawing; it is not recognition evidence.
For saved, stationary rig RGB frames, create a manifest with original camera
names, dimensions, and capture timestamps (not file read time):

```json
{"frames": [
  {"path": "top.png", "camera": "top_cam", "captured_at": 123.125, "height": 224, "width": 224},
  {"path": "left.png", "camera": "left_cam", "captured_at": 123.250, "height": 224, "width": 224},
  {"path": "right.png", "camera": "right_cam", "captured_at": 123.375, "height": 224, "width": 224}
]}
```

Replace the example timestamps and dimensions with values from the saved
samples. For tray recognition, run:

```bash
python plugins/inspect-robots-jev/scripts/check_vision.py \
  --manifest /path/to/frames.json --profile right_red_block_tray \
  --detector-revision RESOLVED_40_HEX_COMMIT \
  --segmenter-revision RESOLVED_40_HEX_COMMIT \
  --output-dir /path/to/tray-review
```

The script checks image dimensions without resizing and sends original
`captured_at` unchanged. Offline saved-frame mode skips live age checks;
live policy calls still enforce them. The tray manifest must include top and
both wrist cameras. `results.json` and per-camera PNG overlays include frame
failure states, model fingerprint, masks and scores for usable or rejected
valid regions, and a `human_review` field initially marked `pending`. An
operator must inspect red block visibility, tray opening, mask fit, misses,
false positives, occlusion, and duplicate instances for each frame. A successful
2D overlay does not establish calibration or permit robot motion.
