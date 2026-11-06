# Run CaP-X code-as-policy on a real SO-101

This cookbook sets up the `capx` policy (LLM-written Python driving perception
and inverse kinematics) on a real SO-101 arm, from a fresh clone to a scored
pick-up. Every step before first motion can be checked off-robot.

In each turn the model sees the robot state and camera images, writes a short
Python snippet, and the snippet runs in-process with helpers bound to the
robot: `segment()` calls a SAM3 server, `solve_ik()` runs the arm's own IK,
and `move_to_joints()`, `open_gripper()`, and `close_gripper()` queue
speed-limited joint targets. The queued motion executes as one open-loop chunk
through the standard clamp and delta-limit guardrails, then the model sees the
result and writes the next snippet.

Two processes cooperate:

| Process | Environment | Role |
|---|---|---|
| SAM3 server | `servers/sam3/.venv` (Python 3.10, GPU) | Text-prompt segmentation over HTTP on port 8114 |
| `inspect-robots run` | `.venv` (Python 3.12) | LLM loop, code execution, IK, arm and cameras, guardrails, logging |

Two operating modes share the same code:

- **Closed loop (default):** no camera calibration. The model aligns the
  object between the jaws using the wrist camera and corrects every turn.
- **Calibrated:** with a camera-to-base transform, camera-frame points map
  directly to IK targets. See [Optional: calibrated mode](#optional-calibrated-mode).

## Prerequisites

- **Arm:** an SO-101 follower with a LeRobot calibration. A leader arm is
  optional (only for the hand-eye calibration tool).
- **Front camera:** an Intel RealSense D4xx (tested with a D435). Depth is
  required.
- **Wrist camera:** a USB camera mounted on the gripper (tested with a
  generic UVC camera at 640x480).
- **GPU:** an NVIDIA GPU for SAM3 (tested on an RTX 5090 Laptop GPU; the
  server holds
  about 5 to 7 GB of VRAM).
- **Software:** Linux, `git`, and [uv](https://docs.astral.sh/uv/).
- **LLM access:** an API key for at least one provider (Anthropic, OpenAI,
  xAI, or OpenRouter).

## 1. Clone and install the evaluator

```bash
git clone -b robocolosseum-capx https://github.com/<your-fork>/inspect-robots.git
cd inspect-robots
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e . -e ./plugins/inspect-robots-agent -e ./plugins/inspect-robots-capx \
    -e ./_so101_instructions
```

Keep the `./` prefixes: without them uv reads `_so101_instructions` as a
package name instead of a directory.

`_so101_instructions` is the SO-101 rig package. It pulls in the
`inspect-robots-so101` driver, LeRobot, `pyrealsense2`, and the placo IK
stack, and registers the `so101_capx_configured` embodiment. Check that both
components are registered:

```bash
inspect-robots list policies      # includes: capx
inspect-robots list embodiments   # includes: so101_capx_configured
```

## 2. Download the SO-101 model

IK and forward kinematics need the official SO-101 URDF and meshes. The
script fetches a pinned revision of
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
(Apache-2.0), checks the URDF checksum, and installs it to
`.so101/models/SO101/`:

```bash
python _so101_instructions/_fetch_so101_model.py
```

Optionally, run the offline test suites (no hardware, no network). The model
tests need the URDF from this step:

```bash
uv pip install pytest
python -m pytest plugins/inspect-robots-agent/tests plugins/inspect-robots-capx/tests \
    _so101_instructions -q
```

One test is expected to be skipped: it covers the optional Claude Code
subscription client, which this repository does not ship.

## 3. Start the SAM3 server

The server lives in `servers/sam3/` with its own environment, so PyTorch and
the SAM3 model code never enter the evaluator environment:

```bash
uv venv servers/sam3/.venv --python 3.10
uv pip install --python servers/sam3/.venv/bin/python -e servers/sam3
servers/sam3/.venv/bin/python -m sam3_server --port 8114
```

The first start downloads `sam3.pt` (3.45 GB) from
[ModelScope](https://modelscope.cn/models/facebook/sam3) into
`~/.cache/modelscope` (override with `MODELSCOPE_CACHE`); no Hugging Face
access request is needed. The server is ready when it logs
`Uvicorn running on http://127.0.0.1:8114`. Keep this terminal open.

Other weight sources:

- `--source hf` downloads from the gated `facebook/sam3` repository on
  Hugging Face (request access first, then `hf auth login`).
- `--checkpoint /path/to/sam3.pt` loads a local file.

The server code comes from [CaP-X](https://github.com/capgym/cap-x) (MIT, see
`servers/sam3/LICENSE-CAPX`). The SAM3 model code and weights are covered by
Meta's SAM License, which you accept by installing and downloading them.

## 4. Calibrate the arm motors

Calibrate the follower once with LeRobot, choosing an id you will reuse in
the config:

```bash
ls /dev/serial/by-id/       # find the follower's serial port
lerobot-calibrate --robot.type=so101_follower \
    --robot.port=/dev/serial/by-id/<follower-port> --robot.id=<follower-id>
```

The calibration lands in
`~/.cache/huggingface/lerobot/calibration/robots/so_follower/<follower-id>.json`.
The driver refuses to start if that file does not match the motors.

## 5. Configure the rig

```bash
mkdir -p .so101
cp _so101_instructions/capx-config.ini.example .so101/capx.ini
cp _so101_instructions/capx-rig-notes.example.md .so101/capx-rig-notes.md
```

`.so101/` is gitignored, so device names and paths stay local. In
`.so101/capx.ini`, replace every `<...>` placeholder:

| Key | How to find it |
|---|---|
| `port` | `ls /dev/serial/by-id/` |
| `camera_device` | `ls /dev/v4l/by-id/` (the wrist camera's `video-index0` entry) |
| `head_camera_serial` | `python -c "import pyrealsense2 as rs; print([d.get_info(rs.camera_info.serial_number) for d in rs.context().devices])"` |
| `robot_id` | The `<follower-id>` from step 4 |
| `home_pose` | Your arm's folded rest pose in calibrated degrees (see below) |
| `urdf_path`, `prior_learnings` | Absolute paths under your clone's `.so101/` |

For `home_pose`, the simplest start is to delete the `home_pose` line and set
`capture_home = True`: the pose measured at connection becomes home, and the
arm returns there at the end. Once you know your rest pose (for example from
the first frame of a run's `actions/*.jsonl`), pin it in `home_pose` so every
trial starts identically.

Then validate the configuration without moving anything:

```bash
inspect-robots doctor --config .so101/capx.ini
```

### Rig notes

`prior_learnings` points the model at `.so101/capx-rig-notes.md`, which is
appended to its system prompt. The example file describes the reference rig,
and its numbers will be wrong for yours. Re-measure and edit:

- **Camera layout:** where the RealSense sits relative to the arm, and
  whether the raised arm occludes it.
- **Wrist image geometry:** which jaw is fixed, where it appears in the wrist
  image, and the side the moving jaw closes from. Read these off a stored
  frame (`logs/.../frames/*_wrist_*.npy`) or the Rerun viewer.
- **Table geometry:** table height in the base frame (the folded rest pose's
  `eef_pos` z is a good proxy) and object sizes.

Keep the notes factual. They steer every model equally, so mistakes in them
show up as systematic failures (an earlier version pointed the model at the
wrong spot along the fingers and every grasp closed empty).

## 6. Add your API keys

```bash
cp .env.example .env
```

Fill in the providers you use. `inspect-robots` loads `./.env` automatically
when run from the repository root, and real environment variables take
precedence. `.env` is gitignored. To export the same keys into a shell for
other tools, run `set -a; source .env; set +a`.

## 7. Run

Run a perception-only smoke test first. The arm ramps to `home_pose`, the
model segments the object, and nothing else moves:

```bash
inspect-robots run --config .so101/capx.ini \
  -P model=x-ai/grok-4.7 -P wire=responses -P effort=medium \
  --instruction "Do not move. segment('blue block') on the front camera, print the top score, pixel centroid and median depth, then FINISH." \
  --log-dir logs/so101/capx-smoke
```

Then a pick-up:

```bash
inspect-robots run --config .so101/capx.ini \
  -P model=x-ai/grok-4.7 -P wire=responses -P effort=medium \
  --instruction "Pick up the blue block from the table and lift it about 5 cm." \
  --log-dir logs/so101/capx-grok
```

To compare models, change only the model arguments and the log directory:

| Provider | Arguments |
|---|---|
| Anthropic | `-P model=anthropic/<model-id>` |
| OpenAI | `-P model=openai/<model-id> -P wire=responses` |
| xAI | `-P model=x-ai/<model-id> -P wire=responses` |

Name the object as it looks: SAM3 scores a cyan block near 0.97 as
`blue block` and near 0 as `red block`.

The driver asks for confirmation before the home ramp and before the trial
starts. Press `Esc` in the terminal at any time to end the episode; the arm
returns home before torque is released. After the run you are asked for a
success verdict.

## 8. Review a run

```bash
inspect-robots inspect logs/so101/capx-grok/<run>/<run>.json --transcript
inspect-robots view logs/so101/capx-grok/<run>/<run>.json
```

Each run directory holds the log (with every code turn, its stdout and
stderr), `actions/*.jsonl` (every joint command), `frames/` (every camera
frame), and a Rerun `.rrd` recording.

## Settings that matter

These are set in `.so101/capx.ini`; each can be overridden per run with
`-P key=value` (policy) or `-E key=value` (embodiment).

| Setting | Value | Why |
|---|---|---|
| `ik` | `embodiment` | Uses the arm's placo IK in calibrated degrees; no Pyroki server |
| `max_speed_frac` | `0.05` | About 1 degree per 10 Hz step for the arm |
| `gripper_speed_frac` | `0.5` | A full open or close in about 2 s instead of about 20 s |
| `image_horizon` | `2` | Only the two newest observations keep images, which bounds request size |
| `max_llm_calls` | `60` | Closed-loop grasps spend turns probing and correcting |
| `max_steps` | `2000` | About 200 s of motion at 10 Hz |
| `home_tolerance`, `settle_tolerance` | `2.0` | The extended arm holds about 1 to 1.5 degrees off target under gravity |

## Optional: calibrated mode

With a camera-to-base transform, the model can map camera-frame points and
Contact-GraspNet grasps straight into IK targets. The repository includes an
eye-to-hand calibration tool that needs the leader arm, an ArUco marker taped
near the gripper tip, and the Rerun viewer:

```bash
python _so101_instructions/_so101_handeye.py --print-marker marker.png
python _so101_instructions/_so101_handeye.py --config .so101/capx.ini --fresh
python _so101_instructions/_so101_handeye.py --config .so101/capx.ini --verify
```

Capture 12 to 15 poses spread over the workspace with varied wrist angles.
The tool reports per-sample residuals and writes
`.so101/front_camera_to_base.json`. Then set `camera_extrinsics` to that file
in `.so101/capx.ini`.

## Troubleshooting

- **`Could not connect on port ...`:** the configured serial port is not
  plugged in. Check `ls /dev/serial/by-id/` against `port` (and `leader_port`
  if set).
- **`home pose not reached ... error=1.2 degrees`:** a joint holds slightly
  off target under load. Raise `home_tolerance`.
- **`settle timeout budget exhausted`:** the same effect during the trial.
  Raise `settle_tolerance`.
- **SAM3 connection errors:** the server is not running or not on port 8114;
  the error names the URL that failed.
- **`HTTP 401` from the LLM provider:** the key in `.env` is wrong or
  expired.
- **`segment()` scores near 0:** the prompt does not match the object (check
  its color), or the object is occluded in that camera.
- **The trial ends at the step limit:** raise `max_steps`, or check that
  `gripper_speed_frac` is set (slow gripper ramps used to consume most of the
  budget).

## Safety

Model-written Python runs in the evaluator process with your user's
privileges. It is not a sandbox: it can read files, open network
connections, and import any installed package. Every robot action still goes
through the clamp and delta-limit guardrails, but run untrusted models inside
a container. Stay at the robot during every run.
