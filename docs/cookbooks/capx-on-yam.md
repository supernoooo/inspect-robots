# Run CaP-X code-as-policy on a real YAM arm

This cookbook sets up the `capx` policy (LLM-written Python driving perception
and inverse kinematics) on one arm of a real
[I2RT YAM](https://i2rt.com/products/yam-6-dof-arm) bimanual rig, from a fresh
clone to a scored pick-up. It mirrors the [SO-101 cookbook](capx-on-so101.md):
the two rigs share the policy, the SAM3 server, the execution model, and the
`CapxArmEmbodiment` adapter base from `inspect-robots-capx`. Only the
robot-specific parts differ, and this page covers them.

In each turn the model sees the robot state and camera images, writes a short
Python snippet, and the snippet runs in-process with helpers bound to the
robot: `segment()` calls a SAM3 server, `solve_ik()` runs a mink solver over
i2rt's MuJoCo model of the arm, and `move_to_joints()`, `open_gripper()`, and
`close_gripper()` queue speed-limited joint targets. The queued motion executes
as one open-loop chunk through the standard clamp and delta-limit guardrails,
then the model sees the result and writes the next snippet.

`capx` drives a single arm. The `yam_capx_configured` embodiment presents the
left or right arm (`arm = left|right`) as a 7-D joint profile, holds the other
arm at its start pose for the whole trial, and keeps YAM's homing, operator
gates, thermal guardrail, and parking unchanged.

Two processes cooperate:

| Process | Environment | Role |
|---|---|---|
| SAM3 server | `servers/sam3/.venv` (Python 3.10, GPU) | Text-prompt segmentation over HTTP on port 8114 |
| `inspect-robots run` | `.venv` | LLM loop, code execution, IK, arms and cameras, guardrails, logging |

Two operating modes share the same code:

- **Closed loop (default):** no camera calibration. The model aligns the
  object between the jaws using the arm's wrist camera and corrects every turn.
- **Calibrated:** with a camera-to-base transform, camera-frame points map
  directly to IK targets. See [Optional: calibrated mode](#optional-calibrated-mode).

## Prerequisites

- **Arms:** a YAM bimanual rig set up per the
  [inspect-robots-yam README](https://github.com/robocurve/inspect-robots-yam#run-on-hardware):
  CAN channels up, `inspect-robots-yam-health` green. Both arms connect even
  though one is driven.
- **Cameras:** three Intel RealSense cameras (the standard top D435 and two
  wrist D405s). Depth is required on the camera `segment()` uses, so configure
  that slot as `*_depth_serial`, not `*_cam_device`.
- **GPU:** an NVIDIA GPU for SAM3 (about 5 to 7 GB of VRAM).
- **Software:** Linux, `git`, and [uv](https://docs.astral.sh/uv/).
- **LLM access:** an API key for at least one provider (Anthropic, OpenAI,
  xAI, or OpenRouter).

## 1. Clone and install the evaluator

```bash
git clone -b robocolosseum-capx https://github.com/<your-fork>/inspect-robots.git
cd inspect-robots
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e . -e ./plugins/inspect-robots-agent -e ./plugins/inspect-robots-capx \
    -e ./_yam_instructions
```

Keep the `./` prefixes: without them uv reads `_yam_instructions` as a
package name instead of a directory.

`_yam_instructions` is the YAM rig package. It pulls in
`inspect-robots-yam[depth]` (the driver adapter plus `pyrealsense2`), `mink`,
and `mujoco`, and registers the `yam_capx_configured` embodiment. The i2rt
driver is git-only and carries the robot model used for IK, so install it
separately with the pinned command from the YAM README:

```bash
echo 'scikit-build-core<0.10' > build-constraints.txt
uv pip install --build-constraints build-constraints.txt \
    "i2rt @ git+https://github.com/i2rt-robotics/i2rt@ac096928d6899ddf852a71c5e8fbaa6055cd9745"
```

Check that both components are registered:

```bash
inspect-robots list policies      # includes: capx
inspect-robots list embodiments   # includes: yam_capx_configured
```

Optionally, run the offline test suites (no hardware, no network). The
real-kinematics tests need `i2rt`, `mink`, and `mujoco` from the step above:

```bash
uv pip install pytest
python -m pytest plugins/inspect-robots-agent/tests plugins/inspect-robots-capx/tests \
    _yam_instructions -q
```

## 2. Start the SAM3 server

The server lives in `servers/sam3/` with its own environment, so PyTorch and
the SAM3 model code never enter the evaluator environment. Follow
[step 3 of the SO-101 cookbook](capx-on-so101.md#3-start-the-sam3-server);
the server is identical. It is ready when it logs
`Uvicorn running on http://127.0.0.1:8114`.

## 3. Configure the rig

```bash
mkdir -p .yam
cp _yam_instructions/capx-config.ini.example .yam/capx.ini
cp _yam_instructions/capx-rig-notes.example.md .yam/capx-rig-notes.md
```

`.yam/` is gitignored, so device names and paths stay local. In
`.yam/capx.ini`, replace every `<...>` placeholder:

| Key | How to find it |
|---|---|
| `left_channel`, `right_channel` | `ip link show` (or reuse the values `inspect-robots setup` wrote for `yam_arms`) |
| `top_depth_serial`, `left_depth_serial`, `right_depth_serial` | `rs-enumerate-devices`; quote all-digit serials |
| `arm` | `left` or `right`: the arm the model drives |
| `depth_camera` | The `*_depth_serial` camera `segment()` and `plan_grasp()` use (default `top_cam`) |
| `prior_learnings` | Absolute path under your clone's `.yam/` |

Settings that differ from a VLA run on the same rig:

| Setting | Value | Why |
|---|---|---|
| `cam_width`, `cam_height` | `640`, `480` | SAM3 needs more than the 224x224 VLA default; intrinsics are rescaled automatically |
| `collision_guardrail` | `false` until measured | The MuJoCo guardrail reviews the full two-arm command with the idle arm held; unmeasured base offsets cause false holds |
| `settle_tolerance` | `0.02` | Observe only after the arm stops, so images match `joint_pos` |
| `max_speed_frac` | `0.03` | About 1 degree per 10 Hz step over the default joint range |

Then validate the configuration without moving anything:

```bash
inspect-robots doctor --config .yam/capx.ini
```

### Rig notes

`prior_learnings` points the model at `.yam/capx-rig-notes.md`, which is
appended to its system prompt. The example file is a template: every
`<...>` must be measured on your rig. Fill in:

- **Camera layout:** where the top RealSense sits relative to the driven arm's
  base, and whether the raised arm occludes it.
- **Wrist image geometry:** where the jaws appear in the wrist image and the
  pixel band in which a grasp succeeded. Read these off stored frames
  (`logs/.../frames/*_left_cam_*.npy`) or the Rerun viewer.
- **Table geometry:** table height in the driven arm's base frame and object
  sizes. The folded home pose's `eef_pos` z is a good proxy for the surface.

Keep the notes factual: they steer every model equally, so mistakes in them
show up as systematic failures.

## 4. Add your API keys

```bash
cp .env.example .env
```

Fill in the providers you use. `inspect-robots` loads `./.env` automatically
when run from the repository root, and real environment variables take
precedence. `.env` is gitignored.

## 5. Run

Run a perception-only smoke test first. Both arms ramp to the home pose, the
model segments the object, and nothing else moves:

```bash
inspect-robots run --config .yam/capx.ini \
  -P model=x-ai/grok-4.7 -P wire=responses -P effort=medium \
  --instruction "Do not move. segment('blue block') on the top camera, print the top score, pixel centroid and median depth, then FINISH." \
  --log-dir logs/yam/capx-smoke
```

Then a pick-up:

```bash
inspect-robots run --config .yam/capx.ini \
  -P model=x-ai/grok-4.7 -P wire=responses -P effort=medium \
  --instruction "Pick up the blue block from the table and lift it about 5 cm." \
  --log-dir logs/yam/capx-grok
```

To drive the other arm, add `-E arm=right` (the policy's `camera` stays the
top RealSense unless you point it at that arm's wrist camera with
`-P camera=right_cam -E depth_camera=right_cam`). To compare models, change
only the model arguments and the log directory:

| Provider | Arguments |
|---|---|
| Anthropic | `-P model=anthropic/<model-id>` |
| OpenAI | `-P model=openai/<model-id> -P wire=responses` |
| xAI | `-P model=x-ai/<model-id> -P wire=responses` |

The driver asks for confirmation before the home ramp and before the trial
starts. Press `Esc` in the terminal at any time to end the episode; both arms
park before torque is released. After the run you are asked for a success
verdict.

## 6. Review a run

```bash
inspect-robots inspect logs/yam/capx-grok/<run>/<run>.json --transcript
inspect-robots view logs/yam/capx-grok/<run>/<run>.json
```

Each run directory holds the log (with every code turn, its stdout and
stderr), `actions/*.jsonl` (every 7-D joint command of the driven arm),
`frames/` (every camera frame), and a Rerun `.rrd` recording. The observation
keeps the full two-arm vector under `bimanual_joint_pos` next to the driven
arm's `joint_pos`.

## What the model sees

The embodiment notes injected into the system prompt describe the driven arm:
six joints in radians plus a 0..1 gripper, `solve_ik()` over the `grasp_site`
frame (between the fingertips) in that arm's base frame, the tool axes (+z is
the approach direction, +y the jaw axis; at zero joints the tool points
forward along base +x, and a straight-down approach is close to
`quaternion_wxyz [0, 0, 1, 0]`), and the closed-loop probing procedure when no
camera calibration exists. The upstream YAM joint guide follows, so the
positive direction of every joint is spelled out.

## Optional: calibrated mode

With a camera-to-base transform for `depth_camera` in the **driven arm's**
base frame, the model can map camera-frame points and Contact-GraspNet grasps
straight into IK targets. Set `camera_extrinsics` in `.yam/capx.ini` to a
JSON (`{"camera_to_base": [[...], ...]}` or a bare 4x4 list) or `.npy` file;
the matrix is checked for rigidity before hardware connects. The SO-101
hand-eye tool is LeRobot-specific and does not run on YAM; derive the transform
with your preferred eye-to-hand calibration against the arm's measured
`eef_pos` (the grasp-site FK, published in every observation).

## Troubleshooting

- **`yam_arms has no cameras configured`:** every camera slot needs exactly
  one source. Set the three `*_depth_serial` values, or `*_cam_device` for a
  slot without depth.
- **`depth_camera 'top_cam' needs top_depth_serial`:** the camera
  `segment()` uses was configured through V4L2, which has no depth. Switch
  that slot to its RealSense serial.
- **`i2rt provides the YAM robot model used for IK`:** the git-only driver is
  not installed in this environment; run the pinned install command from
  step 1.
- **`YAM left arm IK cannot reach position ...`:** the target is outside the
  arm's reach or joint limits. The model sees this on stderr and should pick a
  closer target; recurring failures usually mean the rig notes' table height
  or reach numbers are wrong.
- **The collision guardrail holds every move:** `collision_guardrail = true`
  with unmeasured `collision_*_base_pos`. Measure the base offsets (see the
  YAM README) or set it to `false` and keep an operator at the e-stop.
- **SAM3 connection errors:** the server is not running or not on port 8114;
  the error names the URL that failed.
- **`segment()` scores near 0:** the prompt does not match the object (check
  its color), or the object is occluded in that camera.

## Safety

Model-written Python runs in the evaluator process with your user's
privileges. It is not a sandbox: it can read files, open network
connections, and import any installed package. Every robot action still goes
through the clamp and delta-limit guardrails, and the idle arm never receives
a target other than its start pose, but run untrusted models inside a
container. Stay at the robot during every run.
