# inspect-robots-isaacsim-2yam

An independent Isaac Lab embodiment for MolmoAct2's two-arm YAM checkpoint. It
does not edit, subclass, or replace the existing Franka `isaacsim` plugin.

The plugin includes:

- embodiment `isaacsim-2yam`;
- task `isaacsim-2yam-put-everything-in-box`;
- 14-D YAM action/state packing;
- top, left-wrist, and right-wrist RGB cameras at `360x640`;
- a direct Isaac Lab environment with two objects, an open box, and privileged
  success detection.

## Install

Install Inspect Robots, this plugin, and the YAM policy plugin into the Python
environment that already imports Isaac Lab and Isaac Sim. For example:

```bash
cd /path/to/inspect-robots
uv pip install --python /path/to/isaaclab/bin/python \
  -e . \
  -e plugins/inspect-robots-isaacsim-2yam \
  inspect-robots-yam
```

Download the YAM MJCF from the MolmoAct2 checkout:

```bash
cd /path/to/molmoact2
uv run python sim_eval/scripts/download_assets.py
export MOLMOACT2_ROOT="$PWD"
```

The resolved asset must be
`$MOLMOACT2_ROOT/sim_eval/assets/yam/yam_mujoco/bimanual_yam_linear_flattened.xml`.
Alternatively set `MOLMOACT2_YAM_MJCF` or pass the absolute file path with
`-E asset_path=...`.

The embodiment prepares a private temporary copy of the MJCF and its meshes for
Isaac's importer. MolmoAct2's source assets are never changed.

## Start the model server

Start the YAM server, not the DROID server:

```bash
cd /path/to/molmoact2
uv run python examples/yam/host_server_yam.py \
  --host 0.0.0.0 --port 8202 --dtype bfloat16
```

`GET /act` must report both `checkpoint` and the resolved immutable
`revision`. Inspect Robots probes it once at run start and stores the response
under `eval.policy_server` in the JSON/HTML report. Restart an already-running
server after updating `host_server_yam.py`; otherwise the old process cannot
expose these new fields.

## Run an evaluation

Run Inspect Robots with the Isaac Lab interpreter or its activated environment:

```bash
/path/to/isaaclab/bin/inspect-robots run \
  --task isaacsim-2yam-put-everything-in-box \
  --policy molmoact2 \
  --embodiment isaacsim-2yam \
  --save-video \
  -P server_url=http://127.0.0.1:8202 \
  -P cam_height=360 \
  -P cam_width=640 \
  -E asset_path=/absolute/path/to/bimanual_yam_linear_flattened.xml
```

Use the server machine's reachable IP instead of `127.0.0.1` when inference and
Isaac Sim run on different machines.

The action order is `[left j0..j5, left gripper, right j0..j5, right gripper]`.
Arm values are absolute radians. Gripper `0` is closed and `1` is open.

## Headless cameras and video

Headless Isaac Sim still renders all three policy cameras: the embodiment always
starts Kit with `enable_cameras=True`. It also requests one RTX re-render after
every reset so the first image of a new episode matches the reset physics state
instead of repeating the preceding episode's final image.

`--save-video` is model-independent. It enables frame storage for the run and,
after Isaac Sim has closed, invokes ffmpeg to write one MP4 per trial and camera
under:

```text
logs/YYYYMMDD_runNNNN/videos/
```

The raw `.npy` frames remain in the same run's `frames/` directory. To encode an
older run, use `inspect-robots video LOG.json --out RUN_DIR/videos`.

The browser opened by `inspect-robots view` is an evaluation report, not the
native Isaac Sim viewport. It now shows a sampled camera flipbook even for
ordinary VLA policies that do not produce a chat transcript. On the SSH host:

```bash
/path/to/isaaclab/bin/inspect-robots view logs/ --serve \
  --host 127.0.0.1 --port 8300 --force
```

Then forward the port from your laptop with `ssh -L 8300:127.0.0.1:8300 ...`
and open `http://127.0.0.1:8300`. `--force` regenerates reports created before
camera-only flipbooks were supported. A native interactive Isaac window instead
requires a working graphical display and `-E headless=false`; it is not needed
for camera capture or MP4 export.

## Smoke test

Before a full model evaluation, verify that Isaac can import the robot and return
the expected observation shapes:

```bash
/path/to/isaaclab/bin/python \
  plugins/inspect-robots-isaacsim-2yam/scripts/boot_proof.py \
  --asset-path /absolute/path/to/bimanual_yam_linear_flattened.xml
```

Expected keys and shapes are `top_cam`, `left_cam`, and `right_cam`, each
`(360, 640, 3)`, plus state `(14,)`. The smoke test performs a step and a second
reset, and fails if that reset returns an unchanged stale camera buffer.

To save visual evidence as PNGs (and print each camera's pixel range), add:

```bash
--output-dir /tmp/isaacsim-2yam-camera-proof
```

The camera quaternions come from ManiSkill/SAPIEN and are interpreted as
`+X` forward, `+Z` up poses. The Isaac adapter marks them with Isaac Lab's
`world` camera convention so they are converted to USD/OpenGL correctly.

> [!IMPORTANT]
> The Isaac task uses local primitive collision/visual proxies for the Duplo and
> tennis ball so it remains independent of remote Isaac asset servers. It has the
> same control and observation contract as MolmoAct2's YAM task, but it is not a
> pixel-identical port of ManiSkill's YCB assets. Treat its success rate as an
> Isaac-domain evaluation, not as directly comparable to the ManiSkill number.
