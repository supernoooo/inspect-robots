# inspect-robots-libero

An independent LIBERO plugin for evaluating a single-arm Franka with Inspect
Robots. It includes the `libero` embodiment, all five official suite task
factories, and the `molmoact2-libero` HTTP policy adapter.

## Install

LIBERO's published environment is old, while Inspect Robots requires Python
3.10 or newer. Keep this evaluation stack isolated from both MolmoAct2 and
Isaac Lab, and use Python 3.10 for the combined environment:

```bash
conda create -n libero-inspect python=3.10 -y
conda activate libero-inspect

git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install -e .

cd /path/to/inspect-robots
uv pip install --python "$(which python)" \
  -e . \
  -e plugins/inspect-robots-libero
```

The official LIBERO install creates `~/.libero/config.yaml`. Confirm that its
`bddl_files`, `init_states`, and `assets` entries point to the cloned checkout.
The rollout plugin needs task assets and benchmark initial states, but it does
not require demonstration datasets.

For headless rendering, configure the MuJoCo backend before running:

```bash
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
```

## Start MolmoAct2-LIBERO

The DROID server is not compatible with LIBERO. Start the LIBERO checkpoint in
the separate MolmoAct2 environment through its generic server:

```bash
cd /path/to/molmoact2
uv run python experiments/scripts/serve_policy.py \
  --checkpoint allenai/MolmoAct2-LIBERO \
  --hf_ckpt true \
  --image_keys libero \
  --norm_tag libero \
  --img_resize 256x256 \
  --n_action_steps 10 \
  --host 0.0.0.0 \
  --port 8204
```

## Run an evaluation

Run one LIBERO-Goal task with ten fixed benchmark initial states:

```bash
conda activate libero-inspect
export MUJOCO_GL=egl

inspect-robots run \
  --task libero-goal \
  --policy molmoact2-libero \
  --embodiment libero \
  -P server_url=http://127.0.0.1:8204 \
  -T task_ids=0 \
  -T init_state_ids=0,1,2,3,4,5,6,7,8,9
```

Use the server machine's reachable IP instead of `127.0.0.1` when inference and
LIBERO run on different machines. Remove `-T task_ids=0` to evaluate every task
in the suite. Registered tasks are `libero-spatial`, `libero-object`,
`libero-goal`, `libero-10`, and `libero-90`.

The adapter follows MolmoAct2's LIBERO preprocessing contract:

- raw LIBERO images are rotated 180 degrees;
- `image` maps from `agentview_image`;
- `wrist_image` maps from `robot0_eye_in_hand_image`;
- state is `[eef position (3), eef axis-angle (3), gripper qpos (2)]`;
- action is normalized relative `[x, y, z, roll, pitch, yaw, gripper]`.

> [!NOTE]
> The model server and simulator are intentionally separate environments. Only
> their HTTP `/act` contract is shared, which avoids mixing MolmoAct2's Torch
> dependencies with LIBERO's MuJoCo and robosuite dependencies.
