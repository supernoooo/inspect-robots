# archi
```text
终端 A：openpi/.venv
pi05_droid server，端口 8000
             │ WebSocket
             ▼
终端 B：inspect-isaac conda 环境
Inspect Robots + OpenPI client + Isaac Lab + Franka 仿真
```

# start pi0.5 server
```bash
cd /home/jjn/jjn/proj/vla/openpi
source .venv/bin/activate

uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05_droid \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_droid \
  --port=8000
```

# start isaac env
```bash
conda activate env_isaaclab
```