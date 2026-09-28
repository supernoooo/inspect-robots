# 分步骤操作笔记；请按需复制相应命令，完整账号/模型说明见 _instructions_llm_accounts.md。

cd /home/magiclab/inspect-robots
source .venv/bin/activate

# install plugin
uv pip install --python .venv/bin/python --no-sources \
  'inspect-robots==0.59.0' \
  -e ./plugins/inspect-robots-agent


uv pip check --python .venv/bin/python
inspect-robots list embodiments  # 应包含 so_arm
inspect-robots list policies     # 应包含 lerobot、agent
inspect-robots-so101-preflight --dry-run
inspect-robots doctor --embodiment so_arm

python _so101_instructions/_run_so101.py setup --config "$PWD/.so101/config.ini"

# check config
python _so101_instructions/_run_so101.py doctor \
  --embodiment so101_configured \
  --config "$PWD/.so101/config.ini"




# setup CAN and camera
export SO101_PORT='/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E114717-if00'
export SO101_ROBOT_ID=my_so101
export SO101_CAMERA='/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0'
export SO101_CAMERA_NAME=front



# calibration

export SO101_ROBOT_ID=follower_arm
export SO101_CALIBRATION_DIR="$HOME/.cache/huggingface/lerobot/calibration/robots/so_follower"

# /home/magiclab/inspect-robots/.so101/calibration/my_so101.json
export SO101_CALIBRATION_DIR="$PWD/.so101/calibration"
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port="$SO101_PORT" \
  --robot.id="$SO101_ROBOT_ID" \
  --robot.calibration_dir="$SO101_CALIBRATION_DIR"



# check setup
python _so101_instructions/_run_so101.py doctor --embodiment so101_configured --config "$PWD/.so101/config.ini"



### agent

# cc
export SO101_CLAUDE_COMMAND="$PWD/.so101/claude-cli/claude_agent_sdk/_bundled/claude"
"$SO101_CLAUDE_COMMAND" auth login
"$SO101_CLAUDE_COMMAND" auth status
export SO101_CLAUDE_MODEL='claude-opus-5-5'

# gpt
export OPENAI_API_KEY=
python _so101_instructions/_so101_llm_probe.py --wire responses --model "$SO101_GPT_MODEL"
export SO101_GPT_MODEL='openai/gpt-6-astra'

# gemini
export GEMINI_API_KEY=
export SO101_GEMINI_MODEL='google/gemini-3.1-pro'
# grok
export XAI_API_KEY=
export SO101_GROK_MODEL='x-ai/grok-4.7'




## eval

# cc
python _so101_instructions/_run_so101.py run \
  --config "$PWD/.so101/config.ini" \
  --embodiment so101_configured \
  --policy agent \
  -P wire=claude-code \
  -P model="$SO101_CLAUDE_MODEL" \
  -P claude_command="$SO101_CLAUDE_COMMAND" \
  -P images=always \
  -P max_speed_frac=0.05 \
  -P max_llm_calls=60 \
  --instruction 'Pick up the bottle from the table and then place it back to its original place.' \
  --max-action-delta 1.0 \
  --max-steps 1200 \
  --epochs 1 \
  --grader operator \
  --scorer operator \
  --store-frames \
  --no-rerun \
  --no-rerun-save \
  --log-dir logs/so101/claude



# gpt
python _so101_instructions/_run_so101.py run \
  --config "$PWD/.so101/config.ini" \
  --embodiment so101_configured \
  --policy agent \
  -P wire=responses \
  -P model="$SO101_GPT_MODEL" \
  -P images=always \
  -P max_speed_frac=0.05 \
  -P max_llm_calls=5 \
  --instruction 'Move shoulder_pan approximately 2 degrees from its measured position, keep the other joints and gripper unchanged, then finish.' \
  --max-action-delta 1.0 --max-steps 30 --epochs 1 \
  --grader operator --scorer operator \
  --store-frames --no-rerun --no-rerun-save \
  --log-dir logs/so101/gpt



# 在交互终端运行。插件连接后先等待 Enter，确认场景再开始。
# 结束后根据实际完成情况填写 y / n / partial / skip 及备注，CLI 写入分数和日志。
# GPT 上面的示例仍是小动作；完整瓶子任务使用下面的统一启动入口。
# Ctrl+C 中断；机械臂由 CLI 清理并断开。终止操作不等同于成功评分。

# GPT 评测继续读取 OPENAI_API_KEY，不使用 Claude 登录；保留以上相同的机械臂与动作参数，
# 改为 -P wire=responses -P model=openai/你的视觉模型ID，并删除 -P claude_command=...。
# GPT 首先也可运行无硬件探针；完整评测命令见 _instructions_llm_accounts.md。







# 6. 可选：评测训练好的 LeRobot checkpoint（ACT 示例）

export SO101_CHECKPOINT='/absolute/path/to/your/SO101_ACT_checkpoint'

python _so101_instructions/_run_so101.py run \
  --config "$PWD/.so101/config.ini" \
  --embodiment so101_configured \
  --policy lerobot \
  -P pretrained_path="$SO101_CHECKPOINT" \
  -P policy_type=act \
  -P device=cuda \
  -P chunk_size=5 \
  -P use_degrees=False \
  -E use_degrees=False \
  -E control_hz=30 \
  --instruction 'Reach for the cube' \
  --max-action-delta 1.0 \
  --max-steps 300 \
  --epochs 1 \
  --grader operator \
  --scorer operator \
  --store-frames \
  --no-rerun --no-rerun-save \
  --log-dir logs/so101

# policy_type=smolvla 用于匹配的 SmolVLA checkpoint；smolvla_base 并不保证完成你的任务。
# 确认首轮行为后，可增大 --epochs 做多次相同任务评测。
# 使用 ad-hoc instruction + operator scorer；cubepick-reach 的内置评分不是人工评分。


inspect-robots view logs/so101
# 把下面的路径替换为 run 打印的真实日志文件：
inspect-robots inspect logs/so101/REPLACE_WITH_LOG.json --transcript
uv pip freeze --python .venv/bin/python > /tmp/inspect-robots-so101-installed.txt



# 四种模型的完整瓶子任务。--dry-run 仅显示命令；去掉它才启动真实评测。
# 真实运行要求已有校准文件和 SO101_JOINT_LOW/HIGH。
python _so101_instructions/_eval_so101.py claude --model claude-opus-5-5 --dry-run
python _so101_instructions/_eval_so101.py gpt --model openai/gpt-6-astra --dry-run
bash _so101_instructions/_eval_so101_gemini.sh --model google/gemini-3.8-flash --dry-run
bash _so101_instructions/_eval_so101_grok.sh --model x-ai/grok-4.7 --dry-run
# Gemini/Grok 分别先 export GEMINI_API_KEY / XAI_API_KEY；--probe 只调用模型，不连接硬件。
