cd /home/rc1/inspect-robots
source .venv/bin/activate

# 0. SO101 机器上先确认插件。YAM 机器未安装 so101 插件是正常的。
inspect-robots list embodiments  # SO101 机器应包含 so_arm
inspect-robots list policies     # 应包含 agent
inspect-robots-so101-preflight --dry-run

# 1. `_run_so101.py` 仍有必要：inspect-robots-so101 的 camera_configs 是对象，
# 不能直接写进 CLI config。这个薄层只注册带 OpenCV 相机的 so101_configured，
# 最终评测仍由标准 Inspect Robots CLI 执行。
export SO101_CONFIG="$PWD/.so101/config.ini"
export SO101_PORT='/dev/serial/by-id/REPLACE_WITH_YOUR_FOLLOWER'
export SO101_ROBOT_ID='my_so101'
export SO101_CAMERA='/dev/v4l/by-id/REPLACE_WITH_YOUR_CAMERA-video-index0'
export SO101_CAMERA_NAME='front'
export SO101_CALIBRATION_DIR="$PWD/.so101/calibration"

python _so101_instructions/_run_so101.py setup --config "$SO101_CONFIG"
python _so101_instructions/_run_so101.py doctor \
  --config "$SO101_CONFIG" --embodiment so101_configured

# 2. 若尚未校准，先用 LeRobot 校准。已有校准时不要覆盖它。
# lerobot-calibrate \
#   --robot.type=so101_follower \
#   --robot.port="$SO101_PORT" \
#   --robot.id="$SO101_ROBOT_ID" \
#   --robot.calibration_dir="$SO101_CALIBRATION_DIR"

# 真实评测强制要求六维上下限。请填入这台臂实测的五个关节角（degrees）和
# gripper 0..100 范围；不要复制其他机械臂的数值。
export SO101_JOINT_LOW='[REPLACE_WITH_6_CALIBRATED_LOW_VALUES]'
export SO101_JOINT_HIGH='[REPLACE_WITH_6_CALIBRATED_HIGH_VALUES]'

# 3A. Claude 订阅账号。插件使用官方 Claude Code CLI 的登录，不读取其 token。
claude auth login       # 已登录可跳过
claude auth status
export SO101_CLAUDE_COMMAND="$(command -v claude)"
export SO101_CLAUDE_MODEL='claude-opus-5-5'

# --probe 使用合成图像/关节状态，不打开串口或相机；--dry-run 只打印命令。
python _so101_instructions/_eval_so101.py claude \
  --claude-command "$SO101_CLAUDE_COMMAND" --model "$SO101_CLAUDE_MODEL" --probe
python _so101_instructions/_eval_so101.py claude \
  --claude-command "$SO101_CLAUDE_COMMAND" --model "$SO101_CLAUDE_MODEL" --dry-run
python _so101_instructions/_eval_so101.py claude \
  --claude-command "$SO101_CLAUDE_COMMAND" --model "$SO101_CLAUDE_MODEL"

# 3B. GPT / Gemini / Grok API。用隐藏输入读取 key，避免写入脚本或 shell history。
read -rsp 'OPENAI_API_KEY: ' OPENAI_API_KEY; export OPENAI_API_KEY; printf '\n'
export SO101_GPT_MODEL='openai/gpt-6-astra'
python _so101_instructions/_eval_so101.py gpt --model "$SO101_GPT_MODEL" --probe
python _so101_instructions/_eval_so101.py gpt --model "$SO101_GPT_MODEL" --dry-run
python _so101_instructions/_eval_so101.py gpt --model "$SO101_GPT_MODEL"

read -rsp 'GEMINI_API_KEY: ' GEMINI_API_KEY; export GEMINI_API_KEY; printf '\n'
export SO101_GEMINI_MODEL='google/gemini-3.8-flash'
python _so101_instructions/_eval_so101.py gemini --model "$SO101_GEMINI_MODEL" --probe
python _so101_instructions/_eval_so101.py gemini --model "$SO101_GEMINI_MODEL" --dry-run
python _so101_instructions/_eval_so101.py gemini --model "$SO101_GEMINI_MODEL"

read -rsp 'XAI_API_KEY: ' XAI_API_KEY; export XAI_API_KEY; printf '\n'
export SO101_GROK_MODEL='x-ai/grok-4.7'
python _so101_instructions/_eval_so101.py grok --model "$SO101_GROK_MODEL" --probe
python _so101_instructions/_eval_so101.py grok --model "$SO101_GROK_MODEL" --dry-run
python _so101_instructions/_eval_so101.py grok --model "$SO101_GROK_MODEL"

# 默认真实任务：拿起瓶子、稍微提起、直立放回原位置，然后松爪。
# 每次只有 1 epoch，人工确认/评分，保存相机帧；guardrails 保持开启。
# 默认 SO101 每步 1 degree、max_speed_frac=0.05，可按实测进一步调小：
# python _so101_instructions/_eval_so101.py claude \
#   --max-action-delta 0.5 --max-speed-frac 0.03

inspect-robots view logs/so101
# inspect-robots inspect logs/so101/实际日志.json --transcript

# 更完整的账号原理、校准检查和故障排查见：
# _so101_instructions/_instructions_llm_accounts.md
