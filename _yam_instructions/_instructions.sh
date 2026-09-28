cd /home/rc1/inspect-robots
source .venv/bin/activate


python -c 'import importlib.metadata as m, inspect_robots_agent as a; print(m.version("inspect-robots-agent"), a.__file__)'
inspect-robots list embodiments
inspect-robots list policies


# check yam config
export YAM_CONFIG="$PWD/.yam/config.ini"
mkdir -p "$PWD/.yam"
inspect-robots setup --config "$YAM_CONFIG"
inspect-robots doctor --config "$YAM_CONFIG" --embodiment yam_arms
INSPECT_ROBOTS_CONFIG="$YAM_CONFIG" inspect-robots-yam-health


# can port
ip -details -statistics link show type can

sudo ip link set can_follower_l down
sudo ip link set can_follower_l up type can bitrate 1000000
sudo ip link set can_follower_r down
sudo ip link set can_follower_r up type can bitrate 1000000

.venv/bin/inspect-robots view logs/yam/claude --open
.venv/bin/inspect-robots view logs/yam/claude --serve
ps -p 29244 -o pid,args
kill -TERM 29244
ss -ltnp 'sport = :8300'


### agent
# epochs: try # times for each task
# fail-on-error: stop eval after #*total PolicyError

# cc
claude auth login
claude auth status
export YAM_COMMAND="Pick up the candy box from the table and place it into the red plate."
export YAM_CLAUDE_MODEL='claude-opus-5-5'

python _yam_instructions/_eval_yam.py claude \
  --claude-command "$YAM_COMMAND" \
  --model "$YAM_CLAUDE_MODEL"
# python _yam_instructions/_eval_yam.py claude-api --model anthropic/claude-opus-5-5 --probe
# python _yam_instructions/_eval_yam.py claude-api --model anthropic/claude-opus-5-5

inspect-robots run \
  --config "$YAM_CONFIG" \
  --embodiment yam_arms \
  --policy agent \
  -P wire=claude-code \
  -P model="${YAM_CLAUDE_MODEL:?NO MODEL SELECTED}" \
  -P effort=medium \
  -P images=always \
  -P max_speed_frac=0.05 \
  -P max_llm_calls=60 \
  -E auto_start=False \
  -E unattended=False \
  --instruction "${YAM_COMMAND:?NO INSTRUCTION INPUT}" \
  --max-action-delta 0.05 \
  --max-steps 800 \
  --epochs 1 \
  --grader operator \
  --scorer operator \
  --store-frames \
  --rerun \
  --rerun-save \
  --fail-on-error 1 \
  --log-dir logs/yam/claude




# gpt
export OPENAI_API_KEY=
export YAM_GPT_MODEL='openai/gpt-6-astra'
# python _yam_instructions/_eval_yam.py gpt --model "$YAM_GPT_MODEL" --probe
python _yam_instructions/_eval_yam.py gpt --model "$YAM_GPT_MODEL" --dry-run
# python _yam_instructions/_eval_yam.py gpt --model "$YAM_GPT_MODEL"


inspect-robots run \
  --config "$YAM_CONFIG" \
  --embodiment yam_arms \
  --policy agent \
  -P wire=responses \
  -P model="$YAM_GPT_MODEL" \
  -P effort=medium \
  -P images=always \
  -P max_llm_calls=60 \
  -P max_speed_frac=0.05 \
  -E auto_start=False \
  -E unattended=False \
  --instruction "${YAM_COMMAND:?NO YAM_COMMAND INPUT}" \
  --max-steps 1200 \
  --max-action-delta 0.05 \
  --epochs 1 \
  --grader operator \
  --scorer operator \
  --store-frames \
  --rerun \
  --rerun-save \
  --fail-on-error 1 \
  --log-dir logs/yam/gpt




# gemini
export GEMINI_API_KEY=
export YAM_GEMINI_MODEL='google/gemini-3.1-pro-preview'
# python _yam_instructions/_eval_yam.py gemini --model "$YAM_GEMINI_MODEL" --probe
# python _yam_instructions/_eval_yam.py gemini --model "$YAM_GEMINI_MODEL" --dry-run
python _yam_instructions/_eval_yam.py gemini --model "$YAM_GEMINI_MODEL"

inspect-robots run \
  --config "$YAM_CONFIG" \
  --embodiment yam_arms \
  --policy agent \
  -P wire=interactions \
  -P model="$YAM_GEMINI_MODEL" \
  -P effort=medium \
  -P images=always \
  -P max_llm_calls=60 \
  -P max_speed_frac=0.05 \
  -E auto_start=False \
  -E unattended=False \
  --instruction "${YAM_COMMAND:?NO YAM_COMMAND INPUT}" \
  --max-steps 1200 \
  --max-action-delta 0.05 \
  --epochs 1 \
  --grader operator \
  --scorer operator \
  --store-frames \
  --rerun \
  --rerun-save \
  --fail-on-error 1 -\
  -log-dir logs/yam/gemini








# grok
export XAI_API_KEY=
export YAM_GROK_MODEL='x-ai/grok-4.7'
# python _yam_instructions/_eval_yam.py grok --model "$YAM_GROK_MODEL" --probe
python _yam_instructions/_eval_yam.py grok --model "$YAM_GROK_MODEL" --dry-run
# python _yam_instructions/_eval_yam.py grok --model "$YAM_GROK_MODEL"

.venv/bin/inspect-robots run \
  --config "$YAM_CONFIG" \
  --embodiment yam_arms \
  --policy agent \
  -P wire=responses \
  -P model="$YAM_GROK_MODEL" \
  -P effort=medium \
  -P images=always \
  -P max_llm_calls=60 \
  -P max_speed_frac=0.05 \
  -E auto_start=False \
  -E unattended=False \
  --instruction "${YAM_COMMAND:?NOT YAM_COMMAND INPUT}" \
  --max-steps 1200 \
  --max-action-delta 0.05 \
  --epochs 1 \
  --grader operator \
  --scorer operator \
  --store-frames \
  --rerun \
  --rerun-save \
  --fail-on-error 1 \
  --log-dir logs/yam/grok




# 默认真实任务是单次、人工确认/评分、保存帧，并强制 auto_start=False、
# unattended=False。脚本不使用 --disable-guardrails；默认每步关节变化上限为
# 0.05 rad，agent 速度比例为 0.05。按实际标定可显式调小，例如：
# python _yam_instructions/_eval_yam.py claude --max-action-delta 0.03 --max-speed-frac 0.03

# 查看结果：
# 每次运行保存到 logs/yam/<agent>/YYYYMMDD-runXXX/；各 agent 每天独立递增。
# 主 JSON、actions/、frames/、transcripts/ 和 wire/ 全部归在同一次运行目录下。
inspect-robots view logs/yam/claude
# inspect-robots view logs/yam/gpt
# inspect-robots inspect logs/yam/claude/20260928-run001/20260928-run001.json --transcript
