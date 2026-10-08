cd /home/rc1/inspect-robots
source .venv/bin/activate


inspect-robots list embodiments
inspect-robots list policies


# check yam config
export YAM_CONFIG="$PWD/.yam/config.ini"
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
export YAM_COMMAND="Pick up the blue eclipse box from the table and place it back near its original place on the table."
export YAM_COMMAND="Pick up the blue eclipse box from the table and put it into the red bowl on the table."
export YAM_CLAUDE_MODEL='claude-opus-5-5'

python _yam_instructions/_eval_yam.py claude \
  --claude-command "$YAM_COMMAND" \
  --model "$YAM_CLAUDE_MODEL"

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


inspect-robots run \
  --config "$YAM_CONFIG" \
  --embodiment yam_eef \
  --policy agent \
  -P wire=claude-code \
  -P model="${YAM_CLAUDE_MODEL:?NO MODEL SELECTED}" \
  -P effort=medium \
  -P images=always \
  -P max_llm_calls=60 \
  -P max_speed_frac=0.05 \
  -E auto_start=false \
  -E unattended=false \
  -E eef_orientation=true \
  --instruction "${YAM_COMMAND:?SET YAM_COMMAND}" \
  --max-action-delta 0.05 \
  --max-steps 1200 \
  --grader operator \
  --scorer operator \
  --store-frames \
  --fail-on-error 1 \
  --log-dir logs/yam/eef-claude



# gpt
export OPENAI_API_KEY=
export YAM_GPT_MODEL='openai/gpt-6-astra'
python _yam_instructions/_eval_yam.py gpt --model "$YAM_GPT_MODEL" --dry-run


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


inspect-robots run \
  --config "$YAM_CONFIG" \
  --embodiment yam_eef \
  --policy agent \
  -P wire=responses \
  -P model="$YAM_GPT_MODEL" \
  -P effort=medium \
  -P images=always \
  -P max_llm_calls=60 \
  -P max_speed_frac=0.05 \
  -E auto_start=false \
  -E unattended=false \
  -E eef_orientation=true \
  --instruction "${YAM_COMMAND:?SET YAM_COMMAND}" \
  --max-action-delta 0.05 --max-steps 1200 \
  --grader operator \
  --scorer operator \
  --store-frames \
  --fail-on-error 1 \
  --log-dir logs/yam/eef-gpt



# gemini
export GEMINI_API_KEY=
export YAM_GEMINI_MODEL='google/gemini-3.1-pro-preview'
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
python _yam_instructions/_eval_yam.py grok --model "$YAM_GROK_MODEL" --dry-run

# .venv/bin/inspect-robots run \
#   --config "$YAM_CONFIG" \
#   --embodiment yam_arms \
#   --policy agent \
#   -P wire=responses \
#   -P model="$YAM_GROK_MODEL" \
#   -P effort=medium \
#   -P images=always \
#   -P max_llm_calls=60 \
#   -P max_speed_frac=0.05 \
#   -E auto_start=False \
#   -E unattended=False \
#   --instruction "${YAM_COMMAND:?NOT YAM_COMMAND INPUT}" \
#   --max-steps 1200 \
#   --max-action-delta 0.05 \
#   --epochs 1 \
#   --grader operator \
#   --scorer operator \
#   --store-frames \
#   --rerun \
#   --rerun-save \
#   --fail-on-error 1 \
#   --log-dir logs/yam/grok



inspect-robots run \
  --config "$YAM_CONFIG" \
  --embodiment yam_eef \
  --policy agent \
  -P model="$YAM_GROK_MODEL" \
  -P wire=responses \
  -P effort=medium \
  -P images=always \
  -P max_llm_calls=60 \
  -P max_speed_frac=0.05 \
  -E auto_start=false \
  -E unattended=false \
  -E eef_orientation=true \
  --instruction "${YAM_COMMAND:?NOT YAM_COMMAND INPUT}" \
  --max-action-delta 0.05 \
  --max-steps 1200 \
  --grader operator \
  --scorer operator \
  --store-frames \
  --fail-on-error 1 \
  --log-dir logs/yam/eef-grok




inspect-robots view logs/yam/claude
# inspect-robots view logs/yam/gpt
# inspect-robots inspect logs/yam/claude/20260928-run001/20260928-run001.json --transcript
