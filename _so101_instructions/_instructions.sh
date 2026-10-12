cd /home/magiclab/inspect-robots
source .venv/bin/activate



inspect-robots list embodiments
inspect-robots list policies


export SO101_COMMAND="Pick up the bottle standing on the table, lift it slightly clear of the table, and place it back at its original position."
export SO101_COMMAND="Pick up the red block on the table, lift it slightly clear of the table, and place it back at its original position."
export SO101_COMMAND="Pick up the red block from the table"

export SO101_CONFIG="$PWD/.so101/config.ini"
export INSPECT_ROBOTS_CONFIG="$SO101_CONFIG"
export SO101_PORT='/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E114717-if00'
export SO101_CAMERA='/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0'
export SO101_CAMERA_NAME='front'
export SO101_HEAD_CAMERA_NAME='head'
export SO101_HEAD_CAMERA_SERIAL='262522073381'


# so101_configured use official SO-101 driver, ramp, settling
export SO101_ROBOT_ID='follower_arm'
export SO101_CALIBRATION_DIR="$HOME/.cache/huggingface/lerobot/calibration/robots/so_follower"




inspect-robots setup --config "$SO101_CONFIG"
inspect-robots doctor --config "$SO101_CONFIG"



export SO101_CLAUDE_COMMAND="$PWD/.so101/claude-cli/claude_agent_sdk/_bundled/claude"
"$SO101_CLAUDE_COMMAND" auth login
"$SO101_CLAUDE_COMMAND" auth status
export SO101_CLAUDE_MODEL='claude-opus-5-5'

if [ -f "$PWD/.so101/llm-keys.env" ]; then
  source "$PWD/.so101/llm-keys.env"
fi
export SO101_GPT_MODEL='openai/gpt-6-astra'
export SO101_GROK_MODEL='x-ai/grok-4.7'



inspect-robots colosseum \
  --client-config .so101/colosseum-client.yaml \
  --runtime-config .so101/colosseum-runtime.yaml \
  --record-only \
  --instruction "$SO101_COMMAND" \
  --scene "magiclab" \
  --evaluator "nusmagic" \
  -- \
  --config .so101/config.ini \
  --embodiment so101_eef_configured

  # -P max_speed_frac=0.05 \
  # -P max_llm_calls=60


# review interface
inspect-robots colosseum \
  --client-config .so101/colosseum-client.yaml \
  --review --review-open


# abort eval
inspect-robots colosseum \
  --client-config .so101/colosseum-client.yaml \
  --abort ev_916d792c7af5c1a719e456f98d91d38f \
  --reason "A trial interrupted before recording"




### agents


# cc
inspect-robots run \
  -P wire=claude-code \
  -P claude_command="${SO101_CLAUDE_COMMAND:?NO CLAUDE CLI SELECTED}" \
  -P model="${SO101_CLAUDE_MODEL:?NO MODEL SELECTED}" \
  --instruction "${SO101_COMMAND:?NO INSTRUCTION INPUT}" \
  --rerun \
  --rerun-save \
  --fail-on-error 1 \
  --log-dir logs/so101/claude




# gpt
inspect-robots run \
  -P wire=responses \
  -P model="${SO101_GPT_MODEL:?NO MODEL SELECTED}" \
  --instruction "${SO101_COMMAND:?NO INSTRUCTION INPUT}" \
  --rerun \
  --rerun-save \
  --fail-on-error 1 \
  --log-dir logs/so101/gpt




# grok
inspect-robots run \
  -P wire=responses \
  -P model="${SO101_GROK_MODEL:?NO MODEL SELECTED}" \
  --instruction "${SO101_COMMAND:?NO INSTRUCTION INPUT}" \
  --rerun \
  --rerun-save \
  --fail-on-error 1 \
  --log-dir logs/so101/grok


# View results for the selected agent.
inspect-robots view logs/so101/claude
# inspect-robots view logs/so101/gpt
# inspect-robots view logs/so101/grok
