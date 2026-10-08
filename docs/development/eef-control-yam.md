source .venv/bin/activate


## setup
```python
inspect-robots setup
```
or
```python
export YAM_CONFIG="$PWD/.yam/config.ini"
```


## task
```python
export YAM_COMMAND="Pick up the blue eclipse box from the table and place it back near its original place on the table."
export YAM_COMMAND="Pick up the blue eclipse box from the table and put it into the red bowl on the table."
```



## agent

claude
```python
export ANTHROPIC_API_KEY
export YAM_CLAUDE_MODEL='anthropic/claude-opus-5-5'
```

gpt
```python
export OPENAI_API_KEY
export YAM_GPT_MODEL='openai/gpt-6-astra'
```

grok
```python
export XAI_API_KEY
export YAM_GROK_MODEL='x-ai/grok-4.7'
```


## command

claude
```command
inspect-robots run \
  --config "$YAM_CONFIG" \
  --embodiment yam_eef \
  --policy agent \
  -P wire=messages \
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
```

gpt
```command
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
```

grok
```command
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
```

