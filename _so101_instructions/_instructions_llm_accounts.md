# SO-101：Claude 账号、GPT、Gemini 与 Grok 评测

当前 checkout 是 `main`。`so_arm` 来自已安装的 `inspect-robots-so101` 插件，其原生配置就是单个 SO-101 follower。`_run_so101.py` 是本次添加的本地入口，负责 Sonix 摄像头、单臂 setup 元数据及关节名称。`agent` 的 `wire=claude-code` 也是本次加入的本地功能，原先仅支持 API 后端；不需要切换分支。

本工作流明确使用两种认证方式：**评测 Claude 时使用你自己的 Claude Code 账号登录；评测 GPT 时继续使用 `OPENAI_API_KEY`。** 两条路径共用现有 `agent`、机械臂配置、动作处理、operator 评分和评测日志。

## 1. 启用本地修改

```bash
cd /home/magiclab/inspect-robots
source .venv/bin/activate
uv pip install --python .venv/bin/python --no-sources \
  'inspect-robots==0.59.0' -e ./plugins/inspect-robots-agent
```

你当前已按 editable 方式安装 agent，因此代码修改会直接生效。无需重新安装 SO-101 或更换 Torch。

当前配置文件 `.so101/config.ini` 的 `camera_device` 已改为：

```text
/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0
```

入口只使用一个 follower 和一个逻辑名为 `front` 的摄像头。校准所用的串口、`robot_id` 和校准目录须与评测一致；完整校准步骤见 `_instructions_so101.sh`，其中串口已与当前 setup 保存的值对齐。

## 2. 登录 Claude Code

本机的 PATH 中原先没有 `claude`，已将官方 Python SDK 的平台 wheel 安装到项目私有目录，仅使用其中附带的原版 CLI。它不改变 `.venv` 中现有依赖，也不需要 node/npm。当前验证版本为 Claude Code 2.1.283。

如需重建这份本地 CLI：

```bash
uv pip install --python .venv/bin/python --target .so101/claude-cli --no-deps \
  'claude-agent-sdk==0.2.160'
```

设置 CLI 路径并检查已登录的账号。你已经登录时只需运行 `auth status`；只有未登录时才运行 `auth login`：

```bash
export SO101_CLAUDE_COMMAND="$PWD/.so101/claude-cli/claude_agent_sdk/_bundled/claude"
"$SO101_CLAUDE_COMMAND" auth status
# 未登录时执行："$SO101_CLAUDE_COMMAND" auth login
```

选择 Claude 订阅账号，不添加 `--console`。预期状态包含 `loggedIn: true`，认证方式为 `claude.ai`。如果你已有官方 CLI，可将变量设置为该程序的绝对路径。

后端会在模型调用的子进程中排除 API key、Console/profile 和云提供商选择变量；不会修改当前终端的环境，`OPENAI_API_KEY` 可以一直保留。登录、令牌刷新和认证文件均交由官方 CLI 管理。不要把登录 token 填入 `ANTHROPIC_API_KEY`。官方 CLI 的 bare 模式不使用订阅登录，因此本后端不启用它。

## 3. 先运行无硬件 Claude 模型探针

```bash
python _so101_instructions/_so101_llm_probe.py --wire claude-code --model claude-opus-5-5 \
  --claude-command "$SO101_CLAUDE_COMMAND"
```

探针使用合成的 `front` 图像（灰色背景、蓝色方块）和六维关节状态，请模型观察后调用 `give_up`。它不会打开摄像头或串口，也不会执行动作。输出应包含停止原因和 token 使用量。这里固定使用 Opus 5.5 的完整 ID `claude-opus-5-5`。`opus`、`sonnet` 等别名可能随官方默认模型更新而改变。

这个探针会真实调用模型，使用你的 Claude 账号额度；通过后再进行真机评测。`doctor` 仅检查机械臂接口声明，不会验证模型登录或硬件通信。

## 4. Claude 账号：首次小动作评测

先按 `_instructions_so101.sh` 导出已经校准的 `SO101_ROBOT_ID`、`SO101_CALIBRATION_DIR`、`SO101_JOINT_LOW` 和 `SO101_JOINT_HIGH`。下面命令会实际连接机械臂、读取 Sonix 画面并执行指令：

```bash
python _so101_instructions/_run_so101.py run \
  --config "$PWD/.so101/config.ini" \
  --embodiment so101_configured \
  --policy agent \
  -P wire=claude-code \
  -P model=claude-opus-5-5 \
  -P claude_command="$SO101_CLAUDE_COMMAND" \
  -P images=always \
  -P max_speed_frac=0.05 \
  -P max_llm_calls=5 \
  --instruction 'Move shoulder_pan approximately 2 degrees from its measured position, keep the other joints and gripper unchanged, then finish.' \
  --max-action-delta 1.0 --max-steps 30 --epochs 1 \
  --grader operator --scorer operator \
  --store-frames --no-rerun --no-rerun-save \
  --log-dir logs/so101/claude
```

在交互终端按插件提示确认场景后开始，结束时填写 operator 评分。确认首轮结果后，将 `--instruction` 改为实际任务。

## 5. GPT API：同一评测流程

GPT 保持原有 API 路径。读取 key 的命令不会将它明文打印到终端：

```bash
read -rsp 'OPENAI_API_KEY: ' OPENAI_API_KEY
export OPENAI_API_KEY
printf '\n'
export SO101_GPT_MODEL='openai/gpt-6-astra'
```

GPT-6 Astra 的 API ID 是 `gpt-6-astra`；项目中加上 `openai/` 前缀，配合 `OPENAI_API_KEY` 直连 OpenAI。先运行同一个无硬件探针：

```bash
python _so101_instructions/_so101_llm_probe.py --wire responses --model "$SO101_GPT_MODEL"
```

探针通过并完成机械臂校准、限位设置后，使用以下完整评测命令：

```bash
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
```

GPT 命令不传 `claude_command`，不依赖 Claude 登录。两种模型分别存储日志，便于用相同任务和评分比较结果。

## 6. 在命令中选择精确模型

当前安装的 agent 0.26.0 已支持 Gemini 与 Grok；Gemini 有 `chat`、`interactions` 与 `gemini-live` 后端，Grok 可用 `chat` 或 `responses`。这里给 Gemini 3.8 Flash 选 `interactions`，给 Grok 4.7 选 `responses`；这些模型均支持图像输入和函数调用。无需再添加 provider 后端。

| 评测模型 | 直接传给 agent 的参数 | 认证 |
| --- | --- | --- |
| Claude Opus 5.5 | `-P wire=claude-code -P model=claude-opus-5-5` | 已登录的 Claude Code 订阅账号 |
| GPT-6 Astra | `-P wire=responses -P model=openai/gpt-6-astra` | `OPENAI_API_KEY` |
| Gemini 3.8 Flash | `-P wire=interactions -P model=google/gemini-3.8-flash` | `GEMINI_API_KEY` |
| Grok 4.7 | `-P wire=responses -P model=x-ai/grok-4.7` | `XAI_API_KEY` |

`opus-5.5` 是展示名称的写法，完整模型 ID 应写成 `claude-opus-5-5`。Opus 5.5 需要 Claude Code 2.1.280 或更新；本项目附带的 2.1.283 已满足。上述 ID 固定模型系列；账号仍须有访问权限。若服务商另行提供日期快照，可用其官方公布的快照 ID 进一步固定版本，不自行拼接日期。

官方依据：[Claude Code 模型配置](https://code.claude.com/docs/en/model-config)、[GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra)、[Gemini 3.8 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash)、[Gemini Interactions](https://ai.google.dev/gemini-api/docs/interactions-overview)、[Grok 4.7](https://docs.x.ai/developers/models/grok-4.7)、[xAI Responses API](https://docs.x.ai/developers/model-capabilities/text/comparison)。Grok 的前缀是 `x-ai/`；`groq/` 是另一家公司。

## 7. 使用已登录 Claude 账号完成一次瓶子任务

当前项目目录下未找到 `.so101/calibration/my_so101.json`，但本机已有 `~/.cache/huggingface/lerobot/calibration/robots/so_follower/follower_arm.json`。如果这份文件属于当前这只 follower，先复用它：

```bash
export SO101_ROBOT_ID=follower_arm
export SO101_CALIBRATION_DIR="$HOME/.cache/huggingface/lerobot/calibration/robots/so_follower"
```

文件名对应 `robot_id`；不能把 leader 的校准文件用于 follower。驱动仍会检查 JSON 中的校准是否匹配当前电机内的记录。仅改文件名、创建空 JSON 或跳过检查不会建立有效校准。校准有效且机构/电机配置未改变时，不需要每次 eval 重新校准；[LeRobot 官方说明](https://huggingface.co/docs/lerobot/il_robots)要求同一套硬件在评测时使用相同的 `id`。

若该文件属于别的机械臂，或复用后仍提示电机校准不匹配，才按下面步骤重新校准。完成后评测沿用同一组 ID 和目录：

```bash
cd /home/magiclab/inspect-robots
source .venv/bin/activate
export SO101_PORT='/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E114717-if00'
export SO101_ROBOT_ID=my_so101
export SO101_CALIBRATION_DIR="$PWD/.so101/calibration"
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port="$SO101_PORT" \
  --robot.id="$SO101_ROBOT_ID" \
  --robot.calibration_dir="$SO101_CALIBRATION_DIR"
```

校准命令会连接机械臂；按 LeRobot 提示进行。确认上述端口对应你的 follower。随后设置实际工作范围：`SO101_JOINT_LOW` 与 `SO101_JOINT_HIGH` 各为六个数的 JSON 数组，顺序是 `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`。前五项单位为度，夹爪范围在 `0..100` 内。校准 JSON 记录电机校准值，不会自动导出这两组工作限位；请选择你已验证过的范围。可在终端输入已有范围：

```bash
read -rp 'SO101_JOINT_LOW (6-number JSON array): ' SO101_JOINT_LOW
read -rp 'SO101_JOINT_HIGH (6-number JSON array): ' SO101_JOINT_HIGH
export SO101_JOINT_LOW SO101_JOINT_HIGH
```

然后按顺序执行：

```bash
export SO101_CLAUDE_COMMAND="$PWD/.so101/claude-cli/claude_agent_sdk/_bundled/claude"
"$SO101_CLAUDE_COMMAND" auth status

# 仅打印真实评测命令：不调用模型，不连接机械臂。
python _so101_instructions/_eval_so101.py claude --model claude-opus-5-5 --dry-run

# 真实调用 Opus 5.5，但只使用合成图像，不连接硬件。
python _so101_instructions/_eval_so101.py claude --model claude-opus-5-5 --probe

# 单次真实瓶子评测：一个 follower、Sonix front 摄像头、operator 评分。
python _so101_instructions/_eval_so101.py claude --model claude-opus-5-5
```

默认任务是：拿起桌上竖立的瓶子，稍微提离桌面，放回最初位置并保持竖立；桌面支撑瓶子后松开夹爪并结束。运行前让摄像头能看到瓶子、夹爪和放置位置，并记下瓶子起始位置，以便人工评估“放回原位”。

启动器固定 `--epochs 1`、`images=always`、`max_speed_frac=0.05`、`--max-action-delta 1.0`、`--grader operator`、`--scorer operator` 和保存画面。它使用 60 次模型决策、1200 个控制步，替代小动作示例中的 5 次/30 步预算。控制频率为 10 Hz，但模型等待会增加总耗时；预算不保证任务完成。

在交互终端按现有提示确认场景后开始。结束时根据实际拿起和放回结果填写 `y / n / partial / skip` 及备注。Claude 账号额度用于模型调用；当前终端的 `OPENAI_API_KEY` 保留，GPT 评测继续使用 API。

可以修改任务和预算：

```bash
python _so101_instructions/_eval_so101.py claude --model claude-opus-5-5 \
  --instruction 'Pick up the bottle on the table and place it back at its original position.' \
  --max-llm-calls 80 --max-steps 1600
```

默认日志目录为 `logs/so101/claude/claude-opus-5-5/bottle`，最终日志文件名由 `run` 打印。查看时使用该文件名：

```bash
inspect-robots view logs/so101/claude/claude-opus-5-5/bottle
# 将路径替换为 run 打印的真实日志文件：
inspect-robots inspect /absolute/path/to/run.json --transcript
```

模型 ID 记录在 policy 配置中；Claude 原始 wire capture 的结果对象还包含 `modelUsage`，可核对实际使用的模型，包括 CLI 是否发生 fallback。模型探针通过只说明登录、图像和工具输出链路可用；真实抓取能力由上述实际评测确定。

## 8. GPT-6 Astra：相同瓶子任务，使用 API key

已导出 key 时无需重复输入。需要设置时：

```bash
read -rsp 'OPENAI_API_KEY: ' OPENAI_API_KEY
export OPENAI_API_KEY
printf '\n'
python _so101_instructions/_eval_so101.py gpt --model openai/gpt-6-astra --probe
python _so101_instructions/_eval_so101.py gpt --model openai/gpt-6-astra
```

机械臂校准与限位沿用第 7 节，任务、预算和评分一致。日志位于 `logs/so101/gpt/openai_gpt-6-astra/bottle`。不会调用 Claude Code。

## 9. Gemini 与 Grok：以后评测的独立脚本

已添加 `_eval_so101_gemini.sh` 和 `_eval_so101_grok.sh`；二者调用统一入口 `_eval_so101.py`，沿用相同的单臂配置、瓶子任务、评分与日志流程。`--model` 可以覆盖默认模型；脚本接受完整的 provider/model ID，也接受原生模型 ID 并自动补上 provider 前缀。

Gemini：

```bash
read -rsp 'GEMINI_API_KEY: ' GEMINI_API_KEY
export GEMINI_API_KEY
printf '\n'
bash _so101_instructions/_eval_so101_gemini.sh --model google/gemini-3.8-flash --dry-run
bash _so101_instructions/_eval_so101_gemini.sh --model google/gemini-3.8-flash --probe
bash _so101_instructions/_eval_so101_gemini.sh --model google/gemini-3.8-flash
```

Grok：

```bash
read -rsp 'XAI_API_KEY: ' XAI_API_KEY
export XAI_API_KEY
printf '\n'
bash _so101_instructions/_eval_so101_grok.sh --model x-ai/grok-4.7 --dry-run
bash _so101_instructions/_eval_so101_grok.sh --model x-ai/grok-4.7 --probe
bash _so101_instructions/_eval_so101_grok.sh --model x-ai/grok-4.7
```

默认日志分别位于 `logs/so101/gemini/google_gemini-3.8-flash/bottle` 与 `logs/so101/grok/x-ai_grok-4.7/bottle`。`--dry-run` 不需要 key 或校准；`--probe` 使用真实 API 额度但不打开硬件；真实运行要求校准文件与关节工作限位已设置。

## 10. `is not calibrated` 是否可以忽略？

不可以。`EmbodimentFault` 是本次运行的致命错误：插件连接电机后、执行任务动作前调用 `_check_calibrated`，校准缺失或不匹配会停止评测。校准定义电机读数到关节位置及夹爪范围的换算，Claude/GPT 等 LLM 评测和 ACT 等训练模型评测都需要。

你提供的错误查找 `my_so101.json`，本机现有文件却叫 `follower_arm.json`。如果属于同一只 follower，使用第 7 节的两个 export 即可让现有 `_run_so101.py` 读取正确文件；也可以在直接 `run` 命令中加 `-E robot_id=follower_arm -E calibration_dir=/home/magiclab/.cache/huggingface/lerobot/calibration/robots/so_follower` 明确覆盖。

`operator console unavailable` 是另一个非致命提示：当前 SO-101 插件仍自己处理 episode 结束按键，运行中打字反馈不可用；结束后的 operator 评分流程仍可使用。它不是此次校准失败的原因。

若只想先测试账号或模型，不进行真机动作，用 `--probe`，无需机械臂校准。需要使用 Opus 5.5 时，应明确传 `--model claude-opus-5-5`；`anthropic/claude-opus-5.1` 选择的是另一模型。

## 后端行为与验证范围

CLI 从标准输入接收真实图像数据和标记了角色的历史；每次决策返回一个已有机器人工具的结构化调用建议。`agent` 的工具集将建议转成 ActionChunk，现有 controller 和 approver 处理动作，后续观测与执行结果回到历史。operator 反馈、评分、transcript、hindsight、usage 和 wire capture 都继续使用原来的流程。

CLI 内建工具、外部 MCP、hooks、skills 和会话落盘被禁用；模型只提出动作建议。每次决策启动一个新 CLI 会话并发送框架保留的历史，因此有进程启动开销。`max_llm_calls` 统计框架的决策次数，CLI 内部每个决策最多三轮；`-P claude_timeout_s=120` 限制单次决策耗时。

本地已验证模拟单臂评测、图像和状态传递、结构化输出解析、失败与超时处理，以及 GPT API key 路径。本次还验证了四个启动器的 CLI 参数、校准/限位错误时阻止启动、模型探针不依赖校准，以及 Responses/Interactions 的 58 项现有测试；脚本通过 Ruff 与 shell 语法检查。我未使用你的账号进行真实模型调用，也未通过这些命令连接机械臂。你已登录时无需重复登录；实际模型与硬件验证按第 7 节执行。

官方参考：[Claude Code CLI](https://code.claude.com/docs/en/cli-reference)、[程序化调用](https://code.claude.com/docs/en/headless)、[认证](https://code.claude.com/docs/en/authentication)。截至本次核对，官方帮助中心的更新说明 SDK、`claude -p` 仍使用订阅额度，此前宣布的独立月度 credit 调整已暂停；详见[官方账号额度说明](https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan)。GPT API 使用仍按你的 API 账号计费。
