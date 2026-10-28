# SO-101：使用 inspect-robots CLI 评测

可用 `python _so101_instructions/_eval_so101.py PROVIDER` 启动 Claude、GPT、Gemini 或 Grok 评测，其中 `PROVIDER` 分别为 `claude`、`gpt`、`gemini`、`grok`。先加 `--dry-run` 检查生成的 `inspect-robots run` 命令。

当前 checkout 为 `jjn`。电机驱动使用官方 [robocurve/inspect-robots-so101](https://github.com/robocurve/inspect-robots-so101)，固定 commit `096d89e12a9e31228fa57a7e5b7d236b09f3db6e`。`so101_configured` 是配置这台机器的 CLI 注册入口：将相机参数转换成官方 `SOArmConfig` 所需的对象，并补充关节名称、起始 home 和退出归位。电机连接、校准检查、限幅、ramp、pacing 和 settling 均复用官方实现。

## 启用当前 CLI

```bash
cd /home/magiclab/inspect-robots
source .venv/bin/activate
uv pip install --python .venv/bin/python --no-deps -e .
uv pip install --python .venv/bin/python --offline --no-deps --no-build-isolation -e ./_so101_instructions
```

当前机器已完成安装；官方 SO-101 插件、LeRobot 和 RealSense SDK 均已安装。CLI 使用当前工作区源码，agent 插件保持已有 editable 安装。

## 用 setup 配置单臂

```bash
export SO101_CONFIG="$PWD/.so101/config.ini"
inspect-robots setup --config "$SO101_CONFIG"
inspect-robots doctor --config "$SO101_CONFIG"
```

首次创建配置时，向导初始建议是 YAM；选择 `policy=agent`、`embodiment=so101_configured`、`scorer=operator`，图像模式选 `always`。已有此配置时这些值会作为默认值，直接回车即可保留。

`so101_configured` 的设备声明只有一个 follower 串口和 Sonix 相机，没有左右臂或 leader 端口。D435 SDK serial、校准 ID/目录和 home 参数由已有 `[embodiment.args]` 保留。setup 配置设备与行为，不执行机械臂校准或移动；保存时生成 `.bak` 备份。

## 用默认值精简 run 命令

每个终端先选择一次这台 SO-101 的配置：

```bash
export INSPECT_ROBOTS_CONFIG="$PWD/.so101/config.ini"
```

配置已保存：

| 位置 | 默认参数 |
| --- | --- |
| `[defaults]` | `policy=agent`、`embodiment=so101_configured`、`scorer/grader=operator`、`max_steps=1200`、存帧、Rerun 和 Rerun 保存 |
| `[embodiment.args]` | follower 串口、两路相机、校准 ID/目录、角度模式、home 和控制频率 |
| `[policy.args]` | `images=always`、`effort=medium`、`max_speed_frac=0.05`、`max_llm_calls=60` |

因此 GPT 可以直接运行：

```bash
inspect-robots run \
  -P wire=responses \
  -P model="$SO101_GPT_MODEL" \
  --instruction "$SO101_COMMAND" \
  --fail-on-error 1 \
  --log-dir logs/so101/gpt
```

Claude/Grok 可用上述 Python 入口切换 provider，也可参照 GPT 示例直接传入 `-P wire=claude-code` 或 `-P wire=responses` 和对应模型。显式 `-E`、`-P` 和对应 CLI 参数优先于配置；例如 `-P max_llm_calls=10` 可以只覆盖当次运行的预算。模型、wire 也能写入 `[policy.args]`，当前保留在命令中方便切换。Claude 的 `claude_command` 只适用于 `wire=claude-code`，因此不放进多种模型共享的配置。

`epochs` 默认 1。eval 使用官方 `so_arm` 的动作语义：`max_relative_target=None`、不额外声明 `max_step`，也不传入 `--max-action-delta=1.0`。CLI 保留范围与步幅检查，agent 按 `max_speed_frac=0.05` 和读取到的校准范围插值；当前 10 Hz 下，五个关节每步预算约为 `[1.08, 1.034, 0.973, 1.013, 1.8]` degrees，夹爪为 0.5 个归一化单位。这是规划序列的步幅，最终目标仍会完整交给驱动，eval 没有基于实测位置的额外 1° 截断。当前 CLI 尚未从配置读取 `fail_on_error`，命令中保留 `--fail-on-error 1`。任务指令和不同 agent 的日志根目录仍在命令中选择。

## 复用 RoboColosseum 的历史校准

RoboColosseum 的 `scripts/teleoperate.sh` 和 `scripts/run_official_so101_checkpoint.py` 使用 `follower_arm`。历史校准实际保存在两个项目共用的 LeRobot 缓存，而不是 RoboColosseum 仓库目录：

```text
/home/magiclab/.cache/huggingface/lerobot/calibration/robots/so_follower/follower_arm.json
```

这份文件已找到，包含六个电机的校准记录。使用同一只 follower、且校准仍匹配电机时，可以跳过 `lerobot-calibrate`：

```bash
export SO101_ROBOT_ID='follower_arm'
export SO101_CALIBRATION_DIR="$HOME/.cache/huggingface/lerobot/calibration/robots/so_follower"
```

配置已保存对应的 `robot_id` 和 `calibration_dir`。CLI 在构造 embodiment 时读取 `<calibration_dir>/<robot_id>.json`，按官方 LeRobot 的单位转换编码器范围，并用于 agent 的 `move_joints` 工具说明和运行时关节范围检查。当前肩部为约 ±103.385°、肘部为约 ±97.319°；夹爪继续使用 0..100。模型还会收到一侧固定、一侧活动的夹爪结构说明。

原始 JSON 由本地驱动读取；提供给模型的是换算后的关节范围和单位说明。校准文件不是相机标定或机械臂运动学模型，不提供“哪个角度变化等于向前移动”的映射。官方插件使用 `connect(calibrate=False)` 并检查校准与电机匹配；若不匹配会报错，不会在 eval 中自动执行新的校准。

`SO101_JOINT_LOW/HIGH` 不必填写，旧环境变量不影响评测。自定义关节范围可通过配置或显式 `-E joint_low/high` 进一步收窄；不能扩大已有校准文件中的范围。前五项单位为 degrees，夹爪为 0..100。

## 两路相机与 home

| 逻辑名 | 设备 | 配置 |
| --- | --- | --- |
| `front` | 已有 Sonix CAM1 | 稳定的 `/dev/v4l/by-id/...-video-index0` 路径 |
| `head` | RealSense D435 | SDK serial `262522073381`，RGB，640×480，30 FPS |

D435 的 SDK serial 与 V4L 路径里的编号不同。两路画面都会提供给 agent，并由 `--store-frames` 和 Rerun 保存。相机已同时捕获验证；doctor 只检查接口声明，不打开相机或机械臂。

校准文件中的 `homing_offset` 是编码器偏移，`range_min/max` 是活动范围；它没有定义 home 姿态。本机采用 `/home/magiclab/RoboColosseum/scripts/run_checkpoint.sh` → `multi_rollout.py` 的行为：连接后读取当前六个关节位置，将它们作为本次连接的 home。RoboColosseum 的 LeRobot `rollout/context.py` 将初始观测里的 `*.pos` 保存为 `initial_position`，之后返回这个姿态。G05 训练状态均值是另一部署流程的选择，不适用于此处。

```ini
[embodiment.args]
# No fixed home_pose: capture the actual starting pose once per connection.
capture_home = True
return_home = True
home_max_relative_target = 1.0
```

启动前将机械臂摆在希望的起始姿态。首次连接后，终端中的 `homing: ramping SO-101 to home position [...]` 会显示实际捕获的 home。顺序为 `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`；前五项为 degrees，夹爪为 0..100。首次启动时目标就是当前姿态，因此通常看不到明显回零运动；后续 trial 和正常退出渐进返回同一 home。重新连接时重新捕获，避免沿用前一次连接的姿态。

直接使用现有 `inspect-robots run` 命令即可。首次连接会提示 `Arm will move to the home pose - stand clear, then press Enter...`；按 Enter 后显示 ramping 状态，通过官方 `_home()` 渐进归位并执行到位等待，然后显示 `Position the scene, then press Enter to start...`。同一连接内的后续 trial 继续归位，但不重复第一道确认提示。

`home_max_relative_target=1.0` 只在归位期间临时应用。`control_hz=10.0` 时，五个臂关节每次回零目标增量不超过 1°，间隔至少 0.1 秒；夹爪增量不超过 1 个归一化单位。归位后恢复 eval 的 `max_relative_target=None`。`settle_tolerance=1.0` 是到位等待阈值，不是动作限幅。回零插值检查关节范围，不进行环境碰撞检测。

评测自然结束或第一次按 Ctrl-C 停止时，先用官方 ramp 渐进返回本次连接记录的 home，再断开连接；归位期间再次按 Ctrl-C 会立即中止回程并断开连接。硬件故障或其他异常时仍跳过退出归位；尚未确认首次 home 运动提示时取消，也不会发送位置命令。可用 `-E return_home=False` 关闭正常结束及 Ctrl-C 后的退出归位。需要跨次运行固定同一 home 时，应先确认六个关节的实测姿态，再显式保存 `home_pose=[...]`；校准偏移和训练均值不能替代这一步。


## 账号与模型

| Agent | 参数 | 认证 |
| --- | --- | --- |
| Claude | `-P wire=claude-code -P model="$SO101_CLAUDE_MODEL"` | Claude Code 账号登录 |
| GPT | `-P wire=responses -P model="$SO101_GPT_MODEL"` | `OPENAI_API_KEY` |
| Grok | `-P wire=responses -P model="$SO101_GROK_MODEL"` | `XAI_API_KEY` |

本机的官方 Claude Code CLI 位于：

```bash
export SO101_CLAUDE_COMMAND="$PWD/.so101/claude-cli/claude_agent_sdk/_bundled/claude"
"$SO101_CLAUDE_COMMAND" auth status
# 尚未登录时：
# "$SO101_CLAUDE_COMMAND" auth login
```

Claude 使用订阅账号，不读取登录 token。GPT 和 Grok 保持 API key 登录方式。可用隐藏输入设置 key：

```bash
read -rsp 'OPENAI_API_KEY: ' OPENAI_API_KEY
export OPENAI_API_KEY
printf '\n'
read -rsp 'XAI_API_KEY: ' XAI_API_KEY
export XAI_API_KEY
printf '\n'
```

## 每次评测的日志父目录

之前 CLI 加载的是 `.venv` 中的 PyPI 0.59.0，它把 JSON 直接写入 `--log-dir`。当前工作区源码已实现日期与编号目录；切换到 editable 安装后，`--log-dir logs/so101/claude` 会自动生成：

```text
logs/so101/claude/20260928-run001/
  20260928-run001.json
  frames/
  actions/
  ... agent 状态、转录和启用时的 .rrd
```

同一天在该日志根目录内继续运行会生成 `run002`、`run003`；编号按各日志根目录分别递增，不需要手工设置 DATE 或 run 编号。过去的日志保留原来的位置。

```bash
inspect-robots view logs/so101/claude
# 查看某次评测：
# inspect-robots inspect logs/so101/claude/20260928-run001/20260928-run001.json --transcript
```


## Rerun 实时连接与到位警告

本次排查发现旧 Viewer 仍停在 09:47 的旧 recording，只有 recording 属性，未显示后续实时数据；磁盘 `.rrd` 中实际保存了两路图像和六个关节状态。新开的 Viewer 使用显式连接后已验证能收到诊断图像和曲线。如仍看到旧空白窗口，可在另一个已激活虚拟环境的终端运行：

```bash
rerun --connect rerun+http://127.0.0.1:9876/proxy
```

在新窗口选择本次 `inspect_robots` recording，时间轴切到最新数据。初始相机图像和关节状态会在 `reset()` 完成后、第一次调用 agent 之前发布；等待 `Position the scene, then press Enter to start...` 时尚未完成 reset。此后画面随评测动作更新，LLM 思考期间不是独立的连续视频流。

`settle timeout budget exhausted` 表示累计 20 次到位等待超时，插件在本 trial 剩余部分停止等待到位，但仍读取实际关节观测。`residual=2.5934 degrees` 是肘关节实际位置与目标之差，超过当前 `settle_tolerance=1.0`，不是动作限幅。应结合目标/实测曲线判断跟踪误差；若持续偏离或误差增长，需要排查实际跟踪情况。增大容差只会改变到位判断，不能修复关节跟踪偏差；本次保留 1° 阈值。
