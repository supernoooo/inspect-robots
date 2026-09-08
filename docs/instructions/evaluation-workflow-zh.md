# Inspect Robots 评测组件、Operator Console 与环境切换手册

本文面向当前仓库和当前 Python 环境，说明如何选择 task、policy、embodiment、
scorer/grader，如何判断一次实验是否成功，以及如何在仿真和真机之间切换。

> 真机安全：Rerun 和 HTML 页面只是观察、审计工具，不是安全系统。运行真机时必须有
> 人员在机械臂旁监视并能够触达物理急停。不要使用 `--disable-guardrails`。首次使用新
> checkpoint、动作接口或机器人配置时，先运行无动作检查，再进行单步、低速验证。

## 1. 一次评测由什么组成

Inspect Robots 将机器人评测拆成几个彼此独立的部分：

| 部分 | 含义 | CLI 选择方式 |
|---|---|---|
| Task | 场景集合、每个场景的 instruction、目标、步数/时间预算和 scorer | `--task NAME` |
| Policy | 根据图像和状态产生动作的“大脑” | `--policy NAME` |
| Embodiment | 产生观察并执行动作的“身体 + 环境” | `--embodiment NAME` 或 `--sim` |
| Grader | 在 rollout 结束后记录成功/失败判定 | `--grader operator` 或 `--grader vlm` |
| Scorer | 从轨迹和 grader 判定计算数值分数 | `--scorer NAME`，仅 ad-hoc/auto task 可用 |
| Sink | 将过程写入 JSON、实时 JSON 或 Rerun | 通常由 CLI 自动配置 |

Policy 和 embodiment 必须在动作维度、控制模式、夹爪语义、相机、状态字段和控制频率上
兼容。Inspect Robots 会在动作执行前检查这些契约。

始终先确认当前终端实际使用的环境和已注册组件：

```bash
which inspect-robots
inspect-robots --version
inspect-robots list
inspect-robots config show
```

不要把 README 中的 `<name>`、`<joint-space-embodiment>` 一类占位符原样复制到 Bash；
尖括号会被 Bash 当成输入/输出重定向。

## 2. 当前环境中可选的组件

以下是 2026-09-07 在本仓库 `.venv` 中 `inspect-robots list` 的结果。插件安装或卸载后，
应以你自己终端中的最新 `inspect-robots list` 为准。

### 2.1 Tasks

| 名称 | 含义 | 适用范围 |
|---|---|---|
| `cubepick-reach` | 4 个默认种子场景；二维红色末端移动到绿色方块；默认最多 80 步；`success_at_end` 评分 | 框架 smoke test、scripted/random/noop、不同 LLM agent 的小规模对比 |

当前核心仓库只内置这一个 task。KitchenBench、WorldEvals 或其他正式 benchmark 需要单独
安装对应任务包；安装后 task 会自动出现在 `inspect-robots list` 中。

Task 的三种启动方式互斥，必须且只能选择一种：

```bash
# 已注册 benchmark；instruction 已写在各个 Scene 中
inspect-robots run --task cubepick-reach --policy scripted --embodiment cubepick

# 临时单场景任务
inspect-robots run --instruction "reach the cube" \
  --policy agent --embodiment cubepick \
  -P model=openai/gpt-5.6-sol -P wire=responses

# 从初始相机画面生成 instruction 和 rubric
inspect-robots run --auto-task \
  -A model=MODEL_NAME --policy agent --embodiment EMBODIMENT_NAME
```

对于正式批量评测，应将 instruction、seed、target 和 rubric 写入注册 Task，而不是每次
手工输入。运行中的 operator feedback 是补充指导，不会替换日志里的原始 instruction。

### 2.2 Embodiments

当前已安装：

| 名称 | 类型 | 动作契约 | 说明 |
|---|---|---|---|
| `cubepick` | 内置二维仿真 | 2-D `eef_delta_pos`，无夹爪 | 不需要 GPU/硬件；红点到绿点即成功；不是实际抓取仿真 |
| `yam_arms` | I2RT YAM 双臂真机 | 默认 14-D `joint_pos`：左右各 6 关节 + 1 夹爪 | 需要 CAN、三路相机和 `inspect-robots-yam`；可搭配 `agent`、`molmoact2`、`gr00t` 等兼容 policy |

README 中记录、但当前环境未必安装的 embodiment：

| 名称 | 类型/机器人 | 需要的包或系统 |
|---|---|---|
| `isaacsim` | Isaac Lab 仿真，默认 Franka 7 关节 + 夹爪 | `inspect-robots-isaacsim` + 可工作的 Isaac Lab/Isaac Sim 环境 |
| `ros` | 任意通过 rosbridge 暴露标准 topic 的 ROS 1/2 机械臂 | `inspect-robots-ros` + `rosbridge_server` |
| `franka` | Franka FR3/Panda 真机 | 外部 `inspect-robots-franka` 包 |
| `a2_arms` | AgiBot A2 Ultra 双臂 | 外部 `inspect-robots-agibot-a2` 包 |
| `g1_arms` | Unitree G1 双臂 | 外部 `inspect-robots-unitree-g1` 包 |
| `so_arm` | SO-100/SO-101 | 外部 `inspect-robots-so101` 包 |
| `widowx` | WidowX 250S | 外部 `inspect-robots-widowx` 包 |

“README 支持”不代表当前终端已经可用；只有 `inspect-robots list embodiments` 输出的名称
才能被当前环境解析。

### 2.3 Policies

| 名称 | 含义 | 推理位置/依赖 | `cubepick` 是否适合 |
|---|---|---|---|
| `scripted` | 确定性直线 oracle | 本进程 | 是 |
| `random` | 随机小位移 baseline | 本进程 | 是 |
| `noop` | 始终输出零动作 | 本进程 | 是 |
| `agent` | LLM 根据 embodiment 契约调用 `move_by`/`move_joints` 等工具 | OpenAI/Anthropic/Gemini/OpenRouter 或兼容 API | 是 |
| `molmoact2` | YAM MolmoAct2 `/act` 客户端 | GPU 机器上的 MolmoAct2 server，默认 8202 | 否；面向 YAM 14-D 动作 |
| `gr00t` | YAM GR00T `/act` 客户端 | GPU 机器上的 GR00T server，默认 8203 | 否；面向 YAM 14-D 动作 |
| `xpolicylab` | XPolicyLab WebSocket 客户端，可连接 π0/π0.5、GR00T、OpenVLA、ACT 等 | 每个 checkpoint 独立的 XPolicyLab server | 通常否；默认是 8-D joint 或 8-D EEF 绝对动作 |
| `capx` | LLM 生成 Python，调用分割、抓取规划和 IK，再输出关节动作 | LLM API + SAM3 + Contact-GraspNet + Pyroki | 否；v1 要求单臂 joint-space + 单个 `gripper` |

`agent` 和 `capx` 是两种不同的 policy 架构。只更换 `-P model=...` 是比较同一架构的不同
LLM backbone，不是把普通 agent 变成 Cap-X。

### 2.4 Graders 和 scorers

| 名称 | 类型 | 含义 |
|---|---|---|
| `operator` grader | 人工判定 | rollout 后询问 `y/n/partial/skip` 并记录备注 |
| `vlm` grader | 自动判定 | 将初始/最终相机帧、instruction 和 rubric 发给视觉模型 |
| `success_at_end` | scorer | 仅在 embodiment 以 `termination_reason="success"` 结束时为 1 |
| `operator` | scorer | 读取 operator/VLM grader 写入的成功判定 |
| `episode_length` | scorer | 执行步数 |
| `min_distance_to_goal` | scorer | 轨迹中距目标的最小距离；需要 embodiment 提供该信息 |
| `reached_goal_state` | scorer | 最小距离是否低于阈值 |

`cubepick` 有 privileged success，因此适合 `success_at_end`。大多数真机不知道任务是否
成功，应使用 `operator` scorer，并由 operator 或 VLM grader 提供判定。

当前用户配置中的 `scorer` 是 `success_at_end`。运行 YAM 临时任务时应显式覆盖：

```bash
inspect-robots run --instruction "place the fork on the plate" \
  --policy agent --embodiment yam_arms \
  -P model=openai/gpt-5.6-sol -P wire=responses \
  --grader operator --scorer operator
```

也可以把真机默认值改成：

```bash
inspect-robots config set scorer operator
```

注册 Task 自带 scorer，不能再用 CLI `--scorer` 覆盖；正式真机 benchmark 应在 Task 中
配置适当的 scorer。

## 3. Operator Console 应该输入什么

### 3.1 普通 Agent 模式

当 policy 支持运行中反馈时，会显示：

```text
operator console: type a message + Enter to send it to the policy; Esc (or /stop [note]) ends the episode; /y /n /p [note] ends it with a verdict
```

这不是要求你必须输入内容。Agent 正常工作时可以什么都不输入。可选操作如下：

| 输入 | 效果 |
|---|---|
| 普通文本 + Enter | 在下一次推理时发送给 Agent，例如 `the fork is behind the bowl` |
| `Esc` | 立即结束 episode，之后通常进入 grader 判定 |
| `/stop` | 与 Esc 类似；`/stop some note` 会把 note 写入日志 |
| `/y task completed` | 结束并立即记录成功，不再重复询问 verdict |
| `/n failed to grasp` | 结束并立即记录失败 |
| `/p reached but not grasped` | 结束并记录 partial；默认按失败计分 |

### 3.2 Cap-X/end-only 模式

Cap-X 当前不接收运行中的 operator 文本作为下一轮模型反馈，因此显示：

```text
operator console: Esc (or /stop [note]) ends the episode; /y /n /p [note] records a verdict; typed notes are saved to the log
```

此时普通文本只保存为审计备注，不会改变 Cap-X 下一轮决策。使用 Esc、`/stop`、`/y`、
`/n`、`/p` 控制 episode 结束和判定。

### 3.3 rollout 结束后的判定

若 embodiment 没有给出明确的 success/failure，operator grader 会询问：

```text
did the robot succeed? [y/n/partial/skip] (partial scores as failure)
grader notes (Enter for none):
```

含义：

| 输入 | 含义 |
|---|---|
| `y` | 成功 |
| `n` | 失败 |
| `partial` | 部分完成，但 scorer 按失败处理 |
| `skip` | 不提供人工判定 |
| grader notes | 可选解释，不影响数值分数 |

在实验开始前应先写好可观察的成功标准。例如“叉子完全位于盘子内部且机械臂已松开叉子”，
而不是运行结束后临时改变标准。对真机，operator 可以直接观察现场，也可以同时查看 Rerun
相机；GUI 很有帮助，但不是判定成功的唯一方式。

如果希望完全自动评分，可以使用 VLM grader：

```bash
inspect-robots run --instruction "place the fork on the plate" \
  --policy agent --embodiment yam_arms \
  -P model=openai/gpt-5.6-sol -P wire=responses \
  --no-prompt --grader vlm \
  -G model=GRADER_MODEL -G rubric_file=task-rubric.md \
  --scorer operator --store-frames
```

## 4. 如何实时观察和事后判断成功

### 4.1 实时观察：Rerun；真机还必须现场监视并配备物理急停

```bash
inspect-robots run --instruction "reach the cube" \
  --policy agent --embodiment cubepick \
  -P model=openai/gpt-5.6-sol -P wire=responses \
  --store-frames --rerun --rerun-save
```

Rerun 显示相机、测量状态、命令动作、reward、termination marker 和 Agent transcript。它
可能因为可视化队列压力丢帧，所以动作 JSONL 才是“机器人收到什么命令”的权威记录。

远程真机没有桌面时，可在操作者电脑启动 Rerun 并建立 SSH 反向隧道，再在机器人端使用
`--rerun-connect`。仅需离线审计时使用 `--rerun-save --no-rerun`。

### 4.2 事后检查

运行结束会打印确切 JSON 路径。不要在存在多个文件时把 `logs/*.json` 直接传给只接收一个
文件的命令。

```bash
inspect-robots inspect logs/ACTUAL_LOG.json
inspect-robots inspect logs/ACTUAL_LOG.json --transcript
inspect-robots inspect logs/ACTUAL_LOG.json --wire
inspect-robots view logs/ --serve --open
inspect-robots video logs/ACTUAL_LOG.json
```

重点检查：

- `run status`：评测程序是否正常完成；它不等同于任务成功率。
- `metrics`：Task scorer 聚合后的指标。
- `termination reason`：`success`、`failure`、`max_steps`、`done`、`give_up`、`operator_end` 等。
- `operator judgement` 和 `judgement source`：判定来自 embodiment、console 还是 VLM。
- transcript/wire：模型看到了什么、输出了什么、API 是否重试或报错。
- action side-car：实际发给 embodiment 的动作。

## 5. YAM + Cap-X 错误解释

以下错误发生在 rollout 和 Cap-X 动作之前：

```text
ValueError: plan 0021 CaP-X v1 profile requires exactly one dim_labels entry named 'gripper'; found []
```

`yam_arms` 是双臂 14-D 动作，标签是：

```text
left_j0 ... left_j5, left_gripper,
right_j0 ... right_j5, right_gripper
```

Cap-X v1 是单臂 profile，要求恰好一个标签的名字严格等于 `gripper`。YAM 既没有这个精确
标签，又有两个夹爪，因此 `yam_arms + capx` 当前不兼容。不要仅把 `left_gripper` 重命名为
`gripper` 来绕过检查；Pyroki 机器人模型、关节维度、另一只手臂的保持动作、状态向量和抓取
坐标系仍然不匹配。

可行路径只有：

1. 使用真正符合 Cap-X v1 契约的单臂 joint-space embodiment；或
2. 编写安全的 YAM 单臂视图 adapter，明确固定另一臂并提供正确相机/深度/坐标变换；或
3. 扩展 Cap-X，使其正式支持双臂和两个 gripper。

错误前的警告：

```text
collision guardrail disabled by config
```

是另一项独立问题。当前仍有 clamp 和 delta-limit，但没有 YAM 的预测碰撞检查。不要把未测量
的默认基座位置当成真实几何，也不要直接开启一个使用占位几何的碰撞模型。应先测量左右基座
位置/yaw 和桌面高度，安装 collision extra，写入配置后再启用：

```bash
uv pip install "inspect-robots-yam[collision]"
inspect-robots setup
inspect-robots-yam-health
inspect-robots-yam-preflight --dry-run
```

需要测量并配置的字段包括 `collision_left_base_pos`、`collision_right_base_pos`、
`collision_left_base_yaw`、`collision_right_base_yaw` 和 `collision_table_height`。仓库文档中
出现的默认基座偏移只是未验证占位值，不能直接用于你的 rig。

preflight 检查配置文件当前选中的 policy/embodiment 组合；它没有 `--policy` 参数。你当前
配置的默认 policy 是 `molmoact2`，因此这条 preflight 不等于检查 `agent + yam_arms`。
无论检查哪种组合，preflight 都只验证维度、语义、相机和状态兼容性，不证明 checkpoint
的关节方向正确，也不替代低速单步验证、现场监视和急停。

## 6. 切换仿真/真机和 policy 时改哪些部分

### 6.1 内置 CubePick 仿真

只需要显式选择 task、policy、embodiment：

```bash
inspect-robots run --task cubepick-reach \
  --policy scripted --embodiment cubepick \
  --seed 0 --store-frames --rerun
```

切换 baseline：

```bash
inspect-robots run --task cubepick-reach --policy random --embodiment cubepick
inspect-robots run --task cubepick-reach --policy noop --embodiment cubepick
```

切换不同 LLM agent，只修改模型路由参数，其他条件保持一致：

```bash
export OPENAI_API_KEY=...
inspect-robots run --task cubepick-reach \
  --policy agent --embodiment cubepick \
  -P model=openai/gpt-5.6-sol -P wire=responses -P effort=medium
```

### 6.2 用 `--sim` 选择配置好的仿真 counterpart

先设置一次：

```bash
inspect-robots config set sim_embodiment cubepick
```

之后：

```bash
inspect-robots run --task cubepick-reach --policy scripted --sim
```

`--sim` 与显式 `--embodiment` 不能同时使用。`--sim` 使用 `[sim_embodiment.args]`，不会把
真机的串口、CAN 或相机 ID 泄漏到仿真配置中。

### 6.3 Isaac Lab 仿真 + XPolicyLab VLA

需要三部分：已注册 benchmark、Isaac Lab embodiment、GPU 上的 VLA server。

```bash
# Inspect Robots/Isaac 环境
uv pip install -e ./plugins/inspect-robots-isaacsim
uv pip install -e ./plugins/inspect-robots-xpolicylab

# GPU 端按 XPolicyLab 文档启动与该任务/机器人匹配的 checkpoint server

# 评测端；my-benchmark 必须是真实已注册名称
inspect-robots run --task my-benchmark \
  --policy xpolicylab --embodiment isaacsim \
  -P url=ws://GPU_HOST:19000 \
  -P action_type=joint \
  -P cameras=cam_head:base_rgb \
  -P name=xpolicylab:MODEL_TAG \
  -E task_id=ISAAC_TASK_ID -E headless=true \
  --store-frames --rerun-save --no-rerun
```

真正加载哪个 VLA 由 GPU server 的 checkpoint 决定；`-P name=` 只给日志加标签，不会
切换模型。换模型时通常修改 server 启动配置/checkpoint/端口，并同步修改 action/camera
mapping 和日志标签。

### 6.4 YAM 真机 + Agent

先进行 idle health/无任务动作检查，并保持人员在急停旁：

```bash
inspect-robots-yam-health
inspect-robots-yam-preflight --dry-run
```

这里的 preflight 检查配置文件当前选中的 policy；当前默认是 `molmoact2`。若要验证其他
policy，仍需确认后续 `inspect-robots run` 打印的 compatibility/guardrail 信息，并从低速、
小动作开始。

确认 scene、相机、home/park 和安全配置后才运行：

```bash
export OPENAI_API_KEY=...
inspect-robots run --instruction "place the fork on the plate" \
  --policy agent --embodiment yam_arms \
  -P model=openai/gpt-5.6-sol -P wire=responses \
  -P effort=medium -P max_speed_frac=0.1 \
  --grader operator --scorer operator \
  --store-frames --rerun --rerun-save
```

切换 LLM 时主要改 `model`、`wire`、对应 API key；不要改变机器人动作配置来“适配”一个
模型名字。

### 6.5 YAM 真机 + MolmoAct2/GR00T VLA

GPU 端先启动与 checkpoint 对应的服务。MolmoAct2 示例：

```bash
# 必须在 MolmoAct2 仓库中运行，不是在 inspect-robots 仓库
python examples/yam/host_server_yam.py --host 0.0.0.0 --port 8202
curl http://127.0.0.1:8202/act
```

评测端：

```bash
inspect-robots run --instruction "place the fork on the plate" \
  --policy molmoact2 --embodiment yam_arms \
  -P server_url=http://GPU_HOST:8202 \
  --grader operator --scorer operator \
  --store-frames --rerun
```

GR00T 使用另一个 server/checkpoint 和默认端口 8203：

```bash
inspect-robots run --instruction "stack the red block on the blue block" \
  --policy gr00t --embodiment yam_arms \
  -P server_url=http://GPU_HOST:8203 \
  --grader operator --scorer operator \
  --store-frames --rerun
```

### 6.6 ROS 真机 + Agent 或 XPolicyLab

安装 ROS adapter，配置 rosbridge URL、关节名、命令 topic、动作边界和相机映射：

```bash
uv pip install -e ./plugins/inspect-robots-ros

inspect-robots run --instruction "reach forward slowly" \
  --policy agent --embodiment ros \
  -P model=openai/gpt-5.6-sol -P wire=responses \
  -E url=ws://ROBOT_HOST:9090 \
  -E joints=joint1,joint2,joint3,joint4,joint5,joint6 \
  -E command_topic=/joint_trajectory_controller/joint_trajectory \
  -E action_low=-3.1,-2.2,-2.9,-3.1,-2.9,-3.1 \
  -E action_high=3.1,2.2,2.9,3.1,2.9,3.1 \
  --grader operator --scorer operator
```

切换成 XPolicyLab VLA 时保留 `-E` 机器人参数，修改 `--policy` 和 `-P` server/mapping 参数；
前提是 VLA 的动作契约与这个 ROS embodiment 一致。

### 6.7 Cap-X

Cap-X v1 需要一个已注册的、兼容的单臂 joint-space embodiment。先在独立 CaP-X/GPU 环境
启动三个服务：

```bash
uv run capx/serving/launch_sam3_server.py --port 8114
uv run capx/serving/launch_contact_graspnet_server.py --port 8115
uv run python -c 'from capx.serving.launch_pyroki_server import main; main(robot="panda_description", port=8116)'
```

然后在评测端使用真实注册名称；下面的 `my_joint_arm` 只是示例名称，执行前必须替换：

```bash
export OPENAI_API_KEY=...
inspect-robots run --instruction "pick up the red cube" \
  --policy capx --embodiment my_joint_arm \
  -P model=openai/gpt-5.6-sol -P wire=responses \
  -P sam3_url=http://GPU_HOST:8114 \
  -P graspnet_url=http://GPU_HOST:8115 \
  -P pyroki_url=http://GPU_HOST:8116 \
  --grader operator --scorer operator \
  --store-frames --rerun
```

兼容 embodiment 至少需要：一维有限边界的 `joint_pos` action box、恰好一个 `gripper`
标签、同维完整关节状态、正 control rate、RGB 相机。`plan_grasp()` 还需要 observation extra
中的 depth、intrinsics、camera-to-base extrinsics。当前 `cubepick` 和双臂 `yam_arms` 均不
满足 Cap-X v1 profile。

Cap-X 会在评测进程内执行模型生成的 Python，它不是安全沙箱。不可信模型必须放在容器或
其他外部隔离边界中运行。

## 7. 常用可选参数

### 7.1 选择组件及其参数

| 参数 | 含义 |
|---|---|
| `--task NAME` | 选择已注册 Task |
| `--instruction TEXT` | 构造单个 ad-hoc Scene |
| `--auto-task` | 从初始相机帧自动生成 task/rubric |
| `-T key=value` | Task factory 参数，例如 `-T num_scenes=20` |
| `-A key=value` | auto-task 生成器参数 |
| `--policy NAME` | 选择 policy |
| `-P key=value` | policy 参数，例如模型、wire、server URL |
| `--embodiment NAME` | 选择真机/仿真 embodiment |
| `-E key=value` | embodiment 参数，例如控制模式、设备、task ID |
| `--grader NAME` | `operator`、`vlm` 或插件 grader |
| `-G key=value` | grader 参数，例如 model、rubric、API URL |
| `--scorer NAME` | ad-hoc/auto task 的 scorer；与 `--task` 一起使用会报错 |

### 7.2 复现、批量和错误策略

| 参数 | 含义 |
|---|---|
| `--seed N` | 固定评测 seed |
| `--epochs N` | 每个 Scene 重复 N 次 |
| `--max-steps N` | ad-hoc/auto task 的步数上限；注册 Task 用自己的 horizon |
| `--fail-on-error X` | PolicyError 达到指定次数/比例后停止 |
| `--log-dir DIR` | 日志目录，默认 `logs` |
| `--no-live-log` | 不写供 live HTML 使用的临时快照 |
| `--config PATH` | 为某台 rig 选择独立配置文件 |

### 7.3 图像、可视化和交互

| 参数 | 含义 |
|---|---|
| `--store-frames` | 将相机帧写到 `logs/frames/...`，供 HTML、视频、VLM grader 使用 |
| `--rerun` | 启动本地 Rerun Viewer |
| `--rerun-save` | 保存 `.rrd`；可以不启动 GUI |
| `--no-rerun` | 禁止本地 GUI |
| `--rerun-connect [URL]` | 连接已运行的远程/本地 Rerun Viewer |
| `--rerun-port N` | 本地 Rerun Viewer 端口 |
| `--voice` / `-V` | 语音 operator feedback，需要 voice 插件 |
| `--speak` / `-S` | 朗读 policy note/终端摘要，需要 voice 插件 |
| `--no-prompt` | 禁用 operator grader 和交互 console；不能与显式 `--grader operator` 合用 |

### 7.4 安全相关

| 参数 | 含义 |
|---|---|
| `--max-action-delta D` | 收紧每控制步允许的最大动作变化 |
| `--disable-guardrails` | 关闭默认 clamp/delta-limit；真机不要使用 |

默认 CLI guardrails 只约束动作范围和每步变化；具体机器人还可以贡献碰撞、温度、工作空间
等 guardrail。任何 guardrail 都不能替代正确标定、现场监视和物理急停。

## 8. 推荐的实际推进顺序

1. 用 `cubepick + scripted/random/noop` 验证框架和日志。
2. 用相同 `cubepick-reach`、seed、epochs 比较不同 `agent` LLM backbone。
3. 选择与真实 VLA checkpoint 匹配的机器人/仿真和 benchmark，而不是强行把它接到 CubePick。
4. 在 GPU 端启动 checkpoint server，在评测端只安装轻量 policy adapter。
5. 对 YAM 先完成 health、preflight、相机/关节方向和低速动作验证，再跑完整 task。
6. 只有在具备单臂 joint-space + RGB-D/标定信息的 embodiment 后再尝试 Cap-X。
