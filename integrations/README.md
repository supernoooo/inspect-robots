# Colosseum bridge

SO-101 的 6 维、10 Hz、双相机配置与命令见 [SO-101 评测流水线](../_so101_instructions/COLOSSEUM.md)。

这里保存了从本机 `colosseum-client` 和 `colosseum-policy-server` 复制的源码、
配置示例、文档和测试。各项目的 `VENDORED.json` 记录来源 commit。原项目、虚拟环境、
Git 历史、真实 token、local 配置和评测录制没有复制。

桥接命令是 **`inspect-robots colosseum`**。当前项目脚本默认用 `--local-pair`
在本地选择 A/B LLM，运行原生 Inspect agent，录制实测状态、执行动作和三路相机，
评分后导出 LeRobot v3。先保存、回看，再用现有网页评测和数据集导入接口上传。
没有 Router assignment、模型分配或 VLA 推理；原来的 Router 分配模式仍可显式使用。

本项目仅评测 `grok-4.7`、`gpt-6-astra`、`claude-opus-5-5` 三个 LLM。
runtime 不接受 VLA 或其他模型；使用 Router 分配模式且收到 VLA 时，会在 agent 和机械臂
初始化前拒绝，不发送 ready/start/step，也不提交评分或数据。
`router-models.json` 保存收到的公共模型描述，便于定位服务端分配问题。
LLM key/provider 声明不会排除 Router 模型池中的 VLA；线上 `/api/eval/next`
目前没有 model_type、policyA/policyB 或 model-a/model-b 筛选字段。
Router 分配模式的固定配对需要服务端支持相应筛选。`--local-pair` 则使用本地
选择和现有手动评测/数据集导入接口，完全绕过这个分配流程。
桥接不会把 VLA descriptor 改成 LLM，也不会更改线上全局模型池或反复申请随机模型。

桥接模式不需要另开 Policy Server 进程：该项目的 YAML catalog 用于核对分配的
模型身份，实际 agent 就在 Inspect 进程内初始化。Router 模式使用现有 Protobuf
生命周期协议，catalog 必须与其分配完全一致。本地模式明确标记
`model_selection: local` 和 `inspect-robots-local-v1`，不伪造 Router readiness、
assignment 或执行确认，只发布实际完成并评分的轨迹。

## 安装和配置

在项目根目录使用已经安装 Yam 和 agent 插件的 Inspect 环境：

```bash
uv pip install --python .venv/bin/python -e integrations/colosseum-client
uv pip install --python .venv/bin/python -r integrations/excel-requirements.txt
cp integrations/colosseum-client/configs/robot.inspect-yam-account.yaml.example .yam/colosseum-client.yaml
cp integrations/colosseum-policy-server/configs/local-runtime-yam-account.yaml.example .yam/colosseum-runtime.yaml
```

系统需要 `ffmpeg`。填入 `.yam/colosseum-client.yaml` 中的 Router URL、Client token、
institution 和 LLM key 环境变量。真实硬件设置由原生 `.yam/config.ini` 提供，
桥接不使用 Colosseum Client 的硬件驱动。

旧的全 API 示例 catalog 使用 `https://api.yhlxj.ai` 中转，需要有效的中转 key，不能使用
Claude Code 的订阅登录替代。修改模型地址时必须同时修改 Router 注册和 Client
的 `llm_api_base_urls`。后者是本地实际调用地址；发给 Router 的 `llm_api_urls`
仍使用其 Next schema 规定的官方 provider 地址作为能力标识，不能发送中转地址。
不配置本地地址时仍使用官方地址。key 不会发送给 Router，也不写入桥接 manifest。

配置了 key 或支持的账号登录的 provider 会向 Router 声明可用。Router 仍负责选模型；
桥接遇到 catalog 不包含的模型会在创建机械臂前停止。

原生桥接也支持 OpenAI/xAI API key 与 Claude Code 账号登录混用：

```yaml
llm_api_keys:
  openai: env:OPENAI_API_KEY
  xai: env:XAI_API_KEY
llm_account_logins:
  anthropic: claude-code
llm_api_base_urls:
  openai: https://api.openai.com/v1
  xai: https://api.x.ai/v1
  anthropic: https://api.anthropic.com/v1
```

使用已安装的官方 `claude` CLI，运行 `claude auth login` 登录，然后用
`claude auth status` 检查。本地预检只调用登录状态检查；实际推理使用原生
agent 已实现的 `wire=claude-code`，不读取登录凭据文件，不提取 OAuth token，
也不需要 `ANTHROPIC_API_KEY`。同一 provider 不能同时设置 API key 和账号登录。
该模式只用于 `inspect-robots colosseum`，普通独立 Colosseum Client/Policy Server
仍需要 API key。账号登录的内建工具和 MCP 被禁用，机械臂仍由 Inspect 执行。

当前 `.yam` 配置使用官方 OpenAI/xAI API 地址。官网生成的 key 应配合这些
官方地址使用；全中转示例仅用于持有对应中转服务 key 的设置。

账号登录只声明官方 Anthropic 来源，不能代替 `api.yhlxj.ai` 上的 Claude key。
Client 和 runtime 中的 Claude URL 都必须是 `https://api.anthropic.com/v1`，
Router 也必须注册/启用相同 URL、模型名称、revision 和动作契约。
桥接将完整的 Router 模型名传给 Claude CLI，不使用 `opus` 等浮动别名替换它。
没有可用的官方注册模型时，客户端修改不能让 Router 分配账号登录模型。
新 run 的 `recording-context.json` 和上传元数据会记录 `native_llm_auth: claude-code`。
可参考 `robot.inspect-yam-account.yaml.example` 和
`local-runtime-yam-account.yaml.example`；现有全 API key 示例继续保留。

先做本地检查，环境变量可以通过外部凭据管理设置：

```bash
.venv/bin/inspect-robots colosseum \
  --client-config .yam/colosseum-client.yaml \
  --runtime-config .yam/colosseum-runtime.yaml \
  --dry-run -- --config .yam/config.ini --embodiment yam_arms
```

`--dry-run` 验证配置、key 是否已设置和 catalog 地址匹配；不连接 Router、不调用
模型，也不初始化机械臂。启用账号登录时会调用 CLI 的登录状态检查。
它不证明 key 有效、订阅能调用指定模型或 Router 已注册/启用这些模型。

## 正式评测

只录制 A/B、随后在本项目的网页里评分和上传时，使用 `--record-only`：

```bash
.venv/bin/inspect-robots colosseum \
  --client-config .yam/colosseum-client.yaml \
  --runtime-config .yam/colosseum-runtime.yaml \
  --local-pair --record-only \
  --instruction "Pick up the candy box and place it into the bowl." \
  -- --config .yam/config.ini --embodiment yam_eef \
  -E eef_orientation=true \
  -P effort=medium -P max_llm_calls=60 -P max_speed_frac=0.05 -P images=always

.venv/bin/inspect-robots colosseum \
  --client-config .yam/colosseum-client.yaml --review --review-open
```

本项目默认地址为 **http://127.0.0.1:8767/**，与原 Colosseum Client 的 8765 分开。
可用 `--review-port PORT` 改端口。网页列出 client 配置 `evaluation_dir` 中的新旧
Inspect bridge 记录，支持三路视频同步播放、变速、查看实际 wire、effort、频率、
调用预算、步数和新增记录的动作边界、guardrails 等参数。
原始 PNG 预览保留采集时间间隔；已导出的视频按数据集声明的名义 FPS 播放。

`--record-only` 不收集人工评分、不导出 LeRobot、不上传结果或数据，A/B 都完成后停止。
它支持本地配对及 Router 分配；后者仍发送 assignment 和 ready/step/finish 生命周期。
网页的 **Save locally** 仅保存本地评分和偏好草稿；本地配对用整数 0–4，Router 用
整数 0–100。完整填写后点击 **Save + Submit all data** 才运行当前项目的
`--submit-only --resume ID`，导出数据、提交评分并上传，绝不重新执行机械臂。
网页提交要求 `skip_upload: false`。上传中断可用 **Retry / finish upload** 补传。
尚未提交的旧导出可重新评分：提交时校验录制来源和已有文件，仅更新评分元数据/CSV，
不修改视频或状态/动作数组。开始发布后评分锁定。Discard 保留本地证据；
本地 pair 不联系 Router，Router assignment 则调用现有 abort 接口。

新录制的 `recording-context.json` 和上传数据集的 `meta/colosseum.json` 保存实际执行
参数，网页记录的 notes 也保留各 run 的参数。上传时不会用后来修改的 runtime 覆盖它们。
审阅服务器只监听 localhost；浏览和保存不初始化 agent、机械臂或 Router 连接。

对齐独立 `run` 的参数会改变评测条件和实际轨迹，不必改变 LeRobot 的数据结构。
wire、effort、调用预算、速度、EEF 姿态边界应按实际运行记录，并在 A/B 间保持一致。
频率还影响插值、视频 FPS 和名义时间戳；桥接从 Client 的 `control_hz` 读取频率，
runtime 中每个模型的 `control_hz` 必须与它一致（当前项目为 30 Hz），
不能只将原生控制改为 10 Hz 而继续上传声明为 30 Hz 的数据。Router 分配模式还必须
满足服务端模型注册；本地手动导入是否纳入排行榜由服务端决定。
已经录好的轨迹不能通过改配置变成另一频率、wire 或预算下的试验，需按新条件重新录制。

默认使用本地配对，明确选择两个模型：

```bash
.venv/bin/inspect-robots colosseum \
  --client-config .yam/colosseum-client.yaml \
  --runtime-config .yam/colosseum-runtime.yaml \
  --local-pair --model-a gpt-6-astra --model-b grok-4.7 \
  --defer-submit \
  --instruction "Pick up the candy box and place it into the red plate." \
  --scene "Candy box on the table; red plate on the left." \
  -- --config .yam/config.ini --embodiment yam_arms \
  -P images=always -P max_speed_frac=0.05 -P max_llm_calls=60 \
  --max-action-delta 0.05 --store-frames
```

`--` 前面是 Colosseum 参数，后面是标准 Inspect `run` 参数。桥接绑定评测的
instruction、max_steps 和模型身份，因此不能通过 `-P` 改 model 或 base_url。
wire 沿用独立 Inspect 命令：GPT/Grok 使用 `responses`，Claude 账号登录使用
`claude-code`，Claude API key 默认使用 `messages`。模型的 `backend_options.wire`
可配置协议，原生 INI `[policy.args] wire` 和显式 `-P wire=...` 依次覆盖它；
账号登录要求 `claude-code`。当前 `.yam` 配置与全部 LLM 运行脚本使用 800 执行步预算，
它与 `max_llm_calls` 的推理调用次数预算分别配置。
每个 run 固定为一个 scene、一个 epoch。实际评分由 Colosseum 的 0–100 提示收集；
原生 scorer 默认只记录 episode_length。

`--model-a/--model-b` 只接受这三个 LLM，且 A/B 必须不同。命令执行 A，恢复场景后
执行 B；提交前不会联系 Router 获取模型。本地模式评分使用整数 0–4（0 无进展，
4 完全成功），与已部署的数据集导入器 `score-0-4-success-4-v1` 一致；保存原始分数，
网页 A/B 和 Excel 中的完成度为 score / 4。输入偏好、反馈和 clear/close/uncertain
的置信程度。原生 Inspect 的 episode_length 仍只是运行指标。

保留 `--local-pair`，同时省略 `--model-a` 和 `--model-b` 时，会从 runtime 中
已配置 API key 或账号登录的 LLM 随机选出两个不同模型，并随机安排 A/B。
至少需要两个模型具备凭据；选中的账号登录仍会检查有效登录状态。
终端会打印选择结果，manifest 保存实际模型和 `model_selection: random`。
`--resume`、`--submit-only` 沿用已保存的模型，不会重新抽选。
如果只传其中一个参数，会报错；明确选择时需同时传两个。
`--dry-run` 的随机选择仅用于检查，不会锁定下一次真实评测的模型。

可做 GPT/Grok、GPT/Claude、Grok/Claude 三组配对。当前脚本用
`YAM_MODEL_A`、`YAM_MODEL_B` 选择模型，未设置时默认 GPT/Grok。
本地模式允许多组未提交配对，每组保存为 `logs/colosseum/local_<uuid>/`。
每次选择的全部模型都会先检查凭据；缺少任一个模型的凭据时，不开始 A。

回看后提交某个本地 pair，不再启动机械臂：

```bash
.venv/bin/inspect-robots colosseum \
  --client-config .yam/colosseum-client.yaml \
  --submit-only --resume local_YOUR_ID
```

上传使用 `/api/evaluations` 创建明确 A/B 的手动网页记录，`/api/datasets` 注册每个
run 的 LeRobot 数据集、逐文件上传、校验并发布，再上传网页 A/B 视频并完成评测记录。
web evaluation ID 和 dataset ID 保存到本地 manifest；注册不会伪造 HF revision，
真实 API 模型名/revision、EEF/IK 来源和实测数据保存在数据集元信息。
这属于本地选择的手动配对和数据集导入，不是 Router 自动分配的 assignment。
导入器是否将数据纳入排行榜仍由服务端规则决定，客户端不声称获得正式配额或排名资格。
Client token 必须有这些接口的上传权限；真实上传仍需在完成实机评测并回看后验证。
`test: true` 的本地试验只允许本地保存，现有手动评测 API 没有 test 字段，不能冒充实机数据提交。

发布中断后重复上述命令会复用已有网页记录和 dataset ID，并补传缺失文件。
创建网页记录的响应丢失时，用 notes 中唯一的本地 ID 找回它；若无法确认创建结果，
停止创建新记录，避免重复发布。`--skip-upload` 仍只延后录像/数据集，网页评分会提交。
普通 `--resume local_YOUR_ID` 保留暂存决定；`--submit-only` 才发布。
旧 Router assignment ID 不能转换为本地 pair；新的本地命令不会执行其 VLA 分配。

当前实现支持 **Yam 双臂绝对 joint_position、14 维**，执行频率由 Client
`control_hz` 决定（当前为 30 Hz），录制和导出使用相同频率。执行频率和动作
接口会显式传给原生 Yam 插件。可选 `yam_arms` 关节控制，或 **`yam_eef` 末端控制**：
在上面的命令中将 `--embodiment yam_arms` 改为 `--embodiment yam_eef` 即可。
安装当前项目中的 `plugins/inspect-robots-yam-eef` 后，agent 使用 EEF `move_to`，
Yam 通过 IK 转成关节目标；导出的 LeRobot `action` 是 IK 和关节限位处理后实际
发送的归一化关节命令，保持 Router 的 `joint_position` 契约。原始 EEF 目标保留
在本地 Inspect `actions/` 日志；`recording-context.json` 会标明原生 EEF 模式和
关节命令来源。缺少本步 IK 命令时会中止评测，不用 EEF 向量、实测状态或上一帧
命令代替。

`yam_eef` 会沿用指定 `.yam/config.ini` 中属于 `yam_arms`/`yam_eef` 的 CAN、相机
和其他 Yam 设置，再应用桥接与显式 `-E` 参数。默认 `auto_start=False`、
`unattended=False`，并保留原生 guardrails、IK 限位和硬件关闭逻辑。
EEF x/y/z 单位为米，位于各机械臂自己的基座坐标系；yaw/pitch/roll 单位为 rad，
相对 reset 时的姿态。原生 EEF 模式的可达性、姿态范围和碰撞检查限制仍然适用，
详见 [EEF 插件说明](../plugins/inspect-robots-yam-eef/README.md)。
API agent 的 stop 请求结束原生 Inspect trial，成功与否仍由人工评分。

自动上传发生在**整次 assignment 的所有 run 完成并评分之后**。Open track 通常
需要完成 A/B、分别评分，再输入偏好和反馈；随后提交评分并上传两份数据集。
它不会在 A 刚结束时提前上传。需要 `recording: true`、`skip_upload: false`；
命令行的 `--skip-upload` 也会延后上传。

需要先保存全部 A/B run、回看后再正式提交时，在 `--` 前加 `--defer-submit`。
它完成原生运行、人工评分、偏好/反馈和 LeRobot 导出，并更新本地 Excel，
但不调用最终 result 提交或数据集上传接口。Router 仍接收 assignment 请求、
ready/step/finish 和心跳；这个模式不是完全离线运行。
`--skip-upload` 与它不同：前者会立即提交评分，只延后数据集。

每个 run 的本地 Inspect 回看可在另一个终端启动（替换 ID；8765 已被占用时用 8766）：

```bash
.venv/bin/inspect-robots view \
  logs/colosseum/ev_YOUR_ID/run_YOUR_ID/inspect \
  --serve --port 8766 --open
```

打开 `http://127.0.0.1:8766/` 查看对话、动作和相机帧。
查看器只扫描指定目录及下一层标准 run 目录，因此应指定每个 run 的 `inspect`
目录，而不是直接传 `logs/colosseum`。导出的完整 MP4 在该 run 的 `lerobot/videos/`
下；Excel 的视频链接也可用于检查录像。查看器中的 Inspect episode-length 指标
与 Colosseum 人工评分不同，后者保存在 manifest 和 Excel。

回看通过后显式提交（不会重跑机械臂，也不会重新询问已保存评分）：

```bash
.venv/bin/inspect-robots colosseum \
  --client-config .yam/colosseum-client.yaml \
  --submit-only --resume ev_YOUR_ID
```

未完成或缺少本地评分的 assignment 会被 `--submit-only` 拒绝。
延后提交状态会保存在 manifest；普通 `--resume` 仍保留这个状态，只有显式
`--submit-only` 才正式发布。若配置 `skip_upload: true`，正式提交仍会延后数据集。
无效评测可用下文 `--abort` 关闭；它不会删除已上传的数据或 Excel 中的行。
Router 模式一次只允许一组未提交的 assignment。本地模式允许积攒多组配对，
回看后按每组的 `local_...` ID 提交。

相机映射：`top_cam → head_image`、`left_cam → left_image`、
`right_cam → right_image`。状态和动作顺序是左臂六关节、左夹爪、右臂六关节、
右夹爪。关节单位 rad；夹爪 0 闭、1 开。EEF 模式记录本步 IK 后最终发送的关节
命令；关节模式记录 guardrail 检查后的关节动作。
不把命令当作物理到位证明。缺失的状态或相机不会用零补齐。

录制时间戳是原生 reset/step 返回观测时的单调时钟，记录在推理之前，保留 API
等待造成的时间间隔；它不是相机曝光时间。LeRobot 的 `timestamp` 仍是
frame_index / 30，真实间隔保存在 `observation.capture_timestamp`。最后一个
无动作的观测只保留在源 JSONL，符合 Client exporter 的现有约定。

## 输出与恢复

要同时更新本地 Excel，在 `--` 前添加：

```bash
--excel /home/rc1/inspect-robots/logs/colosseum/results.xlsx
```

`_yam_instructions/_instructions.sh` 中的 Colosseum 命令已经包含这个参数。
当前本地配置的 `evaluation_dir` 是 `/home/rc1/inspect-robots/logs/colosseum`，
因此从其他目录启动时，日志仍保存在该项目。已有 Excel 已复制到该目录，
原 `/home/rc1/lerobot/eval_logs/ROBOCOLOSSEUM/results.xlsx` 不再由此脚本更新。
每次保存评分、提交结果或上传数据集时，会更新同一工作簿的 `Colosseum Evals`
工作表。每个 run 一行，包含任务、模型、A/B、评分（0–1 和 0–100）、偏好、反馈、
评测人、EEF/关节模式、提交/上传状态，以及本地日志和视频链接。difficulty 只在
已有数据提供时填写，不会自动推测。未评分的 run 不会被填成失败。

旧工作表属于固定任务的 VLA 统计；它们会保留。首次追加新工作表前，会创建
`results.before-colosseum.xlsx` 备份。按 Router/evaluation/run ID 更新同一行，
`--resume` 不会重复添加；Excel 路径保存在 manifest，续传时无需再指定。

Excel/LibreOffice 打开文件时，更新会延后并保存 `excel_pending` 标记；评分提交和
数据上传仍继续，命令退出码为 1 表示有 Excel 更新待完成。关闭表格后执行：

```bash
.venv/bin/inspect-robots colosseum \
  --client-config .yam/colosseum-client.yaml --export-excel-only
```

这个命令只读取本地评测记录，不联系 Router，不调用模型，也不启动机械臂。
可加 `--resume ev_YOUR_ID` 仅更新一个评测；加 `--excel FILE.xlsx` 可更换或选择
输出工作簿。文件采用临时文件加原子替换，写入失败不会损坏原工作簿。
如果用旧的 `build_results_xlsx.py` 重建工作簿，它会覆盖新工作表；随后可用
`--export-excel-only` 从 manifest 恢复 Colosseum 行。

默认文件结构：

```text
logs/colosseum/<evaluation-id>/
  manifest.json
  router-models.json     # 收到的 Router 公共模型描述（包括被拒绝的模型）
  selected-models.json   # 本地模式实际选择的 LLM 描述
  <run-id>/
    inspect/             # 原生 Inspect JSON、frames、actions、transcripts
    inspect-result.json  # 用于桥接恢复的评测结果副本
    bridge-trial.json    # 本地完成与 Router finish 状态
    frames.jsonl
    recording-context.json
    policy.json
    lerobot/             # 标准 LeRobot 数据集
      eval_trials.csv    # 本地配对导入的原始 0–4 评分
```

Client exporter 在成功导出后删除其临时 PNG；原生 Inspect 的帧仍保留。
上传的是 LeRobot 数据集；原生 agent 对话和 wire 记录保留在本地，不会随数据集上传。

提交或上传失败后，使用相同配置和命令尾部恢复：

```bash
.venv/bin/inspect-robots colosseum \
  --client-config .yam/colosseum-client.yaml --resume ev_YOUR_ID
```

已完成的 run 不会重跑，已保存的评分不再询问，导出会验证并复用现有数据。
Router 已完成的 assignment 也可恢复上传。未执行的 B 需要同时传回
`--runtime-config` 和原来的原生参数。CLI 异常、断连或 Ctrl-C 不会被记为成功。
如果 trial 尚在 running 或已 interrupted，需要通过 Router 中止该 assignment；
桥接不会重用其 run ID 执行新的轨迹，也不会自动关闭其他未提交实验。

```bash
.venv/bin/inspect-robots colosseum \
  --client-config .yam/colosseum-client.yaml \
  --abort ev_YOUR_ID --reason "Interrupted native trial"
```

用 `--skip-upload` 或 Client YAML 的 `skip_upload: true` 可以提交评分并延后
数据上传。之后可以用上面的 resume，或现有 Client 命令补传：

```bash
PYTHONPATH=integrations/colosseum-client/src .venv/bin/python \
  -m colosseum_client.dataset_upload .yam/colosseum-client.yaml \
  --evaluation-id ev_YOUR_ID
```

桥接不会把之前的 `logs/yam/claude` 历史实验自动注册成正式评测，因为它们没有
Router assignment，原生旧日志也缺少完整状态序列。

## 本地验证

```bash
.venv/bin/python -m pytest -q tests/test_colosseum_bridge.py
.venv/bin/python -m pytest -q tests/test_colosseum_excel.py
PYTHONPATH=integrations/colosseum-client/src .venv/bin/python -m pytest -q \
  integrations/colosseum-client/tests/test_llm_credentials.py \
  integrations/colosseum-client/tests/test_lerobot_export.py
```

桥接测试使用假的 Router 消息和模拟双臂，实际运行 Inspect rollout、Protobuf
编解码、录制、FFmpeg 和 LeRobot 导出，并验证提交/上传失败后的恢复。
EEF 插件测试用假驱动与运动学验证 IK 命令采集和夹爪单位转换。
真实 Router 接受流程、云 API 权限和实机轨迹仍需在部署环境运行验证。
