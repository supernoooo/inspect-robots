# SO-101 Colosseum 评测流水线

本流程沿用 YAM 的 `inspect-robots colosseum` 配对、录制、网页回看、评分、LeRobot v3 导出与上传逻辑。SO-101 使用 `so101_eef_configured`：agent 的 `move_eef` 工具生成 7 维 EEF 增量，插件经 IK 向单臂驱动发送关节目标。Colosseum 按每次 Router 分配的频率记录驱动实际接受的 6 维关节动作（五个关节为角度，夹爪为 0–100），以及前视与腕部两路相机。

## 准备

```bash
cd /home/magiclab/inspect-robots
uv pip install --python .venv/bin/python -e integrations/colosseum-client
uv pip install --python .venv/bin/python -e ./_so101_instructions
uv pip install --python .venv/bin/python -e ./plugins/inspect-robots-agent
uv pip install --python .venv/bin/python -r integrations/excel-requirements.txt
cp --no-clobber integrations/colosseum-client/configs/robot.inspect-so101-account.yaml.example .so101/colosseum-client.yaml
cp --no-clobber integrations/colosseum-policy-server/configs/local-runtime-so101-agent.yaml.example .so101/colosseum-runtime.yaml
```

录像需要 `ffmpeg` 在当前虚拟环境的 `PATH` 中。此工作站的 `imageio-ffmpeg` 已带可执行文件，可用 `.venv/bin/python -c 'from pathlib import Path; import imageio_ffmpeg; Path(".venv/bin/ffmpeg").symlink_to(imageio_ffmpeg.get_ffmpeg_exe())'` 在环境重建后补上入口，并以 `command -v ffmpeg` 检查。直接使用现有 `.so101/config.ini` 中的串口、相机、校准、URDF 和 EEF 设置；客户端 YAML 中填写 Router URL、token、institution，且 `cameras` 的值必须是原生相机名称（当前为 `front`、`wrist`），不是设备路径或序列号。runtime YAML 中的模型名称、revision、动作维度与 horizon 必须与 Router 分配的模型一致；模型 URL 使用你的官方 API 地址，频率由 Router 在每次运行时下发。按实际可用凭据删去不使用的 provider。API key 从环境变量读取，Claude Code 需先完成官方 CLI 登录。不要把真实凭据写进这些示例文件。

Claude Code 的账号登录与命令行程序路径是两件事。若已用捆绑 CLI 登录，可在本地 runtime YAML 的 `backend.options.llm.claude_command` 设置该可执行文件的绝对路径，或导出 `SO101_CLAUDE_COMMAND`；原生运行参数 `-P claude_command=/absolute/path/to/claude` 也会覆盖这两个设置。Colosseum 会在请求 Router 分配前检查登录状态。Router 若为 SO-101 分配相同名称、revision 和动作契约的 Claude 中转登记，桥接仍通过本机已登录的官方 Claude CLI 执行；该路径使用账号登录，不使用 Anthropic API key。旧命令中的 `.so101/colosseum-config.ini` 现链接到 `.so101/config.ini`，新命令请直接使用后者。

需要重新配置单臂设备时运行 `inspect-robots setup --config .so101/config.ini`。EEF 插件的 setup 槽位是 follower 串口和腕部相机；前视 RealSense 使用配置中的 SDK serial，原有校准和 URDF 设置会保留。

Router 模式下，Router 分配的模型 `control_hz` 会覆盖 SO-101 原生 INI 中的默认频率，并用于录像和 LeRobot 导出；Client 与 runtime YAML 中的 `control_hz` 是本地配对和离线预检的默认值。若在命令行显式传入 `-E control_hz=...`，该值必须等于 Router 分配频率。桥接仍会拒绝 `use_degrees=false`、增量关节控制和其它动作契约不匹配；模型的 `action_dim` 必须是 6。测试模式不能把真实硬件录像作为模拟数据提交。

Client YAML 当前的 `max_trial_steps: 1200` 会在申请评测时提交给 Router，原生试验以 Router 实际返回的 `task.max_steps` 为准。30 Hz 分配下，1200 步相当于每侧 40 秒的理论控制时间；如果 A/B 均为 30 Hz，两侧合计 80 秒。agent 每侧最多使用 60 次模型调用。`move_eef` 的 `steps` 是最少步数：工具按每步 2 mm 的 EEF 平移上限自动增加平滑插值步数，执行完整目标。Router 的 `max_horizon` 限制单次轨迹长度；目标超出时在运动前明确报错，不会悄悄执行半截。推理、机械臂等待、准备场景与导出耗时另计；`max_steps` 不是墙钟超时。EEF tracking 偏差不再向终端打印警告，误差数据仍留在每步结果中；无效 IK 与关节越界仍会停止动作。

## 本地检查与录制

```bash
.venv/bin/inspect-robots colosseum \
  --client-config .so101/colosseum-client.yaml \
  --runtime-config .so101/colosseum-runtime.yaml \
  --local-pair --model-a gpt-6-astra --model-b grok-4.7 \
  --instruction "Pick up the red block, lift it clear of the table, and put it back." \
  --dry-run -- --config .so101/config.ini --embodiment so101_eef_configured

.venv/bin/inspect-robots colosseum \
  --client-config .so101/colosseum-client.yaml \
  --runtime-config .so101/colosseum-runtime.yaml \
  --local-pair --model-a gpt-6-astra --model-b grok-4.7 \
  --record-only \
  --instruction "Pick up the red block, lift it clear of the table, and put it back." \
  -- --config .so101/config.ini \
  --embodiment so101_eef_configured \
  -P images=always -P max_speed_frac=0.05 -P max_llm_calls=60
```

`--dry-run` 只检查本地配置，不调用模型或硬件。真实录制会先执行 A，再由操作者恢复场景后执行 B；每次运动前仍由原生驱动询问操作者。终端会给出 `local_...` ID。`--record-only` 完成后不会评分或上传。回看、评分、提交使用单独的 `--review` 命令；启动新评测不会检查本地回看状态，也不会修改或丢弃旧数据。旧评测即使录制失败，也可以在修复原因后运行同一条不带 `--resume` 的命令请求新配对；Router 是否分配新 ID 仍由 Router 决定。桥接不会为新评测调用 Router 的 `/reset`。本地模型目录默认声明 `joint_position`、6 维、10 Hz；Router 模式实际频率以分配值为准。原生控制是 7 维 `eef_delta_pose`。

```bash
.venv/bin/inspect-robots colosseum \
  --client-config .so101/colosseum-client.yaml --review --review-open
```

在回看页面检查两路视频和动作，填写 A/B 评分及偏好后提交。`local_...` 配对会通过 Open Track 手动导入接口创建网页评测并上传数据集与视频；Router 分配的 `ev_...` 则提交到原评测 ID。上传中断可用原 ID 续传：

```bash
.venv/bin/inspect-robots colosseum \
  --client-config .so101/colosseum-client.yaml \
  --submit-only --resume local_YOUR_ID
```

Router 分配模式使用相同配置，去掉 `--local-pair` 和模型选择；Router 必须注册并分配与 runtime catalog 中名称、revision、6 维关节动作和 horizon 一致的 SO-101 LLM。YAM 示例中的 `--local-pair` 是本地选 A/B，不由 Router 分配。桥接按 Router 频率运行，但本地推理使用 runtime YAML 和 Client YAML 中一致的官方 API URL 与 API key。Router 当前可能分配 VLA 或不同的动作契约，桥接会在机械臂初始化前拒绝这些任务。网页手动导入能否进入排行榜由服务端规则决定。运行前检查工作空间、限位和校准；本仓库测试不代替实机验收。
