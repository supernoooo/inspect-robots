# isaacsim-2yam 与 LIBERO 插件增加日志

## 目标

本次增加两个独立的 Inspect Robots 插件：

1. `inspect-robots-isaacsim-2yam`：在 Isaac Lab 中评测双臂 YAM。
2. `inspect-robots-libero`：在官方 LIBERO 中评测单臂 Franka。

没有修改原有 `plugins/inspect-robots-isaacsim`。Franka Isaac 配置、Isaac Lab
site-packages 和 MolmoAct2 源资产均保持不变。

## 目录与注册入口

两个插件都采用现有 `plugins/*` 的独立 Python 包结构，包含 `pyproject.toml`、
`README.md`、`src/<package>`、`tests` 和 `py.typed`。

`inspect-robots-isaacsim-2yam` 注册：

- embodiment：`isaacsim-2yam`
- task：`isaacsim-2yam-put-everything-in-box`

`inspect-robots-libero` 注册：

- embodiment：`libero`
- policy：`molmoact2-libero`
- tasks：`libero-spatial`、`libero-object`、`libero-goal`、`libero-10`、`libero-90`

根目录 `uv.lock` 已加入两个 workspace 包及 LIBERO HTTP adapter 的依赖。

## isaacsim-2yam 的实现过程

### 静态协议

`contract.py` 定义 MolmoAct2 YAM 的 14 维绝对关节位置协议：

```text
left j0..j5, left gripper, right j0..j5, right gripper
```

夹爪值 `0` 表示关闭，`1` 表示打开。观测包含 `top_cam`、`left_cam`、
`right_cam` 三路 `360x640` RGB 图像，以及 14 维 `joint_pos`。

### Isaac Lab 环境

`isaac_env.py` 实现独立 `DirectRLEnv`：

- 物理频率 120 Hz，控制频率 30 Hz；
- 使用 MolmoAct2 下载的双臂 YAM MJCF；
- 创建桌面、开放盒子、Duplo 代理和球体代理；
- 创建顶部相机和左右腕部相机；
- 把 14 维 wire action 映射到 12 个手臂关节和 4 个手指关节；
- 两个物体都进入盒子时报告 privileged success。

为避免依赖 Isaac 远程 Nucleus 资产，Duplo 和网球使用本地 primitive 代理。
因此该任务保留控制、相机和成功条件，但画面并非 ManiSkill YCB 资产的逐像素移植，
两边成功率不应直接横向比较。

### MJCF 兼容处理

`asset.py` 在临时目录中复制并规范化 YAM MJCF：

- 为右臂复用的 OBJ 创建唯一文件名，避免 Isaac OBJ 转换竞争；
- 显式补全继承的 geom 类型；
- 为固定 articulation wrapper 补充惯性；
- 保持源 MJCF 与 mesh 完全不变；
- 环境关闭时先删除临时资产，再关闭 SimulationApp。

真实启动过程中还确认了 Isaac Sim 5.1 的 articulation root、MJCF importer extension、
腕部相机 USD prim path 及场景坐标。机器人固定在原点，桌面、物体和盒子统一转换到
robot-base 坐标，避免 disjoint fixed joint。

### 延迟导入

Isaac Sim、Isaac Lab、Gymnasium 和 Torch 仅在第一次 `reset()` 时导入并启动。
这样插件发现、协议检查和单元测试可在无 Isaac 的普通 Python 环境运行。

## LIBERO 的实现过程

### 环境与任务

`embodiment.py` 延迟导入官方 LIBERO 的 `OffScreenRenderEnv`。每个 `Scene`
通过 metadata 选择 suite、task id 和官方 initial-state id。切换任务时关闭旧 MuJoCo
环境，单个任务内复用环境且不自动 reset。

`task.py` 从官方 task language 和 initial states 生成 Inspect Robots scenes，并为五个
suite 提供独立 factory。成功条件来自 LIBERO `check_success()`。

### MolmoAct2-LIBERO policy adapter

`policy.py` 对接 MolmoAct2 通用 `/act` server：

- `agentview_image` 旋转 180 度后映射为 `image`；
- `robot0_eye_in_hand_image` 旋转 180 度后映射为 `wrist_image`；
- state 为 EEF 位置 3 维、四元数转 axis-angle 3 维、gripper qpos 2 维；
- action 为 7 维相对 EEF pose 与 gripper；
- 请求包含 `norm_tag=libero` 和 `n_action_steps=10`；
- HTTP body 使用 `json_numpy`，并显式设置 `Content-Type: application/json`。

LIBERO 没有作为普通 PyPI dependency 写入插件。官方项目包含 MuJoCo 资产、BDDL、
initial states 和独立配置文件，因此应在单独的 `libero-inspect` Python 3.10 环境中
安装官方源码，再安装 Inspect Robots 与本插件。模型 server 留在 MolmoAct2 环境，
二者仅通过 HTTP 通信。

## 使用方式

完整安装与命令分别位于：

- `plugins/inspect-robots-isaacsim-2yam/README.md`
- `plugins/inspect-robots-libero/README.md`

双臂 YAM 必须启动 `examples/yam/host_server_yam.py`，不能使用 DROID server。
LIBERO 必须启动 `experiments/scripts/serve_policy.py` 和
`allenai/MolmoAct2-LIBERO`，也不能使用 DROID server。

## 验证记录

- `isaacsim-2yam` portable tests：27 passed，line/branch coverage 100%。
- `libero` tests：35 passed，line/branch coverage 100%。
- Ruff：通过。
- mypy strict：两个插件均通过。
- Inspect Robots 根项目回归：1720 passed，6 skipped。
- `uv lock --check`、Python compileall、wheel/sdist build 和插件 entry-point
  discovery：通过。
- Isaac Sim 5.1 + RTX 4090 真实启动：完成 MJCF import、environment reset 和一个
  action step；三路相机均返回 `(360, 640, 3)`，state 返回 `(14,)`。
- LIBERO live smoke test：当前机器尚未安装官方 LIBERO，未执行。其环境生命周期、
  图像/state 转换、task factory 和 HTTP contract 已由 mock API 单元测试覆盖。
