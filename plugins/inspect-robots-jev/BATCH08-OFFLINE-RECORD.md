# Batch 08 离线阶段门控记录（2026-09-22）

Batch 07 前置状态：**待现场验证**。其现场报告尚无真实 rig 配置/模型与时延门槛签发、独占设备 owner 与命令审计、YAM guardrail 现场注入、操作者/看护人签字。离线夹具中的签名、配置哈希与模型 ID 都是合成值；不授权真机动作。

本批软件在原 `AgentProposer` → `AgentCandidateValidator` → `JevChoiceClient` → `JevAgentPolicy` 路径上增加 `StagedMotionGate`。它使用 Inspect Robots `DefaultController` 逐步取所选前缀，逐步通过含 YAM `CollisionApprover` 的审批链，然后才调用注入的 embodiment。门控固定签发的 rig/模型指纹、观测年龄、微动上限、关节误差、唯一控制锁与人工准许。阶段 1 独立单步保持；阶段 2 覆盖双臂与两夹爪的单关节方向；阶段 3 至少两个安全候选及 hold 由 JEV 选择；阶段 4 至少两轮新观测与 Agent 反馈。审批修改、时龄/前缀/ID 不符、回读误差及异常在下一步假电机命令前停止。失败后只能人工复核并重新提议。

离线验收命令：

```bash
.venv/bin/python -m pytest -q plugins/inspect-robots-jev/tests/test_jev_agent_staged_motion.py plugins/inspect-robots-jev/tests/test_jev_agent_policy.py
```

实际结果：`38 passed in 0.25s`，退出码 0。全部测试使用假 embodiment、合成几何/图像/模型响应；YAM guardrail 是已安装包的内存审批器，没有连接 CAN、相机或真机。测试还证明候选在 JEV 前经同一 YAM 审批链的隔离预审、每步动作与回读及三路帧引用落盘、整场控制锁有效。

关联审计回归：`.venv/bin/python -m pytest -q plugins/inspect-robots-jev/tests/test_jev_agent_audit.py plugins/inspect-robots-jev/tests/test_jev_agent_replay.py plugins/inspect-robots-jev/tests/test_jev_agent_shadow.py`；实际结果 `48 passed in 1.10s`，退出码 0。

软件完成标志：**离线软件门槛满足**。真实四阶段现场记录：**缺失**；真实运动验收与 Batch 09 交接条件：**未满足**。低能力测试 Agent 的 `agent_expansion` 表现暂不阻塞此软件实现；正式候选/JEV 选择评估须使用更强 Agent 在现场另行验证。
