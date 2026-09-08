# 从 NAS 加载 OpenPI π0.5 checkpoint

## 当前机器检查结果

- 本地缓存路径：`/home/jjn/.cache/openpi/openpi-assets/checkpoints/pi05_droid`
- 当前大小：约 12 GB
- NAS 挂载点：`/mnt/yuantao_nas`
- 宿主机 NAS 状态：可写挂载，约 465 GB 可用。Codex 沙箱中的 `ro` 是工作区外
  路径的沙箱限制，不代表宿主机挂载状态。
- CALVIN 迁移前源目录：`/mnt/yuantao_nas/datasets/calvin`，约 518 GiB
- CALVIN 迁移前目标目录：`/mnt/yuantao_nas/data/calvin`，包含一个约 121 GB 的
  未完成 `task_ABC_D.zip`

OpenPI 会把 `gs://openpi-assets/checkpoints/pi05_droid` 缓存在
`~/.cache/openpi/openpi-assets/checkpoints/pi05_droid`。它也接受本地目录作为
`--policy.dir`，因此从 NAS 启动时不需要修改 OpenPI 源码。

## 迁移

先把 NAS 改为可写挂载，并选择一个明确的个人目录。不要直接删除本地缓存。
使用仓库中的迁移脚本先复制、校验 checkpoint，再迁移 CALVIN：

```bash
cd /home/jjn/jjn/proj/vla/inspect-robots

nohup scripts/migrate_openpi_checkpoint_to_nas.sh \
  /home/jjn/.cache/openpi/openpi-assets/checkpoints/pi05_droid \
  /mnt/yuantao_nas/openpi/checkpoints/pi05_droid \
  > /tmp/migrate_pi05_droid.log 2>&1 &

tail -f /tmp/migrate_pi05_droid.log
```

脚本包含两个连续任务：

1. 把 OpenPI checkpoint 复制到 NAS，执行 checksum 校验并保留本地源目录。
2. 把 `/mnt/yuantao_nas/datasets/calvin` 迁移到
   `/mnt/yuantao_nas/data/calvin`。

CALVIN 的源和目标位于同一个 CIFS 文件系统，因此第二项使用目录 rename，不重新复制
518 GiB。若目标目录非空，脚本先将其重命名为带
`.preexisting.<timestamp>.<pid>` 后缀的备份。本机检查时已有约 120 GB 的未完成目标文件，
而且检查时仍在增长，说明有其他主机或 NAS 端任务正在写入。脚本会观察目标文件元数据
30 秒；只要检测到变化就立即退出。确认没有活动写入后，它才会保留旧目标并执行 rename，
不会覆盖或删除已有数据。

如 checkpoint 已经迁移，只执行 CALVIN 任务：

```bash
scripts/migrate_openpi_checkpoint_to_nas.sh --calvin-only
```

## CALVIN 实际迁移结果

2026-09-07 已完成同文件系统 rename：

- 完整文件现在位于
  `/mnt/yuantao_nas/data/calvin/task_ABC_D.zip`，大小 `555309812705` bytes；
- 原路径 `/mnt/yuantao_nas/datasets/calvin` 已不存在；
- 迁移前的部分文件保留在
  `/mnt/yuantao_nas/data/calvin.preexisting.20260907T114728Z.811941/task_ABC_D.zip`，
  大小 `121371594752` bytes。

NAS 启动验证成功后，如确实需要回收本地 checkpoint 的 12 GB，再由操作者单独删除
本地源目录。确认 CALVIN 新目标完整可用后，也可人工处理上述 partial 备份。

> [!IMPORTANT]
> `nohup` 只能抵抗终端或 SSH 断开。电脑关机、休眠或 NAS 断开时，复制与模型
> 服务都会停止。中断的复制会保留 `.partial` 目录，再次执行前应先确认其内容。

## 从 NAS 启动 server

前台启动并完成第一次模型加载验证：

```bash
cd /home/jjn/jjn/proj/vla/inspect-robots
scripts/serve_pi05_droid_from_nas.sh \
  /home/jjn/jjn/proj/vla/openpi \
  /mnt/yuantao_nas/openpi/checkpoints/pi05_droid \
  8000
```

确认能加载后，可以让它在终端断开后继续运行：

```bash
mkdir -p /home/jjn/jjn/proj/vla/openpi/logs
nohup /home/jjn/jjn/proj/vla/inspect-robots/scripts/serve_pi05_droid_from_nas.sh \
  /home/jjn/jjn/proj/vla/openpi \
  /mnt/yuantao_nas/openpi/checkpoints/pi05_droid \
  8000 \
  > /home/jjn/jjn/proj/vla/openpi/logs/pi05_droid_nas.log 2>&1 &
```

也可以保留原来的 `gs://` 参数，并在启动前设置：

```bash
export OPENPI_DATA_HOME=/mnt/yuantao_nas/openpi/cache
```

此方式要求 checkpoint 位于
`$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_droid`。显式传 NAS 本地目录更容易
审计，也不会意外重新下载模型。
