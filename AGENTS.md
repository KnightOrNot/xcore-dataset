# 协作边界

先阅读本项目 README；作为 controller 子模块时阅读父目录 `../AGENTS.md`，
独立 checkout 时阅读相邻 `../xcore-controller/AGENTS.md`。
本项目只负责远端记录／转换／浏览，不直接打开 SDK 或发运动命令。
共享 lerobot-converter 和上游 LeRobot／LeLab 不可修改；代码只使用其公开 API。
LeRobot 0.6.0＋锁定 LeLab，使用 CPU Torch；不要遗漏直接 Torch 依赖和 uv CPU index，
否则可能解析出 CUDA 大包。数据不入 Git，原始记录和发布 revision 均不可变。
图像时间语义：clock_uncertainty_ns 是时钟误差，age_ns 是采样时帧龄；转换只用前者，
后者写入质量报告。max_skew_ms 默认 100ms（15fps 双相机丢帧时最近帧约 68ms）。
当前真实 writer/loader 的双 RGBD、EE10、12 维力、掩码、中文 task 回归已通过；
实际数据验收只完成合成数据，不代表现场运动、同步或连续录像已验收。
LeLab 默认只开放浏览；2026-10-26 用户明确要求后可删除**已发布 revision**：
只允许 <dataset>/rev-NNNN 且存在 READY，拒绝路径穿越、symlink、raw/staging/jobs
和 Hub 删除；原始 raw 永不在此接口范围内。不能顺带开放运动或外部上传。

2026-10-26：新增 pose10／joints 两种表示。joints 从 raw 的 joints_rad／target_rad／
closure／target_closure 构建，七维、rad；缺少字段必须拒绝，不能对 pose10 做 IK。
发布目录写 representation.json；joints 默认 dataset_id 为 `<session>-joints`，
不得与 pose10 互相覆盖。pull 拉取前必须校验表示。

### 修复：真实会话未出现在 LeLab（2026-10-26）

- 用户看到 LeLab 只有三个合成版本，怀疑读的是本地数据。核实：面板链接到
  `127.0.0.1:8090` SSH 隧道，远端 `xcore-dataset.service` 的 `--root` 为
  `/home/rokae/project/xcore-data/lerobot`；LeLab `/datasets` 与远端目录一致，
  **确实是远端数据**。缺少的是刚保存的 session，因为转换 job 失败。
- 两个转换缺陷：
  1. 多行机器人样本引用同一物理相机帧时，`uncertainty_ns`（记录时的帧龄）随样本不同；
     旧代码把整个 image dict 当身份比较，误报 `conflicting image metadata at same
     timestamp`。现在只比较 path／帧号／时间等身份字段，`uncertainty_ns`／`age_ns`
     取保守最大值。
  2. 转换把记录时帧龄当作时钟误差加到匹配误差里，导致 50ms 容差拒绝了实际最近帧
     仅 33–68ms 的真实片段。现在只用 `abs(frame_time-target)+clock_uncertainty_ns`；
     旧 `uncertainty_ns`／新 `age_ns` 的最大值写入
     `quality_report.max_recorded_frame_age_ms`。新录制行写
     `clock_uncertainty_ns=0`、`uncertainty_ns=0`、`age_ns=<采样帧龄>`。
- dataset `max_skew_ms` 默认从 50 改为 **100ms**：15fps 双相机偶发丢帧时最近帧
  可达约 68ms；质量报告仍记录逐流真实误差，不能把 100ms 当作硬同步。
- 用修正后的代码重跑三个失败 job（job 记录的旧 50ms 容差同步更新为 100）：
  - `session_20261026_123845/rev-0001`：117 帧、1 episode（用户刚保存的数据）
  - `session_20261026_111111/rev-0001`：280 帧、1 episode
  - `session_20261026_112434/rev-0001`：414 帧、2 episodes
  全部 `READY`；远端 LeLab `/datasets` 已列出，`/api/sample` 可读回真实 pose10、
  task 和 640×480 深度。测试后未改动机器人、相机或夹爪状态。
- 验证：dataset 10 项、controller recording 6 项通过；`datasets`/LeLab 实读通过。
