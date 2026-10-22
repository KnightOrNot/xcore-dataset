# 协作边界

先阅读本项目 README；作为 controller 子模块时阅读父目录 `../AGENTS.md`，
独立 checkout 时阅读相邻 `../xcore-controller/AGENTS.md`。
本项目只负责远端记录／转换／浏览，不直接打开 SDK 或发运动命令。
共享 lerobot-converter 和上游 LeRobot／LeLab 不可修改；代码只使用其公开 API。
LeRobot 0.6.0＋锁定 LeLab，使用 CPU Torch；不要遗漏直接 Torch 依赖和 uv CPU index，
否则可能解析出 CUDA 大包。数据不入 Git，原始记录和发布 revision 均不可变。
当前真实 writer/loader 的双 RGBD、EE10、12 维力、掩码、中文 task 回归已通过；
实际数据验收只完成合成数据，不代表现场运动、同步或连续录像已验收。
LeLab 只开放浏览接口；增加能力时不能顺带开放运动、外部上传或删除。
