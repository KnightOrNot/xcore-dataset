# xcore-dataset

远端 CR7 数据工具的首个可运行版本：封存 raw → 校验／对齐 → LeRobot v3 →
真实 loader 回读 → 校验清单及不可变目录发布 → LeLab 只读浏览。
当前实际运行验证使用明确标记的**合成数据**，不是机械臂运动采集结果。

状态／目标均为 float32[10]；两台相机各一组 RGBD；力 float32[12]、有效掩码 bool[2]；
任务字符串通过 LeRobot task 元数据保存。RGB 为视频，深度为米制 float32 无损 image；
raw 保留 RGB PNG／原始 uint16 Z16、depth scale、内外参和逐流时间信息。

`requires-python` 为 3.11–3.12：转换、LeRobot 和 LeLab 使用独立 Python 3.12 环境；
远端 controller 的 3.11 SDK 运行时只导入 `xcore_dataset.raw`（numpy/Pillow），
不加载 LeRobot、FastAPI 或转换依赖。

## 本地离线检查

相邻目录须存在 `xcore-adapters/`。使用独立 Python 3.12 环境：

```bash
uv sync --extra dataset --extra app
uv run xcore-dataset demo data/raw/demo
uv run --extra dataset xcore-dataset convert data/raw/demo data/staging/demo data/lerobot/demo/rev-0001
uv run --extra dataset xcore-dataset verify data/lerobot/demo/rev-0001 --readback
uv run --extra dataset --extra app xcore-dataset serve --root data/lerobot
uv run --extra dataset --extra app pytest -q tests ../xcore-adapters/tests
```

默认频率／容差和服务端口在 `config/default.toml`，由 `--config PATH` 覆盖。
上述路径必须是新目录，重复运行换名；转换拒绝覆盖输出。
LeRobot 固定 0.6.0，LeLab 固定提交，Linux 使用 CPU Torch，版本见 `uv.lock`。
视频运行环境需可加载对应 FFmpeg 共享库；现场离线部署说明见 controller 的
`docs/REMOTE_DATA_MILESTONE.md`。

## 文件完整性

`EpisodeWriter` 只在图像存在、记录校验和落盘后发布 `COMMITTED.json`。
转换只接受封存输入；检查每路图像时间误差、时钟域和不确定度，状态和 action 保留同一源行。
原始数据和已发布数据均不覆盖；staging 失败保留供排查。
发布目录含 `READY.json`（文件清单／大小／SHA-256）、质量报告、版本、标定及采样审计。
本地拉取要复制完整目录，不能只取一个 parquet。此版本转换器一次处理一个已封存 episode；
多 episode/session 的任务队列和崩溃恢复尚未实现。

## 浏览

访问 `http://127.0.0.1:8090`，远端通过 SSH 转发。
`/edit-dataset` 为官方 LeLab 浏览器；当前只读，合并／上传／删除、校准、训练和控制接口不开放。
`/api/datasets` 列出已发布数据，`/api/sample?dataset=ID/revision&index=0`
显示状态、目标、12 维力和两路深度米制统计。
`/api/depth-preview.png?dataset=ID/revision&camera=cam1&index=0` 为深度灰度预览，
按 0～3m 映射，**不是训练数据本身**；cam2 同理。

## 尚未完成

真实 SDK arm／跟随后的连续 episode 已由 controller 接入（代码完成，真机未验收）；
跨机时钟估计、多 episode 自动转换任务队列与崩溃恢复、远端任务调度 API 仍待实现。
真机跟随采集均待完成。现有相机 snapshot 的 RGB 是 JPEG，不能转成 PNG 冒充无损采集。
硬件 TCP／模型未验证前不能将法兰当作夹爪 TCP，也不能将 FK 占位值当实测。
独立仓库：[KnightOrNot/xcore-dataset](https://github.com/KnightOrNot/xcore-dataset)，
controller 通过同名 submodule 引用。独立 checkout 与子模块不自动同步；
不上传合成或现场数据到 Git。
