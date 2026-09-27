# Vision Arm Lab

固定 RGB-D 相机下的 Panda 单方块抓放仿真。实现真值专家、颜色/深度视觉策略、共享抓取状态机、批量评测、可选记录和动作回放。使用 MuJoCo / robosuite，不包含训练、真机、ROS 或多环境并行。

开发范围见 [PLAN](docs/PLAN.md)，阶段记录见 [M0](docs/M0.md)、[M1](docs/M1.md)、[M2/M3](docs/M2_M3.md)、[M4](docs/M4.md)。正式验收与限制见 [M5](docs/M5.md)。

## 环境

本机验证：Ubuntu 26.04、Python 3.10.21、robosuite 1.5.2、MuJoCo 3.3.0。完整版本见 `requirements-lock.txt`。

```bash
# 已有环境
source ~/miniforge3/etc/profile.d/conda.sh
conda activate arm_lab
python -m pip install -r requirements-lock.txt
python -m pip check
```

在新机器上可先用 conda 创建 `arm_lab`、指定 `python=3.10.21` 和 pip，再安装锁定依赖。以下命令都在项目根目录、已激活环境中执行，无需安装本项目包。

本机窗口和离屏实际使用 `llvmpipe` 软件渲染；设置 `MUJOCO_GL=egl` 不代表使用 NVIDIA GPU。NVIDIA 硬件渲染尚未验证。受限环境可将 `NUMBA_CACHE_DIR`、`MESA_SHADER_CACHE_DIR` 指向可写临时目录；第三方编译缓存不属于实验记录。

## 运行

离屏专家或视觉抓放（默认不保存实验文件）：

```bash
PYTHONPATH=src MUJOCO_GL=egl python -m vision_arm_lab.runner --config configs/mvp.yaml --policy expert --seed 0
PYTHONPATH=src MUJOCO_GL=egl python -m vision_arm_lab.runner --config configs/mvp.yaml --policy vision --seed 0
```

窗口观察视觉抓放：

```bash
PYTHONPATH=src MUJOCO_GL=glfw python -m vision_arm_lab.runner --config configs/mvp.yaml --policy vision --seed 0 --window
```

`--verbose` 输出状态机阶段转换。`--seed 0 1 2` 按顺序运行多个回合；一次只创建一个环境。默认运行至成功、失败或 60 秒超时。`--steps N` 可提前限制步数，策略未完成时报告 `step_limit`。

仅检查场景、保持末端并打开夹爪：

```bash
PYTHONPATH=src MUJOCO_GL=glfw python -m vision_arm_lab.runner --config configs/mvp.yaml --policy inspect --seed 0 --steps 100 --window
```

`inspect` 的 `running` 表示指定检查步数结束但任务尚未终止，不代表抓放成功。当前未接入可选键盘 XYZ 操作。

## 评测与记录

开发种子为 0–9；正式种子固定在 `configs/test_seeds.yaml`，不可删换失败种子。两种策略使用相同场景与成功判定。

```bash
PYTHONPATH=src MUJOCO_GL=egl python -m vision_arm_lab.runner --config configs/mvp.yaml --policy expert --seed-file configs/test_seeds.yaml --record summary
PYTHONPATH=src MUJOCO_GL=egl python -m vision_arm_lab.runner --config configs/mvp.yaml --policy vision --seed-file configs/test_seeds.yaml --record summary
```

`summary` 保存有效配置、种子、版本/源码摘要、逐回合结果和汇总。输出路径由终端的 `Records:` 给出，默认在被 Git 忽略的 `runs/` 内。

动作、原始观测和视频相互独立，必须显式开启：

```bash
PYTHONPATH=src MUJOCO_GL=egl python -m vision_arm_lab.runner --config configs/mvp.yaml --policy vision --seed 0 --record debug --actions --observations --video all
```

`debug` 本身不打开任何附加记录。参数如下：

| 参数 | 行为 / 默认值 |
| --- | --- |
| `--output` | 运行目录父路径，默认 `runs`；每次创建独立子目录 |
| `--budget-mb` | 总预算，按 MiB 换算；默认 128 |
| `--actions` | 逐控制步 JSONL |
| `--observations` | RGB、深度、标定、机器人状态的 NPZ |
| `--observation-hz` | 观测记录频率，默认 2，最高 20 |
| `--observation-episodes` | 最多记录前几个回合的原始观测，默认 2 |
| `--video` | `off`（默认）、`all`、`failures` 或 `selected` |
| `--video-episodes` | `selected` 模式的零基回合序号，如 `0 2`，不是随机种子 |
| `--video-max` | 最多保留的视频数，默认 3 |
| `--video-fps` | 视频采样/编码帧率，默认 10，最高 20 |
| `--video-size` | 方形视频分辨率，默认 256，必须为偶数 |

预算包括临时视频，并预留轻量结果空间。达到预算会停止附加记录、标注不完整，评测继续；写入失败会在终端报告。仅失败视频在成功回合结束后删除其临时文件。原始 RGB/深度不会随视频自动保存。

环境/软件错误单独计入 `environment_error`，不补抽种子；存在这类错误时命令最终返回非零退出码。中断返回 130。成功率分母包含所有已执行的尝试。

## 回放与临时文件

只有 `result.json` 中 `actions_complete=true` 的回合可重放，且需匹配源码和依赖版本：

```bash
PYTHONPATH=src MUJOCO_GL=egl python -m vision_arm_lab.replay runs/<run>/episode_0000
```

增加 `--window` 并改用 `MUJOCO_GL=glfw` 可观察重放。回放使用记录中的配置、种子和动作，不重新调用策略。MP4 可直接用播放器查看；不保证跨版本逐帧一致。

异常强制终止遗留的临时视频可先列出，再显式删除：

```bash
PYTHONPATH=src python -m vision_arm_lab.maintenance runs/<run>
PYTHONPATH=src python -m vision_arm_lab.maintenance runs/<run> --delete
```

只处理该运行下的录制器临时视频，不自动清理历史结果。

## 测试与契约

```bash
python -m pytest -q
MUJOCO_GL=egl python -m pytest -q -m integration
PYTHONPATH=src MUJOCO_GL=egl python scripts/smoke_sim.py --t03 --seed 0 --image-size 128
```

快速测试覆盖动作边界、观测隔离、定位失败、状态机、任务判定、记录预算、写入故障与视频筛选。MuJoCo 集成测试检查场景参数、开发种子、重置、物理放置后的成功判定、超时和默认零实验文件；直接放置检查不计入抓放成功率。

T03 动作使用世界坐标、米和弧度，每轴平移增量上限 5 mm。策略旋转增量必须为零，固定姿态修正由执行器计算；夹爪 -1 打开、+1 闭合。RGB 原点为左上像素中心；深度为米制光轴深度，四元数顺序 xyzw。

正式场景已关闭离屏 MSAA；旧 M0 冒烟场景保留上游默认抗锯齿，用于复查之前的深度采样偏差。不要用 `python -O` 运行冒烟脚本，它会禁用诊断断言。

## 适用范围

视觉策略只在初始化后定位一次，依赖单个红色方块、已知尺寸、桌面高度、相机标定和固定目标区；不是持续视觉跟踪，也不是多物体或任意姿态方案。数据记录仅提供轻量格式，未实现训练数据集导出。可选键盘控制、GPU 渲染排查与跨分布泛化不属于已通过的核心 MVP 验收。
