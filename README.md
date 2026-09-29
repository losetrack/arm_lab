# Vision Arm Lab

固定 RGB-D 相机下的 Panda 单方块抓放仿真。提供 Python API 与 `vision-arm` 命令行，支持独立环境生命周期、自定义策略/定位器接入、批量评测、可选记录和动作回放。使用 MuJoCo / robosuite，当前为同步单环境。

算法接口、模块边界及 0.2.0 迁移说明见 [API.md](API.md)，自定义算法示例见 [examples/custom_policy.py](examples/custom_policy.py)。历史开发文档保存在本地被 Git 忽略的 `docs/`，不随仓库分发。

场景模型由 [placement.xml](src/vision_arm_lab/simulation/assets/placement.xml) 定义，编辑与自定义加载方式见 [XML 场景说明](SCENE.md)。

## 项目目录

```text
vision-arm-lab/
├── README.md              # 安装、运行与验证说明
├── API.md                 # 公共接口与算法接入说明
├── SCENE.md               # XML 场景编辑、加载与参数归属
├── pyproject.toml         # Python 包与命令行入口配置
├── requirements-lock.txt  # 已验证的依赖版本
├── src/vision_arm_lab/    # 环境、算法接口、评测与记录实现
│   ├── core/              # 公共契约、配置与通用环境生命周期
│   ├── simulation/        # 仿真后端、控制、组装及 assets/placement.xml 场景源文件
│   ├── algorithms/        # 内置视觉定位和抓放策略
│   ├── tasks/             # 独立的任务成功/失败判定
│   ├── recording/         # 数据记录、源码指纹与动作回放
│   ├── cli/               # 命令行适配与安装诊断
│   └── evaluation.py      # 单回合与批量评测编排
├── configs/               # 场景配置与固定验收种子
├── examples/              # 自定义算法接入示例
├── scripts/               # 环境与动作契约诊断脚本
├── tests/                 # 按对应功能分类的单元与仿真集成测试
├── docs/                  # 本地开发资料，入口为 docs/README.md
│   ├── mvp/               # MVP 方案、参数与阶段验收
│   ├── engineering/       # 环境与接口工程化方案、报告
│   └── roadmap/           # 后续方向讨论
└── runs/                  # 本地实验与验收记录
```

`docs/`、`runs/` 和本地开发工具配置不随 Git 分发。构建产物与 Python/测试缓存由工具生成，已加入忽略规则。

## 环境

本机验证：Ubuntu 26.04、Python 3.10.21、robosuite 1.5.2、MuJoCo 3.3.0。完整版本见 `requirements-lock.txt`。

```bash
# 已有环境
source ~/miniforge3/etc/profile.d/conda.sh
conda activate arm_lab
python -m pip install -r requirements-lock.txt
python -m pip install --no-build-isolation -e .
python -m pip check
vision-arm doctor
```

在新机器上可先用 conda 创建独立环境、指定 `python=3.10.21` 和 pip，再安装锁定依赖和本项目。也可用已有 Python 3.10 的 `python -m venv .venv` 创建隔离环境。锁文件已包含 setuptools/wheel，因此可使用上面的 `--no-build-isolation` 避免额外解析构建版本。Linux 安装部分依赖可能需要系统编译工具；图形运行依赖可用的 EGL/GLFW 系统库。

以下命令默认在项目根目录执行。安装后可在任意目录导入 Python API 或使用 CLI；从其他目录运行时传配置/种子文件的绝对路径。

本机窗口和离屏实际使用 `llvmpipe` 软件渲染；设置 `MUJOCO_GL=egl` 不代表使用 NVIDIA GPU。NVIDIA 硬件渲染尚未验证。受限环境可将 `NUMBA_CACHE_DIR`、`MESA_SHADER_CACHE_DIR` 指向可写临时目录；第三方编译缓存不属于实验记录。

## 运行

先做依赖检查，再显式检查所需渲染模式：

```bash
vision-arm doctor
MUJOCO_GL=egl vision-arm doctor --config configs/mvp.yaml
MUJOCO_GL=glfw vision-arm doctor --config configs/mvp.yaml --window --steps 20
```

`doctor` 不带配置时不创建仿真。带配置时会实际读取 RGB-D 并推进少量控制步，报告实际渲染器；不自动安装驱动或修改系统。CLI 也可通过 `python -m vision_arm_lab` 调用。

统一启动与算法评测：

```bash
MUJOCO_GL=egl vision-arm inspect --config configs/mvp.yaml --seed 0 --steps 100
MUJOCO_GL=egl vision-arm evaluate --config configs/mvp.yaml --policy vision --seed 0
MUJOCO_GL=glfw vision-arm evaluate --config configs/mvp.yaml --policy expert --seed 0 --window
```

Python 中可直接 `evaluate(config, policy_factory=..., seeds=...)`，或用 `make_environment(config)` 自己控制循环，详见 [完整 API 与示例](API.md)。以下旧 runner 命令仍可使用，与新入口共用运行逻辑。

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

后端/环境错误计入 `environment_error`，意外算法错误计入 `policy_error`，不补抽种子；命令存在这些错误时返回 1，配置/启动错误返回 2，中断返回 130。成功率分母包含所有已执行的尝试。

预期算法失败统一计入 `policy_failure`，具体原因保存在 `failure_reason`；`task_status` 单独保存任务状态，算法异常消息不会被当作成功结果。动作中的 NumPy 整数和数值向量在入口统一规范化，开启动作记录不会改变其执行语义。

## 回放与临时文件

只有 `result.json` 中 `recording.actions_complete=true` 的回合可重放，且需匹配源码和依赖版本：

```bash
PYTHONPATH=src MUJOCO_GL=egl python -m vision_arm_lab.replay runs/<run>/episode_0000
# 同一回放的已安装入口
MUJOCO_GL=egl vision-arm replay runs/<run>/episode_0000
```

增加 `--window` 并改用 `MUJOCO_GL=glfw` 可观察重放。回放使用记录中的配置、种子和动作，不重新调用策略。MP4 可直接用播放器查看；不保证跨版本逐帧一致。

异常强制终止遗留的临时视频可先列出，再显式删除：

```bash
PYTHONPATH=src python -m vision_arm_lab.maintenance runs/<run>
PYTHONPATH=src python -m vision_arm_lab.maintenance runs/<run> --delete
```

只处理该运行下的录制器临时视频，不自动清理历史结果。

## 测试与契约

2026-09-29 工程化回归：64 项快速测试、2 项 MuJoCo 集成测试通过；原固定种子 1000–1049 下专家和视觉均为 50/50，逐种子的任务结果、控制步数、仿真完成时间与状态机转换记录和原 MVP 一致。干净环境安装、正式 wheel、窗口/离屏诊断、外部算法接入及短回合记录/回放均已检查。

XML 场景封装后，81 项快速测试和 3 项 MuJoCo 离屏集成测试通过。默认场景的 19 组模型/初始状态/图像数组与封装前完全一致，seed 0 视觉抓放成功；wheel 资源加载及自定义 XML 快照记录/回放均已验证。上述 100 回合成功率来自先前的工程化回归，本次未重复运行。

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
