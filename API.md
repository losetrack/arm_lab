# 环境与算法开发接口

适用于 0.2.0。安装和启动见 [README](README.md)，外部算法示例见 [examples](examples/custom_policy.py)。当前支持同步单环境、Panda 单方块任务和 T03；公共 API 从 `vision_arm_lab` 导入。

## 创建与驱动环境

```python
import numpy as np
from vision_arm_lab import make_environment, Action, ActionChunk

with make_environment("configs/mvp.yaml", render_mode="offscreen") as env:
    obs = env.reset(seed=0)
    for _ in range(20):
        camera = obs.cameras[env.spec.task.camera_name]
        rgb, depth = camera.rgb, camera.depth_m
        eef_position = obs.robot.eef_position_m
        # 保持末端位置和固定姿态，并打开夹爪。
        command = ActionChunk(
            (Action(np.zeros(3), np.zeros(3), -1),), env.spec.action_spec,
        )
        step = env.step(command)
        obs = step.observation
        if step.done:
            break
```

`config` 接受 YAML 路径或 `configuration.EnvironmentConfig`。相对路径以调用进程当前工作目录为基准；从其他目录运行时传绝对路径。工厂组装组件，但直到 `reset` 才创建物理环境和渲染上下文。

配置中的数值向量和开发种子在加载时转换为不可变元组；YAML/JSON 中仍使用数组。修改运行参数时可用 `dataclasses.replace(config.simulation, ...)` 或 `dataclasses.replace(config.task, ...)` 创建配置，再用 `dataclasses.replace(config, simulation=..., task=...)` 组合并校验，修改派生的物理参数仍需编辑 XML 后重新加载。配置视图与公开规格不会受原始输入列表的后续修改影响。

| 能力 | 语义 |
| --- | --- |
| `env.spec` | 动作规格、相机名/分辨率、机器人关节顺序、传感器能力、任务先验和时限 |
| `reset(seed=0)` | 返回初始 Observation；重建物理环境并重置任务、控制器和随机源 |
| `step(ActionChunk)` | 执行一个控制周期，返回 StepResult；其 `done` 来自任务终止状态 |
| `env.observation` | 最近一次 reset/step 的观测；关闭后为 None |
| `render()` | 仅 window 模式使用，更新窗口但不推进物理；窗口循环由调用者决定是否等待 |
| `env.diagnostics` | 后端和实际 GL 渲染器信息；reset 前渲染器为 None |
| `close()` / `with` | 关闭资源；重复关闭允许；close 后需 reset 才能再次 step |

首次 step 前必须 reset，任务终止后也必须 reset。非法动作在推进物理前拒绝，调用者可以修正；物理/任务执行出现意外异常时关闭环境，再次使用需 reset。建议始终用 `with` 管理用户代码异常。

`render_mode` 选择是否显示窗口；图形驱动由进程启动前的 `MUJOCO_GL` 配置决定。离屏示例使用 `MUJOCO_GL=egl`，窗口示例使用 `MUJOCO_GL=glfw`。更换图形驱动应启动新进程，不能在已导入仿真库后切换。

## 观测和动作

Observation 是一次采样快照，读取多个字段不会推进仿真。数组与仿真内部缓冲区隔离，默认只读；修改图像做预处理时先 `.copy()`。相机映射也不可直接修改。

| 字段 | 形状 / 语义 |
| --- | --- |
| `obs.cameras[name].rgb` | `(H, W, 3)`、uint8、RGB；左上像素中心 `(0, 0)` |
| `.depth_m` | `(H, W)`，米制光轴深度；不是到相机的欧氏距离 |
| `.intrinsics` | `(3, 3)` 相机内参 K |
| `.camera_to_world` | `(4, 4)`，将 OpenCV 相机坐标变换至世界坐标 |
| `.timestamp_s` | 相机采样的仿真时间 |
| `obs.robot.joint_names` | 关节顺序；Panda 为 `robot0_joint1` 至 `robot0_joint7` |
| `.joint_position_rad` / `.joint_velocity_rad_s` | `(7,)`，弧度 / 弧度每秒 |
| `.eef_position_m` / `.eef_quaternion_xyzw` | `(3,)` 世界坐标位置 / `(4,)` xyzw 四元数，同一末端控制 site |
| `.gripper_position_m` | `(2,)` 两指关节位置，米 |
| `obs.timestamp_s` | 本次环境观测的仿真时间 |
| `obs.task_context` | TaskContext：任务标识、种子、该回合公开目标和语言指令 |
| `obs.language_instruction` | 与 task_context 中的指令一致，供语言条件策略使用 |

RGB/深度的契约允许 None 表示缺失；当前后端提供两者。算法声明的输入若不受支持或首次观测缺失，评测会明确报告。当前每个控制步更新一份 20 Hz 的 RGB-D/机器人观测，物理仿真为 500 Hz。

当前 placement 的公开先验类型为 `tasks.placement.PlacementTaskInfo`，在 `env.spec.task` 中提供：`camera_name`、`table_height_m`、`cube_side_m`、`target_center_m`、`target_size_m`。其中没有随机采样的方块位姿、接触状态或其他真值。

T03 动作保持不变：

- `delta_position_m`：世界坐标 XYZ 增量，每轴绝对值≤0.005 m；增量相对已达到的末端位置，不是速度。
- `delta_rotation_rad`：三维全零。固定俯视姿态由适配器维持。
- `gripper`：-1 打开，+1 闭合。
- 每个 ActionChunk 恰有一个 Action，规格必须与 `env.spec.action_spec` 相同；控制周期 0.05 秒。
- 越界、非有限值、错误形状和不支持的规格明确报错，不自动裁剪。

`Action` 创建时将列表或 NumPy 向量复制为只读浮点数组，将 Python/NumPy 整数夹爪值统一为 Python `int`；浮点夹爪值不会被截断成整数。控制、回调与记录使用同一份动作数据。数值类型错误在动作构造时报告，物理范围和规格仍在 step 前校验。

## 接入完整策略

实现三个成员：`action_spec`、`required_inputs`、`reset()/act(observation)`。最小示例见 [custom_policy.py](examples/custom_policy.py)；无需继承项目基类，也不需要 `phase` 或 `events`。

```python
from vision_arm_lab import evaluate
from my_algorithm import make_policy

report = evaluate(
    "configs/mvp.yaml",
    policy_factory=make_policy,  # make_policy(EnvironmentSpec) -> Policy
    seeds=[0, 1, 2], steps=100, record="off",
)
print(report.summary)
```

工厂在一次 evaluate 中构造一个策略，每个回合调用一次 `reset()`。算法应在 reset 中清空回合状态；自己的模型、参数和随机数管理由算法负责。工厂只接收公开规格，不接收后端。

`required_inputs` 是传感器名称的 frozenset，可使用 `rgb`、`depth`、`calibration`、`robot_state`、`language`。构造时读取 TaskInfo 所需先验；不要在这个集合里混入内部定位模块或不受支持的真值字段。

预期算法失败可以 `raise PolicyFailure("localization_failed")`，回合 `status` 固定为 `policy_failure`，消息写入 `failure_reason`，不会被解释为任务状态。意外算法异常记录为 `policy_error`；后端异常为 `environment_error`；中断为 `interrupted`。不会替换失败种子或自动重试。

可选实现 `diagnostics() -> Mapping` 返回 JSON 可序列化诊断。通用运行器不依赖诊断内容；内置抓放策略在这里报告状态机阶段。命令行 `--verbose` 在诊断变化时输出。

## 只替换视觉定位器

`locator_factory(PlacementTaskInfo)` 返回一个可调用定位器，接受 Observation，返回世界坐标下的三维方块中心；声明其需要的 `required_inputs`。调用：

```python
report = evaluate(
    "configs/mvp.yaml", locator_factory=make_locator, seeds=[0, 1],
)
```

该入口复用当前 GraspPolicy，仍在初始等待后定位一次。完整策略接入点可自行决定何时使用视觉数据。内置专家由单独组装路径注入真值读取器，其结果明确标注真值来源；普通自定义工厂不获得这个接口。

`policy="expert"/"vision"/"inspect"`、`policy_factory`、`locator_factory` 三选一；都未传时默认 vision。不要给自定义算法同时传入内置策略名。

## 命令行接入同一个算法

从仓库根目录运行：

```bash
PYTHONPATH=examples MUJOCO_GL=egl vision-arm evaluate \
  --config configs/mvp.yaml --policy-factory custom_policy:make_policy --seed 0 --steps 20

PYTHONPATH=examples MUJOCO_GL=egl vision-arm evaluate \
  --config configs/mvp.yaml --locator-factory custom_policy:make_locator --seed 0
```

自定义模块需要安装到当前 Python 环境，或将其所在目录加入 PYTHONPATH。这里只导入用户显式指定的 `MODULE:CALLABLE`，没有插件扫描或动态注册服务。示例 HoldPosition 只保持位置，短回合结果应为 step_limit，不代表抓放成功。

## 评测结果与记录

`evaluate` 返回 EvaluationReport：

- `episodes`：逐回合字典，包含种子、执行状态 `status`、任务状态 `task_status`、实际执行步数、仿真/墙钟时间、信息来源、算法诊断和环境诊断；预期算法失败含 `failure_reason`，意外异常含阶段、类型和消息。
- `summary`：已尝试回合数、成功数/率、成功平均仿真时间、结果分类、记录模式、警告与预算。
- `records`：开启记录时的输出目录；off 为 None。

任务成功规则沿用 MVP。达到运行器 `steps` 上限时，普通算法返回 step_limit；inspect 返回 running。任务超时与步数上限不是成功。Python API 返回失败结果供调用者处理，不退出进程；CLI 遇算法/环境错误返回 1，配置/启动错误返回 2，中断返回 130，已完成的任务成败统计通常返回 0。

`task_status` 来自任务评估器的最后一次结果；reset 成功后为 running，reset 失败时为 None。成功只来自任务评估器，`PolicyFailure("success")` 也会记录为算法失败，不能提高成功率。原先按 `status == "localization_failed"` 等失败原因筛选的代码，需改为检查 `status == "policy_failure"` 和 `failure_reason`。

默认 off 不创建实验文件。需要完整记录选项时传 `RecordOptions`：

```python
from vision_arm_lab import RecordOptions

options = RecordOptions(mode="debug", output="runs", actions=True,
                        observations=True, observation_hz=20)
report = evaluate("configs/mvp.yaml", policy="vision", seeds=[0], record=options)
```

新增的 `run_episode(env, policy, seed, steps, on_transition=...)` 可用于自行组合环境与消费者。Transition 含步号、执行前观测、动作、执行后观测和任务结果，第一步的执行前观测就是 reset 返回值。该函数不拥有传入环境的生命周期，调用者负责 `with`/close；外部回调异常原样传播。

本轮记录格式为 v2：元数据新增任务标识、规则版本、独立任务配置；回合结果新增 task_id、task_context 和 task_metrics。动作与执行后观测的时序沿用原格式，初始观测不单独保存，仍属于调试记录。v1 记录需要匹配的旧代码回放，不自动猜测任务或迁移。自定义算法的参数、代码版本等可通过 `evaluate(..., policy_metadata={...})` 显式记录；自动源码指纹覆盖项目包，不自动扫描算法的外部依赖代码。

## 内部边界与迁移

| 模块 | 职责 |
| --- | --- |
| `core/` | `contracts.py` 定义公共数据、动作规范化、单位和时序；`environment.py` 管理通用生命周期，不解析 YAML/XML 或导入仿真模块 |
| `simulation/` | `config.py` 校验物理配置和恢复场景快照；`scene_xml.py` 读取 XML 与资源，`placement_xml.py` 提取放置几何；`factory.py` 创建后端与私有状态读取器；`scene.py`、`robosuite.py`、`control.py` 实现场景、物理、传感器与控制；`placement_state.py` 读取私有真值 |
| `algorithms/` | `perception.py` 实现颜色定位；`policies.py` 实现保持与抓放策略，通过公共观测和动作契约工作 |
| `tasks/` | `placement.py` 定义 PlacementTask、PlacementTaskInfo、PlacementConfig、PlacementResult；任务拥有指令、公开上下文和判定规则，不导入仿真器 |
| `application.py` | 公共 make_environment / evaluate 入口及回放组装，选择任务、算法并连接组件 |
| `configuration.py` / `task_registry.py` | 加载和拆分运行配置；显式配方连接纯任务与对应仿真适配，校验任务版本及共享物理参数 |
| `evaluation.py` | 单回合及批量执行、失败分类、成功率等指标；消费已组装组件，不导入仿真、算法实现或记录器实现 |
| `recording/` | `recorder.py` 消费 Transition、管理文件预算、保存调用方提供的结果与指标；`provenance.py` 生成版本指纹；`replay.py` 校验并重放完整动作记录 |
| `cli/` | 参数解析、用户指定模块导入、安装诊断、终端输出与退出码；复用公共 API |
| 根目录命令模块 | `__main__.py`、`runner.py`、`replay.py`、`maintenance.py` 保留既有命令入口，转交 `cli/` 执行 |

配置入口仍是 YAML；物理场景参数来自包内 `simulation/assets/placement.xml`，或 YAML 的 `scene_xml` 指定的文件。XML 路径相对 YAML 解析，加载时保存快照并从中提取尺寸、目标区和相机参数；详见 [XML 场景说明](SCENE.md)。内部按 SimulationConfig、PlacementConfig 和公开 TaskInfo 分配配置，避免算法持有完整后端配置。公开导出是稳定调用入口，内部模块构造函数不承诺兼容。

测试按相同功能归档在 `tests/` 的对应子目录。通用环境只通过协议组合后端和任务判定；算法不导入具体仿真模块。场景和后端的组装集中在 `simulation/factory.py`，算法选择及环境、算法、记录器之间的连接集中在 `application.py`。专家定位能力由仿真层的专用适配器提供，普通算法工厂只收到公开规格。

评测层负责执行步数、任务结果和汇总指标；记录器保留这些字段，仅补充记录状态。动作记录不依赖相机；当前图像和视频记录仍使用观测中的第一台相机，未扩展多相机选择接口。

任务和场景元数据由 `EnvironmentConfig.record_metadata()` 汇集；回放通过 `application.make_recorded_environment()` 选择相同配方，其中 XML 和资源恢复继续委托仿真层。算法只接收公开 TaskInfo / TaskContext，不持有运行配置。

目录整理后，内部导入路径随功能迁移，例如 `vision_arm_lab.perception` 改为 `vision_arm_lab.algorithms.perception`。场景配置与 XML 解析进一步从 `core.config`、`core.scene_xml` 移到 `simulation.config`、`simulation.scene_xml`。外部代码若直接导入旧内部模块，需更新路径。顶层公共导出和现有命令参数保持不变。

本轮 `evaluate` 实现从 `evaluation.py` 移到 `application.py`；外部调用推荐继续使用 `from vision_arm_lab import evaluate`。内部 `Recorder.finish` 接收已计算的指标字典。任务独立阶段新增 v2 记录格式。

从 MVP 迁移需要注意：

- 原 `python -m vision_arm_lab.runner` 命令保留，内部转到公共 evaluate；其默认策略仍为 inspect。新 `vision-arm evaluate` 默认 vision。
- 原直接构造 RobosuiteBackend 的调用改用 `make_environment`；`step` 返回 StepResult，使用 `.observation/.task_result/.done`。
- `final_phase/phases` 移入结果的 `policy_diagnostics`；GL 信息在 `env.diagnostics['gl_renderer']`；新算法错误单独标记 policy_error。
- 观测数组变为只读，修改前先复制。
- 回放复用公共环境工厂。源码指纹现在以包内相对路径计算，正式安装包也可记录；旧版本记录按原有严格匹配规则被拒绝，若需重放应在匹配的旧代码/依赖环境执行。


## 任务独立与扩展

现有配置可选增加 `task_id: placement`；未指定时明确使用原 placement。当前仅注册这一项，未知任务启动时报错。`vision`、`expert` 和 `locator_factory` 限用于 placement；新任务可使用完整 `policy_factory`，`inspect` 仍只保持机器人状态。

核心 Task 协议只包含 `info`、`reset(seed, state, observation) -> TaskContext`、`update(state, observation, actions) -> TaskResult`。state 由仿真适配读取且只传入任务；reset 在后端完成重置后运行，生成当前回合的公开目标。核心每次返回观测时附加同一回合上下文，不解释夹爪、方块或目标区。

上下文目标使用任务自己的不可变 TaskInfo 子类，其字段应能转换为 JSON 数据；任务指标使用名称到数值的映射。上下文在回合内固定，在线修改目标与 VLA 动作序列失效规则不在本阶段范围内。

`TaskInfo` 现在是只含 task_id 的通用基类；原方块字段移到 PlacementTaskInfo。`TaskResult` 包含 status、elapsed_s 和只读 metrics；放置任务返回 PlacementResult，仍可读取 `.stable_s`，也可通过 `metrics['stable_s']` 获取。直接构造旧 TaskInfo/TaskResult 或 Environment 的代码需要适配新任务类型和 Task 协议，顶层 make_environment / evaluate 的调用签名保持不变。

新增任务需要：在 tasks 定义类型明确的规则与公开先验；在 simulation 提供必要的场景加载、后端和状态读取；在应用层 task_registry.RECIPES 添加 TaskRecipe。配方显式声明任务配置类型、来自场景的物理字段、场景加载/快照恢复/元数据、后端构造和公开规格构造。任务规则不导入 XML，任务标识不进入核心或通用评测循环。

`configuration.load_config` 取代原 `simulation.config.load_config`；内部配置现在通过 `.simulation`、`.task` 和 `.development_seeds` 访问。既有平铺 YAML 保持可读，未知字段仍报错；内部已经分离配置职责，后续需要嵌套 YAML 时另行明确迁移。
