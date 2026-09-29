# XML 场景配置

默认场景源文件为 [placement.xml](src/vision_arm_lab/simulation/assets/placement.xml)。文件包含桌面、方块、目标区、相机、灯光、材质和物理/渲染选项；Panda、夹爪与控制器由现有 robosuite 后端组装。它是项目使用的 MJCF 场景源文件，包含项目的资源路径解析约定，需通过本项目 API/CLI 加载。

## 默认调用

原有调用方式保持不变，默认从安装包读取 XML：

```python
from vision_arm_lab import make_environment

with make_environment("configs/mvp.yaml") as env:
    observation = env.reset(seed=0)
```

```bash
MUJOCO_GL=egl vision-arm inspect --config configs/mvp.yaml --seed 0 --steps 20
MUJOCO_GL=egl vision-arm evaluate --config configs/mvp.yaml --policy vision --seed 0
```

## 使用自己的 XML

在项目根目录复制场景和运行配置：

```bash
cp src/vision_arm_lab/simulation/assets/placement.xml configs/my_scene.xml
cp configs/mvp.yaml configs/my_scene.yaml
```

在 `configs/my_scene.yaml` 顶层增加这一行，再编辑 `configs/my_scene.xml`：

```yaml
scene_xml: my_scene.xml
```

`scene_xml` 相对 YAML 文件所在目录解析，也接受绝对路径，与启动命令的工作目录无关。未配置时使用包内默认场景；路径错误会直接报错。

```bash
MUJOCO_GL=egl vision-arm doctor --config configs/my_scene.yaml
MUJOCO_GL=egl vision-arm inspect --config configs/my_scene.yaml --seed 0 --steps 20
```

Python API 的配置路径同样改为 `configs/my_scene.yaml`。读取配置时会保存 XML 快照；修改源文件后需重新创建环境，已有环境的 `reset()` 继续使用原快照。

配置读取与 XML 解析分别位于 `simulation/config.py`、`simulation/scene_xml.py`，资源解析和场景恢复也由仿真层负责。核心层只管理环境生命周期和公共数据，算法只接收公开观测及 TaskInfo 先验。

## 场景创建流程

1. API/CLI 调用 `simulation.factory.make_environment()`，仿真层读取 YAML、解析 XML 与资源路径，并生成场景快照和公开的 `EnvironmentSpec`。
2. 仿真层工厂组装后端、动作适配器及任务评估器，将这些对象注入核心层的 `Environment`。此时尚未创建物理环境。
3. 用户调用 `Environment.reset(seed)`，核心层通过后端接口委托 `RobosuiteBackend` 创建 `CubePlacement`、Panda 和控制器；场景对象与模型资源均留在仿真层。
4. 算法只使用公开的规格、观测和动作接口。桌面高度、方块尺寸及目标区等通过 `TaskInfo` 提供，无须读取 YAML、XML 或接触场景对象。

记录时由仿真配置提供场景快照与资源指纹；回放调用 `simulation.factory.make_recorded_environment()`，由仿真层校验资源并恢复场景，记录层不解释 XML 或组装场景。

## 参数归属

| 内容 | 修改位置 |
| --- | --- |
| 桌面大小、高度、摩擦 | XML 的 `table`、`table_collision`、`table_visual` |
| 方块边长、质量、外观 | XML 的 `cube_g0`、`cube_g0_vis` |
| 目标区位置、大小 | XML 的 `target_region` site |
| 相机位置、朝向、视场角 | XML 对应的 camera，例如 `agentview` |
| 灯光、背景、材质、额外静态物体 | XML 的 `worldbody` 和 `asset` |
| 仿真步长、渲染选项 | XML 的 `option`、`visual` |
| 采样范围、控制频率、图像分辨率、选择哪个相机 | YAML |
| 成功阈值、稳定时长、回合超时、开发种子 | YAML |

MuJoCo 的 box `size` 是**半尺寸**，例如 `0.02 0.02 0.02` 对应边长 4 cm。碰撞与视觉 geom 的尺寸必须一起修改。桌面顶面高度等于 `table.pos.z + table_collision.size.z`；更改桌面高度或厚度时也应同步调整桌腿、`table_top` 和目标区可视化高度。

方块质量由 `cube_g0` 的 `mass` 或 `density` 指定，`mass` 优先；`cube_g0_vis` 保留原有极小视觉质量 `1e-8`。例如在 `cube_g0` 添加 `mass="0.2"` 可设为约 0.2 kg。修改选中相机的 `fovy="55"` 会直接改变视场角。

加载器从 XML 提取尺寸、质量、目标区和相机参数，自动传递给采样器、算法公开先验与任务评估器。YAML 中重复指定这些物理参数会报错，避免两个来源互相覆盖。旧 YAML 迁移时删除这些字段，改为编辑 XML；已加载的 `SceneConfig` 也不允许单独修改派生物理字段。

## 当前任务边界

保留 `table`、`table_collision`、`table_visual`、`cube_main`、`cube_g0`、`cube_g0_vis`、`cube_joint0`、`cube_default_site`、`target_region` 名称。当前任务仍要求水平且 XY 居中的桌面、单个正方体及轴对齐目标区。方块初始姿态由 YAML 中的采样范围决定，reset 会覆盖 XML 中的方块初始位置。

现有 T03 契约仍固定为 500 Hz 物理、20 Hz 控制/相机采样；单独修改 XML timestep 无法切换执行契约。修改方块颜色或尺寸、相机、障碍物等可能影响已有视觉策略和成功率，需要重新验证；本次未扩展为通用多物体任务。

场景使用单个 XML，不支持 `<include>`；几何和任务相关属性需显式填写。自定义纹理/网格的 `file` 路径相对 XML 文件解析，不使用 compiler 的 assetdir/meshdir/texturedir/strippath。`robosuite://textures/...` 指向已安装 robosuite 的 `models/assets/`，无须复制其模型和纹理。第三方场景布局来源与许可见 [ROBOSUITE_LICENSE.txt](src/vision_arm_lab/simulation/assets/ROBOSUITE_LICENSE.txt)。

## 记录与回放

实验元数据保存 XML 快照和场景资源哈希。回放使用记录的快照，外部 XML 文件本身不必保留，但其引用的纹理/网格必须仍可访问且内容匹配。包内 XML 和 Python 源码一样计入源码指纹；源码、依赖或资源版本不匹配时继续按原规则拒绝回放。
