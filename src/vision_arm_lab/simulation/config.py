"""Explicit configuration for the confirmed single-scene MVP."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import yaml

from vision_arm_lab.core.contracts import ActionSpec, EnvironmentSpec, TaskInfo
from vision_arm_lab.simulation.scene_xml import read_scene_xml, scene_parameters, scene_asset_fingerprint
from vision_arm_lab.tasks.placement import PlacementConfig


@dataclass(frozen=True)
class SimulationConfig:
    scene_xml: str
    table_size_m: tuple[float, ...]
    table_height_m: float
    friction: tuple[float, ...]
    cube_side_m: float
    cube_mass_kg: float
    spawn_x_m: tuple[float, ...]
    spawn_y_m: tuple[float, ...]
    spawn_yaw_deg: tuple[float, ...]
    spawn_clearance_m: float
    initial_target_gap_m: float
    target_center_m: tuple[float, ...]
    target_size_m: tuple[float, ...]
    grasp_quaternion_xyzw: tuple[float, ...]
    camera_name: str
    camera_position_m: tuple[float, ...]
    camera_quaternion_wxyz: tuple[float, ...]
    camera_fovy_deg: float
    camera_size_px: tuple[int, ...]
    camera_hz: int
    control_hz: int
    physics_hz: int
    offscreen_samples: int

    def __post_init__(self):
        # Own immutable vectors even when constructed from YAML/JSON lists.
        for field in fields(self):
            if field.type in (tuple[float, ...], tuple[int, ...]):
                object.__setattr__(self, field.name, tuple(getattr(self, field.name)))


@dataclass(frozen=True)
class SceneConfig(SimulationConfig):
    bottom_tolerance_m: float
    linear_speed_limit_m_s: float
    angular_speed_limit_rad_s: float
    stable_duration_s: float
    episode_timeout_s: float
    development_seeds: tuple[int, ...]

    def __post_init__(self):
        super().__post_init__()
        if (self.physics_hz, self.control_hz, self.camera_hz) != (500, 20, 20):
            raise ValueError('Only the confirmed 500/20/20 Hz configuration is supported')
        shapes = {
            'table_size_m': 3, 'friction': 3, 'spawn_x_m': 2, 'spawn_y_m': 2,
            'spawn_yaw_deg': 2, 'target_center_m': 2, 'target_size_m': 2,
            'grasp_quaternion_xyzw': 4, 'camera_position_m': 3,
            'camera_quaternion_wxyz': 4, 'camera_size_px': 2,
        }
        for name, size in shapes.items():
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f'{name} must contain {size} finite values')
        for name in ('cube_side_m', 'cube_mass_kg', 'table_height_m', 'camera_fovy_deg',
                     'stable_duration_s', 'episode_timeout_s'):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('bottom_tolerance_m', 'linear_speed_limit_m_s',
                     'angular_speed_limit_rad_s', 'spawn_clearance_m', 'initial_target_gap_m'):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        for name in ('table_size_m', 'target_size_m'):
            if any(value <= 0 for value in getattr(self, name)):
                raise ValueError(f'{name} must be positive')
        for name in ('spawn_x_m', 'spawn_y_m', 'spawn_yaw_deg'):
            low, high = getattr(self, name)
            if low > high:
                raise ValueError(f'{name} must be ordered low to high')
        if any(type(value) is not int or value <= 0 for value in self.camera_size_px):
            raise ValueError('camera_size_px must contain positive integers')
        if not self.camera_name or not 0 < self.camera_fovy_deg < 180:
            raise ValueError('camera name and field of view are invalid')
        if any(value < 0 for value in self.friction):
            raise ValueError('friction must be nonnegative')
        if type(self.offscreen_samples) is not int or self.offscreen_samples < 0:
            raise ValueError('offscreen_samples must be a nonnegative integer')
        for name in ('grasp_quaternion_xyzw', 'camera_quaternion_wxyz'):
            if np.linalg.norm(getattr(self, name)) == 0:
                raise ValueError(f'{name} must be nonzero')
        for name, value in scene_parameters(self.scene_xml, self.camera_name).items():
            if not np.array_equal(np.asarray(getattr(self, name)), np.asarray(value)):
                raise ValueError(f'{name} must match scene XML; modify XML instead of derived fields')

    def record_metadata(self):
        """Capture the scene and its resources without exposing XML to consumers."""
        return {'scene': asdict(self),
                'scene_asset_sha256': scene_asset_fingerprint(self.scene_xml)}

    @property
    def simulation(self) -> SimulationConfig:
        return SimulationConfig(**{f.name: getattr(self, f.name) for f in fields(SimulationConfig)})

    @property
    def placement(self) -> PlacementConfig:
        values = {f.name: getattr(self, f.name) for f in fields(PlacementConfig)}
        values['target_center_m'] = tuple(self.target_center_m)
        values['target_size_m'] = tuple(self.target_size_m)
        return PlacementConfig(**values)

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            action_spec=ActionSpec(), camera_names=(self.camera_name,),
            camera_size_px=tuple(self.camera_size_px),
            joint_names=tuple(f'robot0_joint{i}' for i in range(1, 8)),
            capabilities=frozenset({'rgb', 'depth', 'calibration', 'robot_state'}),
            task=TaskInfo(self.camera_name, self.table_height_m, self.cube_side_m,
                          tuple(self.target_center_m), tuple(self.target_size_m)),
            episode_timeout_s=self.episode_timeout_s,
        )


def load_config(path: str | Path | SceneConfig) -> SceneConfig:
    if isinstance(path, SceneConfig):
        return path
    path = Path(path)
    with path.open() as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, dict):
        raise ValueError('Scene configuration must be a YAML mapping')
    scene_path = values.pop('scene_xml', None)
    if scene_path is not None and not isinstance(scene_path, str):
        raise ValueError('scene_xml must be a file path relative to the YAML file')
    snapshot = read_scene_xml(path.parent / scene_path if scene_path is not None else None)
    if 'camera_name' not in values:
        raise ValueError('Scene configuration requires camera_name')
    parameters = scene_parameters(snapshot, values['camera_name'])
    duplicates = values.keys() & parameters.keys()
    if duplicates:
        raise ValueError(f'Physical parameters belong in scene XML, not YAML: {sorted(duplicates)}')
    values.update(parameters, scene_xml=snapshot)
    try:
        return SceneConfig(**values)
    except TypeError as exc:
        raise ValueError(f'Invalid scene configuration: {exc}') from exc
