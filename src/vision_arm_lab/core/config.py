"""Explicit configuration for the confirmed single-scene MVP."""

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import yaml

from vision_arm_lab.core.contracts import ActionSpec, EnvironmentSpec, TaskInfo


@dataclass(frozen=True)
class SimulationConfig:
    table_size_m: list[float]
    table_height_m: float
    friction: list[float]
    cube_side_m: float
    cube_mass_kg: float
    spawn_x_m: list[float]
    spawn_y_m: list[float]
    spawn_yaw_deg: list[float]
    spawn_clearance_m: float
    initial_target_gap_m: float
    target_center_m: list[float]
    target_size_m: list[float]
    grasp_quaternion_xyzw: list[float]
    camera_name: str
    camera_position_m: list[float]
    camera_quaternion_wxyz: list[float]
    camera_fovy_deg: float
    camera_size_px: list[int]
    camera_hz: int
    control_hz: int
    physics_hz: int
    offscreen_samples: int


@dataclass(frozen=True)
class PlacementConfig:
    cube_side_m: float
    table_height_m: float
    target_center_m: tuple[float, float]
    target_size_m: tuple[float, float]
    bottom_tolerance_m: float
    linear_speed_limit_m_s: float
    angular_speed_limit_rad_s: float
    stable_duration_s: float
    episode_timeout_s: float


@dataclass(frozen=True)
class SceneConfig(SimulationConfig):
    bottom_tolerance_m: float
    linear_speed_limit_m_s: float
    angular_speed_limit_rad_s: float
    stable_duration_s: float
    episode_timeout_s: float
    development_seeds: list[int]

    def __post_init__(self):
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
    with Path(path).open() as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, dict):
        raise ValueError('Scene configuration must be a YAML mapping')
    try:
        return SceneConfig(**values)
    except TypeError as exc:
        raise ValueError(f'Invalid scene configuration: {exc}') from exc
