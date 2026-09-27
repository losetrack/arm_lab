"""Explicit configuration for the confirmed single-scene MVP."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SceneConfig:
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
    bottom_tolerance_m: float
    linear_speed_limit_m_s: float
    angular_speed_limit_rad_s: float
    stable_duration_s: float
    episode_timeout_s: float
    development_seeds: list[int]


def load_config(path: str | Path) -> SceneConfig:
    with Path(path).open() as stream:
        config = SceneConfig(**yaml.safe_load(stream))
    if (config.physics_hz, config.control_hz, config.camera_hz) != (500, 20, 20):
        raise ValueError("Only the confirmed 500/20/20 Hz configuration is supported")
    return config
