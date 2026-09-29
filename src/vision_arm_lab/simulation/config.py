"""Explicit configuration for the confirmed single-scene MVP."""

from dataclasses import asdict, dataclass, fields

import numpy as np

from vision_arm_lab.simulation.scene_xml import read_scene_xml, scene_asset_fingerprint
from vision_arm_lab.simulation.placement_xml import scene_parameters


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
        for name in ('cube_side_m', 'cube_mass_kg', 'table_height_m', 'camera_fovy_deg'):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('spawn_clearance_m', 'initial_target_gap_m'):
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


def load_simulation_config(values, directory):
    values = dict(values)
    scene_path = values.pop('scene_xml', None)
    if scene_path is not None and not isinstance(scene_path, str):
        raise ValueError('scene_xml must be a file path relative to the YAML file')
    snapshot = read_scene_xml(directory / scene_path if scene_path is not None else None)
    if 'camera_name' not in values:
        raise ValueError('Scene configuration requires camera_name')
    parameters = scene_parameters(snapshot, values['camera_name'])
    duplicates = values.keys() & parameters.keys()
    if duplicates:
        raise ValueError(f'Physical parameters belong in scene XML, not YAML: {sorted(duplicates)}')
    values.update(parameters, scene_xml=snapshot)
    return SimulationConfig(**values)


def simulation_metadata(config):
    return {'scene': asdict(config),
            'scene_asset_sha256': scene_asset_fingerprint(config.scene_xml)}


def restore_simulation_config(metadata):
    snapshot = metadata['scene']
    if metadata['scene_asset_sha256'] != scene_asset_fingerprint(snapshot['scene_xml']):
        raise ValueError('Replay requires the recorded scene asset versions')
    return SimulationConfig(**snapshot)
