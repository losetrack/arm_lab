from dataclasses import dataclass
from itertools import product

import numpy as np

from vision_arm_lab.core.contracts import TaskContext, TaskInfo, TaskResult


@dataclass(frozen=True)
class PlacementTaskInfo(TaskInfo):
    camera_name: str
    table_height_m: float
    cube_side_m: float
    target_center_m: tuple[float, float]
    target_size_m: tuple[float, float]


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

    def __post_init__(self):
        for name in ('target_center_m', 'target_size_m'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ('stable_duration_s', 'episode_timeout_s'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('bottom_tolerance_m', 'linear_speed_limit_m_s', 'angular_speed_limit_rad_s'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')


@dataclass(frozen=True)
class PlacementResult(TaskResult):
    @property
    def stable_s(self):
        return self.metrics['stable_s']


class PlacementTask:
    """Public task context and the placement-specific evaluator input adapter."""
    task_id = 'placement'
    version = 1

    def __init__(self, config: PlacementConfig, camera_name: str):
        self.config = config
        self.info = PlacementTaskInfo(
            self.task_id, camera_name, config.table_height_m, config.cube_side_m,
            config.target_center_m, config.target_size_m,
        )
        self.evaluator = PlacementEvaluator(config)

    def reset(self, seed, state, observation):
        self.evaluator.reset()
        return TaskContext(self.task_id, seed,
                           'Place the cube inside the target region and release it.', self.info)

    def update(self, state, observation, actions):
        return self.evaluator.update(state, observation.timestamp_s, actions.actions[0].gripper)


@dataclass(frozen=True)
class ObjectState:
    """Privileged world-frame state for expert/evaluation channels only."""

    position_m: np.ndarray
    rotation: np.ndarray
    linear_velocity_m_s: np.ndarray
    angular_velocity_rad_s: np.ndarray
    gripper_contact: bool


class PlacementEvaluator:
    def __init__(self, config: PlacementConfig):
        self.config = config
        self.local_corners = np.array(list(product((-0.5, 0.5), repeat=3))) * config.cube_side_m
        self.reset()

    def reset(self):
        self.stable_since = None
        self.result = PlacementResult("running", 0.0, {'stable_s': 0.0})

    def update(self, state: ObjectState, elapsed_s: float, gripper_command: int | None) -> TaskResult:
        if self.result.terminated:
            return self.result
        c = self.config
        corners = self.local_corners @ state.rotation.T + state.position_m
        half_size = np.asarray(c.target_size_m) / 2
        in_region = np.all(np.abs(corners[:, :2] - c.target_center_m) <= half_size)
        on_table = abs(float(corners[:, 2].min()) - c.table_height_m) <= c.bottom_tolerance_m
        released = gripper_command == -1 and not state.gripper_contact
        quiet = (
            np.linalg.norm(state.linear_velocity_m_s) <= c.linear_speed_limit_m_s
            and np.linalg.norm(state.angular_velocity_rad_s) <= c.angular_speed_limit_rad_s
        )
        if in_region and on_table and released and quiet:
            if self.stable_since is None:
                self.stable_since = elapsed_s
        else:
            self.stable_since = None
        stable = 0.0 if self.stable_since is None else round(elapsed_s - self.stable_since, 9)
        # Success on the deadline takes precedence over timeout.
        status = "running"
        if stable >= c.stable_duration_s:
            status = "success"
        elif round(elapsed_s, 9) >= c.episode_timeout_s:
            status = "timeout"
        self.result = PlacementResult(status, elapsed_s, {'stable_s': stable})
        return self.result
