from dataclasses import dataclass
from itertools import product

import numpy as np

from vision_arm_lab.config import SceneConfig


@dataclass(frozen=True)
class ObjectState:
    """Privileged world-frame state for expert/evaluation channels only."""

    position_m: np.ndarray
    rotation: np.ndarray
    linear_velocity_m_s: np.ndarray
    angular_velocity_rad_s: np.ndarray
    gripper_contact: bool


@dataclass(frozen=True)
class TaskResult:
    status: str
    elapsed_s: float
    stable_s: float

    @property
    def terminated(self) -> bool:
        return self.status != "running"


class PlacementEvaluator:
    def __init__(self, config: SceneConfig):
        self.config = config
        self.local_corners = np.array(list(product((-0.5, 0.5), repeat=3))) * config.cube_side_m
        self.reset()

    def reset(self):
        self.stable_since = None
        self.result = TaskResult("running", 0.0, 0.0)

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
        self.result = TaskResult(status, elapsed_s, stable)
        return self.result
