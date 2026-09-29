"""Simulator-independent T03 contracts. All physical values use SI units."""

from dataclasses import dataclass
from operator import index
from types import MappingProxyType
from typing import Mapping, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


def _snapshot(value):
    if value is None:
        return None
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CameraObservation:
    """OpenCV axes: x right, y down, z forward; top-left pixel center is (0, 0).

    depth_m is optical-axis depth, not Euclidean range. None means absent.
    camera_to_world maps homogeneous camera coordinates to world coordinates.
    """

    rgb: NDArray[np.uint8] | None
    depth_m: NDArray[np.floating] | None
    intrinsics: NDArray[np.floating]
    camera_to_world: NDArray[np.floating]
    timestamp_s: float

    def __post_init__(self):
        for name in ('rgb', 'depth_m', 'intrinsics', 'camera_to_world'):
            object.__setattr__(self, name, _snapshot(getattr(self, name)))


@dataclass(frozen=True)
class RobotState:
    joint_names: tuple[str, ...]
    joint_position_rad: NDArray[np.floating]
    joint_velocity_rad_s: NDArray[np.floating]
    eef_position_m: NDArray[np.floating]
    eef_quaternion_xyzw: NDArray[np.floating]
    gripper_position_m: NDArray[np.floating]

    def __post_init__(self):
        object.__setattr__(self, 'joint_names', tuple(self.joint_names))
        for name in ('joint_position_rad', 'joint_velocity_rad_s', 'eef_position_m',
                     'eef_quaternion_xyzw', 'gripper_position_m'):
            object.__setattr__(self, name, _snapshot(getattr(self, name)))


@dataclass(frozen=True)
class Observation:
    cameras: Mapping[str, CameraObservation]
    robot: RobotState
    timestamp_s: float
    language_instruction: str | None = None

    def __post_init__(self):
        object.__setattr__(self, 'cameras', MappingProxyType(dict(self.cameras)))


@dataclass(frozen=True)
class ActionSpec:
    """Only this mode is implemented; gripper -1 opens and +1 closes.

    XYZ is an increment from achieved end-effector position, not a velocity.
    The orientation target stays fixed; policy rotation increments must be zero.
    Joint order is empty because actions do not command joints.
    """

    mode: str = "eef_delta_pose_fixed_orientation"
    frame: str = "world"
    translation_unit: str = "m"
    rotation_unit: str = "rad"
    joint_names: tuple[str, ...] = ()
    interval_s: float = 0.05
    max_translation_per_axis_m: float = 0.005
    gripper_values: tuple[int, int] = (-1, 1)


@dataclass(frozen=True)
class Action:
    delta_position_m: NDArray[np.floating]
    delta_rotation_rad: NDArray[np.floating]
    gripper: int

    def __post_init__(self):
        # Control, callbacks and recording consume the same owned numeric data.
        # Physical limits are still checked by the action adapter before step.
        try:
            for name in ('delta_position_m', 'delta_rotation_rad'):
                value = np.array(getattr(self, name), dtype=float, copy=True)
                value.setflags(write=False)
                object.__setattr__(self, name, value)
            object.__setattr__(self, 'gripper', index(self.gripper))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ActionValidationError('Action requires numeric vectors and an integer gripper') from exc


@dataclass(frozen=True)
class ActionChunk:
    actions: tuple[Action, ...]
    spec: ActionSpec

    def __post_init__(self):
        object.__setattr__(self, 'actions', tuple(self.actions))


class Policy(Protocol):
    action_spec: ActionSpec
    required_inputs: frozenset[str]

    def reset(self) -> None: ...

    def act(self, observation: Observation) -> ActionChunk: ...


class PolicyFailure(RuntimeError):
    """Expected algorithm failure; its message is a reason, never a task status."""


class ActionValidationError(ValueError):
    """An action rejected before any physics step."""


@dataclass(frozen=True)
class TaskInfo:
    """Declared placement priors; contains no sampled object pose."""
    camera_name: str
    table_height_m: float
    cube_side_m: float
    target_center_m: tuple[float, float]
    target_size_m: tuple[float, float]


@dataclass(frozen=True)
class EnvironmentSpec:
    action_spec: ActionSpec
    camera_names: tuple[str, ...]
    camera_size_px: tuple[int, int]
    joint_names: tuple[str, ...]
    capabilities: frozenset[str]
    task: TaskInfo
    episode_timeout_s: float


@dataclass(frozen=True)
class TaskResult:
    status: str
    elapsed_s: float
    stable_s: float

    @property
    def terminated(self) -> bool:
        return self.status != 'running'


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    task_result: TaskResult

    @property
    def done(self) -> bool:
        return self.task_result.terminated


@dataclass(frozen=True)
class Transition:
    step: int
    observation_before: Observation
    action: ActionChunk
    observation_after: Observation
    task_result: TaskResult


@runtime_checkable
class PolicyDiagnostics(Protocol):
    def diagnostics(self) -> Mapping: ...
