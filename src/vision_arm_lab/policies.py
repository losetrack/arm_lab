"""Shared grasp state machine; localization is the expert/vision difference."""
from dataclasses import dataclass
import numpy as np
from vision_arm_lab.contracts import Action, ActionChunk, ActionSpec


class PolicyFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class GraspConfig:
    settle_s: float = 1.0
    hover_m: float = 0.12
    reach_m: float = 0.003
    close_s: float = 1.0
    release_s: float = 0.5
    release_clearance_m: float = 0.005
    minimum_grasp_gap_m: float = 0.01
    open_gap_m: float = 0.065
    approach_timeout_s: float = 15.0
    motion_timeout_s: float = 10.0
    transfer_timeout_s: float = 15.0
    gripper_timeout_s: float = 2.0
    wait_timeout_s: float = 3.0


class GraspPolicy:
    action_spec = ActionSpec()
    required_inputs = frozenset({"robot_state", "localization"})
    phases = ("settle", "approach", "descend", "close", "lift", "transfer", "lower", "release", "retreat", "wait")

    def __init__(self, scene, locate, config=GraspConfig()):
        self.scene = scene
        self.locate = locate
        self.config = config
        self.reset()

    def reset(self):
        self.phase = "settle"
        self.started = 0.0
        self.target = None
        self.events = [{"phase": self.phase, "time_s": 0.0}]

    def _advance(self, time):
        self.phase = self.phases[self.phases.index(self.phase) + 1]
        self.started = time
        self.events.append({"phase": self.phase, "time_s": round(time, 6)})

    def act(self, observation):
        p = observation.robot.eef_position_m
        time = observation.timestamp_s
        elapsed = time - self.started
        c = self.config
        gap = float(np.abs(observation.robot.gripper_position_m).sum())
        delta = np.zeros(3)
        gripper = 1 if self.phase in ("close", "lift", "transfer", "lower") else -1
        if self.phase == "settle":
            if elapsed >= c.settle_s:
                self.target = np.array(self.locate(observation), dtype=float)
                if self.target.shape != (3,) or not np.isfinite(self.target).all():
                    raise PolicyFailure("localization_failed")
                self._advance(time)
        elif self.phase in ("close", "release"):
            duration = c.close_s if self.phase == "close" else c.release_s
            if elapsed >= duration:
                if self.phase == "close":
                    if not c.minimum_grasp_gap_m <= gap < c.open_gap_m:
                        raise PolicyFailure("grasp_failed")
                    self._advance(time)
                elif gap >= c.open_gap_m:
                    self._advance(time)
            if elapsed > c.gripper_timeout_s:
                raise PolicyFailure("gripper_timeout")
        elif self.phase == "wait":
            if elapsed >= c.wait_timeout_s:
                raise PolicyFailure("not_in_target")
        else:
            target = self.target.copy()
            if self.phase in ("approach", "lift", "transfer", "retreat"):
                target[2] += c.hover_m
            if self.phase in ("transfer", "lower", "retreat"):
                target[:2] = self.scene.target_center_m
            if self.phase == "lower":
                target[2] = self.scene.table_height_m + self.scene.cube_side_m / 2 + c.release_clearance_m
            difference = target - p
            delta = np.clip(difference, -self.action_spec.max_translation_per_axis_m, self.action_spec.max_translation_per_axis_m)
            timeout = c.motion_timeout_s
            if self.phase == "approach":
                timeout = c.approach_timeout_s
            elif self.phase == "transfer":
                timeout = c.transfer_timeout_s
            if np.linalg.norm(difference) <= c.reach_m:
                self._advance(time)
            elif elapsed > timeout:
                raise PolicyFailure("stage_timeout")
        return ActionChunk((Action(delta, np.zeros(3), gripper),), self.action_spec)
