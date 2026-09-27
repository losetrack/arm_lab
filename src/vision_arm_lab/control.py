"""Translate the approved T03 action into a Panda OSC_POSE command."""

import numpy as np
from scipy.spatial.transform import Rotation

from vision_arm_lab.contracts import ActionChunk, ActionSpec


class PandaActionAdapter:
    def __init__(self, target_quaternion_xyzw):
        # Supplied by scene configuration, never inferred from an action.
        self.target = Rotation.from_quat(target_quaternion_xyzw)
        self.spec = ActionSpec()

    def encode(self, chunk: ActionChunk, current_quaternion_xyzw) -> np.ndarray:
        if chunk.spec != self.spec:
            raise ValueError("Unsupported action specification")
        if len(chunk.actions) != 1:
            raise ValueError("MVP execution requires H=1")
        action = chunk.actions[0]
        delta = np.asarray(action.delta_position_m, dtype=float)
        rotation = np.asarray(action.delta_rotation_rad, dtype=float)
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise ValueError("Translation must contain three finite values")
        if np.any(np.abs(delta) > self.spec.max_translation_per_axis_m):
            raise ValueError("Translation exceeds the 5 mm per-axis limit")
        if rotation.shape != (3,) or not np.all(rotation == 0):
            raise ValueError("Fixed-orientation mode requires zero rotation increments")
        if action.gripper not in self.spec.gripper_values:
            raise ValueError("Gripper must be -1 (open) or +1 (close)")
        current = Rotation.from_quat(current_quaternion_xyzw)
        error = (self.target * current.inv()).as_rotvec()
        # Upstream OSC_POSE scales normalized inputs by 0.05 m / 0.5 rad.
        if np.any(np.abs(error) > 0.5):
            raise ValueError("Orientation correction exceeds the OSC input range")
        return np.concatenate((delta / 0.05, error / 0.5, [action.gripper]))
