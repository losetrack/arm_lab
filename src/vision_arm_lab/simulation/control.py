"""Translate the approved T03 action into a Panda OSC_POSE command."""

import numpy as np
from scipy.spatial.transform import Rotation

from vision_arm_lab.core.contracts import Action, ActionChunk, ActionSpec, ActionValidationError


class PandaActionAdapter:
    def __init__(self, target_quaternion_xyzw):
        # Supplied by scene configuration, never inferred from an action.
        self.target = Rotation.from_quat(target_quaternion_xyzw)
        self.spec = ActionSpec()

    def encode(self, chunk: ActionChunk, current_quaternion_xyzw) -> np.ndarray:
        if not isinstance(chunk, ActionChunk):
            raise ActionValidationError('Policy must return an ActionChunk')
        if chunk.spec != self.spec:
            raise ActionValidationError("Unsupported action specification")
        if len(chunk.actions) != 1:
            raise ActionValidationError("MVP execution requires H=1")
        action = chunk.actions[0]
        if not isinstance(action, Action):
            raise ActionValidationError('ActionChunk must contain Action values')
        delta = action.delta_position_m
        rotation = action.delta_rotation_rad
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise ActionValidationError("Translation must contain three finite values")
        if np.any(np.abs(delta) > self.spec.max_translation_per_axis_m):
            raise ActionValidationError("Translation exceeds the 5 mm per-axis limit")
        if rotation.shape != (3,) or not np.all(rotation == 0):
            raise ActionValidationError("Fixed-orientation mode requires zero rotation increments")
        if action.gripper not in self.spec.gripper_values:
            raise ActionValidationError("Gripper must be -1 (open) or +1 (close)")
        current = Rotation.from_quat(current_quaternion_xyzw)
        error = (self.target * current.inv()).as_rotvec()
        # Upstream OSC_POSE scales normalized inputs by 0.05 m / 0.5 rad.
        if np.any(np.abs(error) > 0.5):
            raise ActionValidationError("Orientation correction exceeds the OSC input range")
        return np.concatenate((delta / 0.05, error / 0.5, [action.gripper]))
