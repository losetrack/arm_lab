"""An external algorithm module: importable by CLI or a Python evaluation script.

With this directory on PYTHONPATH, use --policy-factory custom_policy:make_policy
or --locator-factory custom_policy:make_locator. No simulator imports are needed.
"""
import numpy as np

from vision_arm_lab import Action, ActionChunk
from vision_arm_lab.algorithms.perception import ColorLocator, VisionConfig


class HoldPosition:
    required_inputs = frozenset({'robot_state'})

    def __init__(self, spec):
        self.action_spec = spec.action_spec

    def reset(self):
        pass

    def act(self, observation):
        # Replace this decision with your algorithm. Units are metres; each
        # component must stay within action_spec.max_translation_per_axis_m.
        return ActionChunk((Action(np.zeros(3), np.zeros(3), -1),), self.action_spec)


def make_policy(spec):
    return HoldPosition(spec)


def make_locator(task):
    # A locator receives only declared task priors, not a simulator reference.
    return ColorLocator(task, VisionConfig(min_pixels=8))
