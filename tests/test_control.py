from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vision_arm_lab.contracts import Action, ActionChunk, ActionSpec
from vision_arm_lab.control import PandaActionAdapter


def chunk(delta=(0, 0, 0), rotation=(0, 0, 0), gripper=-1):
    return ActionChunk((Action(np.array(delta), np.array(rotation), gripper),), ActionSpec())


def test_physical_units_and_world_rotation_correction():
    target = Rotation.from_euler("xyz", [np.pi, 0, 0])
    current = Rotation.from_rotvec([0.1, -0.07, 0.03]) * target
    adapter = PandaActionAdapter(target.as_quat())
    command = adapter.encode(chunk((0.005, -0.005, 0.001), gripper=1), current.as_quat())
    np.testing.assert_allclose(command[:3], [0.1, -0.1, 0.02])
    corrected = Rotation.from_rotvec(command[3:6] * 0.5) * current
    np.testing.assert_allclose(corrected.as_matrix(), target.as_matrix(), atol=1e-12)
    assert command[-1] == 1


@pytest.mark.parametrize("invalid", [
    chunk((0.005001, 0, 0)), chunk((np.nan, 0, 0)),
    chunk((0, 0)), chunk(rotation=(0, 0, 0.01)),
    chunk(rotation=(0, np.inf, 0)), chunk(gripper=0),
])
def test_invalid_action_is_rejected_without_clipping(invalid):
    with pytest.raises(ValueError):
        PandaActionAdapter([1, 0, 0, 0]).encode(invalid, [1, 0, 0, 0])


def test_unsupported_spec_chunk_and_large_orientation_error():
    adapter = PandaActionAdapter([1, 0, 0, 0])
    for invalid in (
        replace(chunk(), spec=replace(ActionSpec(), frame="base")),
        replace(chunk(), spec=replace(ActionSpec(), interval_s=0.1)),
        replace(chunk(), actions=()),
        replace(chunk(), actions=chunk().actions * 2),
    ):
        with pytest.raises(ValueError):
            adapter.encode(invalid, [1, 0, 0, 0])
    with pytest.raises(ValueError, match="Orientation correction"):
        adapter.encode(chunk(), [0, 0, 0, 1])
