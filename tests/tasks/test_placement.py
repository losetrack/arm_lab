from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vision_arm_lab.tasks.placement import ObjectState, PlacementConfig, PlacementEvaluator


def state(**changes):
    initial = ObjectState(
        np.array([0.1, 0.15, 0.82]), np.eye(3), np.zeros(3), np.zeros(3), False,
    )
    return replace(initial, **changes)


@pytest.fixture
def evaluator():
    return PlacementEvaluator(PlacementConfig(
        cube_side_m=0.04, table_height_m=0.8,
        target_center_m=(0.1, 0.15), target_size_m=(0.12, 0.12),
        bottom_tolerance_m=0.003, linear_speed_limit_m_s=0.01,
        angular_speed_limit_rad_s=0.1, stable_duration_s=1.0, episode_timeout_s=60.0,
    ))


def test_success_requires_full_second_and_latches(evaluator):
    assert evaluator.update(state(), 0.05, -1).status == "running"
    for step in range(2, 21):
        assert evaluator.update(state(), step * 0.05, -1).status == "running"
    result = evaluator.update(state(), 1.05, -1)
    assert result.status == "success" and result.stable_s == 1.0
    assert evaluator.update(state(gripper_contact=True), 2.0, 1) is result
    evaluator.reset()
    assert evaluator.update(state(), 0.05, -1).status == "running"


@pytest.mark.parametrize("bad", [
    state(position_m=np.array([0.15, 0.15, 0.82])),  # center inside, edge outside
    state(position_m=np.array([0.137, 0.15, 0.82]), rotation=Rotation.from_euler("z", 15, degrees=True).as_matrix()),
    state(position_m=np.array([0.1, 0.15, 0.824])),
    state(position_m=np.array([0.1, 0.15, 0.816])),
    state(linear_velocity_m_s=np.array([0.011, 0, 0])),
    state(angular_velocity_rad_s=np.array([0, 0, 0.101])),
    state(gripper_contact=True),
])
def test_invalid_placement_clears_stability(evaluator, bad):
    evaluator.update(state(), 0, -1)
    assert evaluator.update(bad, 0.8, -1).stable_s == 0
    assert evaluator.update(state(), 1, -1).status == "running"
    assert evaluator.update(state(), 1.95, -1).status == "running"
    assert evaluator.update(state(), 2, -1).status == "success"


@pytest.mark.parametrize("command", [None, 1])
def test_release_command_is_required(evaluator, command):
    evaluator.update(state(), 0, command)
    assert evaluator.update(state(), 2, command).status == "running"


def test_timeout_and_success_on_deadline(evaluator):
    assert evaluator.update(state(), 59.95, 1).status == "running"
    assert evaluator.update(state(), 60.0, 1).status == "timeout"
    evaluator.reset()
    evaluator.update(state(), 59.0, -1)
    assert evaluator.update(state(), 60.0, -1).status == "success"
