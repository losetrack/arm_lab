from types import SimpleNamespace
import numpy as np
import pytest
from vision_arm_lab.config import load_config
from vision_arm_lab.policies import GraspPolicy, PolicyFailure


def observation(time, position=(0, 0, 1), gap=0.08):
    return SimpleNamespace(timestamp_s=time, robot=SimpleNamespace(
        eef_position_m=np.array(position, dtype=float),
        gripper_position_m=np.array([gap / 2, -gap / 2]),
    ))


def test_localization_runs_once_and_motion_is_bounded():
    calls = []
    def locate(obs):
        calls.append(obs)
        return np.array([0.1, -0.1, 0.82])
    policy = GraspPolicy(load_config('configs/mvp.yaml'), locate)
    policy.act(observation(1))
    assert policy.phase == 'approach' and len(calls) == 1
    action = policy.act(observation(1.05)).actions[0]
    np.testing.assert_array_equal(action.delta_position_m, [0.005, -0.005, -0.005])
    with pytest.raises(PolicyFailure, match='stage_timeout'):
        policy.act(observation(17))
    assert len(calls) == 1
    policy.reset()
    assert policy.phase == 'settle' and policy.target is None


def test_empty_grasp_and_wait_terminate():
    policy = GraspPolicy(load_config('configs/mvp.yaml'), lambda obs: [0, 0, 0.82])
    policy.phase = 'close'
    with pytest.raises(PolicyFailure, match='grasp_failed'):
        policy.act(observation(1.1, gap=0.001))
    policy.phase = 'wait'
    with pytest.raises(PolicyFailure, match='not_in_target'):
        policy.act(observation(3.1))


def test_release_does_not_advance_until_gripper_opens():
    policy = GraspPolicy(load_config('configs/mvp.yaml'), lambda obs: [0, 0, 0.82])
    policy.phase = 'release'
    policy.act(observation(0.6, gap=0.04))
    assert policy.phase == 'release'
    policy.act(observation(0.7, gap=0.075))
    assert policy.phase == 'retreat'
