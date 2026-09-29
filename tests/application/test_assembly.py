from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from vision_arm_lab.application import assemble_policy
from vision_arm_lab.algorithms.policies import HoldPolicy
from vision_arm_lab.configuration import load_config


@pytest.mark.parametrize('mode', ['inspect', 'vision', 'expert', 'custom', 'custom_locator'])
def test_policy_assembly_preserves_inputs_and_actual_metadata(mode):
    spec = load_config('configs/mvp.yaml').spec
    calls = []
    state = SimpleNamespace(position_m=np.array([0.1, 0.2, 0.82]))
    def read_state():
        calls.append('privileged')
        return state
    environment = SimpleNamespace(spec=spec, _read_task_state=read_state)
    def policy_factory(received):
        assert received is spec
        calls.append('policy')
        return HoldPolicy(received)
    def locator_factory(received):
        assert received is spec.task
        calls.append('locator')
        return lambda observation: np.array([0.2, 0.1, 0.82])
    policy, metadata = assemble_policy(
        mode, environment,
        policy_factory=policy_factory if mode == 'custom' else None,
        locator_factory=locator_factory if mode == 'custom_locator' else None,
    )
    assert 'privileged' not in calls
    if mode == 'expert':
        np.testing.assert_array_equal(policy.locate(None), state.position_m)
        state.position_m = np.array([0.3, 0.4, 0.82])
        np.testing.assert_array_equal(policy.locate(None), state.position_m)
    if mode in ('vision', 'expert', 'custom_locator'):
        assert metadata['grasp'] == asdict(policy.config)
    else:
        assert metadata == {}
    if mode == 'vision':
        assert metadata['vision'] == asdict(policy.locate.config)
