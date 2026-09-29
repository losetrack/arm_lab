"""Public boundaries and lifecycle, tested without constructing a simulator."""
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from vision_arm_lab import (
    Action, ActionChunk, CameraObservation, Environment, Observation, RobotState,
    evaluate, run_episode,
)
from vision_arm_lab.core.config import load_config
from vision_arm_lab.simulation.control import PandaActionAdapter
from vision_arm_lab.algorithms.policies import HoldPolicy
from vision_arm_lab.tasks.placement import ObjectState, PlacementEvaluator

CONFIG = Path(__file__).resolve().parents[2] / 'configs/mvp.yaml'


class PhysicsStub:
    def __init__(self, spec):
        self.spec = spec
        self.closed = 0
        self.time = 0
        self.adapter = PandaActionAdapter([1, 0, 0, 0])

    def observation(self):
        camera = CameraObservation(
            np.zeros((2, 3, 3), dtype=np.uint8), np.ones((2, 3)),
            np.eye(3), np.eye(4), self.time,
        )
        robot = RobotState(self.spec.joint_names, np.zeros(7), np.zeros(7),
                           np.zeros(3), np.array([1., 0, 0, 0]), np.zeros(2))
        return Observation({self.spec.camera_names[0]: camera}, robot, self.time)

    def reset(self, seed):
        self.time = 0
        return self.observation()

    def step(self, action):
        self.adapter.encode(action, [1, 0, 0, 0])
        self.time = round(self.time + 0.05, 9)
        return self.observation()

    @property
    def diagnostics(self):
        return {'backend': 'test'}

    def render(self):
        raise AssertionError('Offscreen loop must not render a window')

    def close(self):
        self.closed += 1


def fixture_environment():
    config = load_config(CONFIG)
    spec = replace(config.spec, camera_size_px=(2, 3))
    backend = PhysicsStub(spec)
    evaluator = PlacementEvaluator(config.placement)
    state = ObjectState(np.array([0.1, 0.15, 0.82]), np.eye(3), np.zeros(3), np.zeros(3), False)
    return Environment(backend, evaluator, lambda: state, spec), backend


def test_public_import_does_not_load_simulator():
    code = "import sys; from vision_arm_lab import evaluate, make_environment, Observation; from vision_arm_lab.tasks.placement import PlacementEvaluator; assert not any(k.split('.')[0] in ('robosuite', 'mujoco', 'OpenGL') for k in sys.modules)"
    subprocess.run([sys.executable, '-c', code], check=True)


def test_observation_snapshots_and_camera_mapping_are_readonly():
    pixels = np.zeros((2, 3, 3), dtype=np.uint8)
    camera = CameraObservation(pixels, None, np.eye(3), np.eye(4), 0)
    pixels[:] = 255
    assert not camera.rgb.any()
    with pytest.raises(ValueError):
        camera.rgb[:] = 1
    environment, _ = fixture_environment()
    obs = environment.reset(0)
    with pytest.raises(TypeError):
        obs.cameras['injected'] = camera
    with pytest.raises(ValueError):
        obs.robot.eef_position_m[:] = 1


def test_lifecycle_termination_reset_and_invalid_action():
    environment, backend = fixture_environment()
    policy = HoldPolicy(environment.spec)
    with pytest.raises(RuntimeError, match='reset'):
        environment.step(policy.act(None))
    with environment:
        environment.reset(0)
        bad = ActionChunk((Action(np.ones(3), np.zeros(3), -1),), environment.spec.action_spec)
        with pytest.raises(ValueError):
            environment.step(bad)
        assert backend.time == 0  # Rejection must precede physics.
        for _ in range(21):
            result = environment.step(policy.act(None))
        assert result.done and result.task_result.status == 'success'
        with pytest.raises(RuntimeError, match='ended'):
            environment.step(policy.act(None))
        assert environment.reset(0).timestamp_s == 0
        assert not environment.step(policy.act(None)).done
    assert backend.closed == 1
    assert environment.observation is None


def test_reset_failure_and_context_failure_release_resources(monkeypatch):
    environment, backend = fixture_environment()
    def fail(seed):
        raise RuntimeError('reset failed')
    monkeypatch.setattr(backend, 'reset', fail)
    with pytest.raises(RuntimeError, match='reset failed'):
        environment.reset(0)
    assert backend.closed == 1
    with pytest.raises(RuntimeError, match='caller failed'):
        with environment:
            raise RuntimeError('caller failed')
    assert backend.closed == 2


def test_generic_policy_and_transition_timing():
    environment, _ = fixture_environment()
    transitions = []
    policy = HoldPolicy(environment.spec)  # No phase, events or diagnostics.
    result = run_episode(environment, policy, 0, 2, on_transition=transitions.append)
    assert result['status'] == 'step_limit' and result['steps'] == 2
    assert result['policy_diagnostics'] == {}
    assert transitions[0].observation_before.timestamp_s == 0
    assert transitions[0].observation_after.timestamp_s == 0.05
    assert transitions[1].observation_before is transitions[0].observation_after


@pytest.mark.parametrize('failure,status', [(RuntimeError('model failed'), 'policy_error'),
                                           (KeyboardInterrupt(), 'interrupted')])
def test_policy_failure_is_not_a_backend_error(failure, status):
    environment, _ = fixture_environment()
    class BrokenPolicy(HoldPolicy):
        def act(self, observation):
            raise failure
    result = run_episode(environment, BrokenPolicy(environment.spec), 0, 2)
    assert result['status'] == status and result['steps'] == 0


def test_backend_and_invalid_command_errors_are_distinct(monkeypatch):
    environment, backend = fixture_environment()
    class BadCommand(HoldPolicy):
        def act(self, observation):
            return ActionChunk((Action(np.ones(3), np.zeros(3), -1),), self.action_spec)
    assert run_episode(environment, BadCommand(environment.spec), 0, 2)['status'] == 'policy_error'
    def fail(action):
        raise RuntimeError('physics failed')
    monkeypatch.setattr(backend, 'step', fail)
    result = run_episode(environment, HoldPolicy(environment.spec), 0, 2)
    assert result['status'] == 'environment_error' and result['error']['stage'] == 'step'
    assert backend.closed == 1 and environment.observation is None


def test_wrong_return_type_is_a_policy_error():
    environment, _ = fixture_environment()
    class WrongType(HoldPolicy):
        def act(self, observation):
            return None
    assert run_episode(environment, WrongType(environment.spec), 0, 1)['status'] == 'policy_error'


def test_capability_and_action_spec_rejected_before_reset():
    environment, backend = fixture_environment()
    policy = HoldPolicy(environment.spec)
    policy.required_inputs = frozenset({'segmentation'})
    with pytest.raises(ValueError, match='segmentation'):
        run_episode(environment, policy, 0, 1)
    policy.required_inputs = frozenset()
    policy.action_spec = replace(policy.action_spec, frame='base')
    with pytest.raises(ValueError, match='specification'):
        run_episode(environment, policy, 0, 1)
    assert backend.time == 0


def test_evaluate_injection_reset_no_files_and_cleanup(tmp_path, monkeypatch):
    environment, backend = fixture_environment()
    monkeypatch.setattr('vision_arm_lab.evaluation.make_environment', lambda *a, **kw: environment)
    monkeypatch.chdir(tmp_path)
    received = []
    def factory(spec):
        received.append(spec)
        assert not hasattr(spec, 'backend') and not hasattr(spec.task, 'position_m')
        return HoldPolicy(spec)
    report = evaluate(CONFIG, seeds=[0, 1], policy_factory=factory, steps=2)
    assert received == [environment.spec]
    assert [r['sim_time_s'] for r in report.episodes] == [0.1, 0.1]
    assert report.summary['episodes'] == 2 and report.records is None
    assert list(tmp_path.iterdir()) == [] and backend.closed == 1


def test_callback_failure_is_visible_and_closes_environment(monkeypatch):
    environment, backend = fixture_environment()
    monkeypatch.setattr('vision_arm_lab.evaluation.make_environment', lambda *a, **kw: environment)
    def fail(result):
        raise RuntimeError('consumer failed')
    with pytest.raises(RuntimeError, match='consumer failed'):
        evaluate(CONFIG, seeds=[0], policy_factory=HoldPolicy, steps=1, on_episode=fail)
    assert backend.closed == 1


@pytest.mark.parametrize('changes', [dict(camera_size_px=[0, 256]), dict(control_hz=10),
                                    dict(camera_position_m=[float('nan'), 0, 0]),
                                    dict(spawn_x_m=[1, 0]), dict(cube_side_m=-1)])
def test_config_validation_precedes_simulation(changes):
    with pytest.raises(ValueError):
        replace(load_config(CONFIG), **changes)
