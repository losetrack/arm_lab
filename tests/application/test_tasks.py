"""A registered test task needs no placement geometry or changes to the runner."""
from dataclasses import asdict, dataclass, replace
import json

import numpy as np
import pytest
import yaml

from vision_arm_lab import Action, ActionChunk, ActionSpec, EnvironmentSpec, Observation, RobotState
from vision_arm_lab import TaskContext, TaskInfo, TaskResult, evaluate, make_environment, RecordOptions
from vision_arm_lab.core.contracts import ActionValidationError
from vision_arm_lab.application import make_recorded_environment
from vision_arm_lab.configuration import load_config
from vision_arm_lab.recording.replay import replay_episode
from vision_arm_lab.task_registry import RECIPES, TaskRecipe


@dataclass(frozen=True)
class TestScene:
    __test__ = False


@dataclass(frozen=True)
class CounterConfig:
    episode_timeout_s: float


@dataclass(frozen=True)
class CounterInfo(TaskInfo):
    ticks: int


class CounterTask:
    info = CounterInfo('counter', 0)

    def reset(self, seed, state, observation):
        self.goal = seed + 2
        return TaskContext('counter', seed, f'Wait for {self.goal} ticks.', CounterInfo('counter', self.goal))

    def update(self, state, observation, actions):
        assert isinstance(actions, ActionChunk)
        return TaskResult('success' if state['ticks'] >= self.goal else 'running',
                          observation.timestamp_s, {'ticks': state['ticks']})


class CounterBackend:
    def __init__(self):
        self.ticks = 0
        self.closed = False

    def observation(self):
        robot = RobotState((), np.zeros(0), np.zeros(0), np.zeros(3),
                           np.array([0, 0, 0, 1]), np.zeros(2))
        return Observation({}, robot, self.ticks * 0.05)

    def reset(self, seed):
        self.ticks = 0
        return self.observation()

    def step(self, action):
        self.ticks += 1
        return self.observation()

    @property
    def diagnostics(self):
        return {'backend': 'counter'}

    def close(self):
        self.closed = True


@pytest.fixture
def counter(monkeypatch, tmp_path):
    def backend(*args, **kwargs):
        value = CounterBackend()
        return value, lambda: {'ticks': value.ticks, 'secret': object()}
    recipe = TaskRecipe(
        1, CounterConfig, frozenset(), lambda config, scene: CounterTask(),
        lambda values, directory: TestScene(**values), lambda metadata: TestScene(**metadata['scene']),
        lambda scene: {'scene': asdict(scene)}, backend,
        lambda scene, task, timeout: EnvironmentSpec(ActionSpec(), (), (0, 0), (),
            frozenset({'robot_state', 'language'}), task, timeout),
    )
    monkeypatch.setitem(RECIPES, 'counter', recipe)
    path = tmp_path / 'counter.yaml'
    path.write_text(yaml.safe_dump({'task_id': 'counter', 'episode_timeout_s': 1,
                                    'development_seeds': [0, 1]}))
    return path


class LanguagePolicy:
    required_inputs = frozenset({'language'})

    def __init__(self, spec):
        self.action_spec = spec.action_spec

    def reset(self):
        pass

    def act(self, obs):
        assert obs.language_instruction == obs.task_context.language_instruction
        assert obs.task_context.goal.ticks == obs.task_context.seed + 2
        assert not hasattr(obs.task_context, 'secret')
        return ActionChunk((Action([0, 0, 0], [0, 0, 0], -1),), self.action_spec)


def test_task_reset_context_metrics_recording_and_replay(counter, tmp_path):
    report = evaluate(counter, seeds=[0, 1], policy_factory=LanguagePolicy,
                      record=RecordOptions(mode='debug', actions=True, output=str(tmp_path / 'runs')))
    assert [result['steps'] for result in report.episodes] == [2, 3]
    assert report.summary['successes'] == 2
    for result in report.episodes:
        assert result['task_id'] == 'counter'
        assert result['task_metrics'] == {'ticks': result['steps']}
        assert result['task_context']['goal']['ticks'] == result['steps']
    metadata = json.loads((report.records / 'metadata.json').read_text())
    assert metadata['task'] == {'id': 'counter', 'version': 1, 'config': {'episode_timeout_s': 1}}
    assert replay_episode(report.records / 'episode_0001')['task_status'] == 'success'
    metadata['task']['version'] = 999
    with pytest.raises(ValueError, match='task version'):
        make_recorded_environment(metadata)


def test_unknown_and_incompatible_tasks_fail_explicitly(counter):
    values = yaml.safe_load(counter.read_text())
    values['task_id'] = 'unknown'
    counter.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match='Unknown task'):
        make_environment(counter)
    values['task_id'] = 'counter'
    counter.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match='placement task'):
        evaluate(counter, seeds=[0], policy='vision')


def test_task_reset_failure_clears_context_and_releases_backend(counter):
    with make_environment(counter) as env:
        env.reset(0)
        def fail(*args):
            raise RuntimeError('task reset failed')
        env._task.reset = fail
        with pytest.raises(RuntimeError, match='task reset failed'):
            env.reset(1)
        assert env.task_context is None and env.observation is None
        assert env._backend.closed


def test_placement_scene_and_rule_overrides_cannot_disagree():
    config = load_config('configs/mvp.yaml')
    with pytest.raises(ValueError, match='must match scene configuration'):
        replace(config, task=replace(config.task, target_center_m=(0.5, 0.5)))


@pytest.mark.parametrize('error', [RuntimeError, ActionValidationError])
def test_task_step_failure_after_physics_closes_environment(counter, error):
    with make_environment(counter) as env:
        obs = env.reset(0)
        def fail(*args):
            raise error('task failed')
        env._task.update = fail
        with pytest.raises(error, match='task failed'):
            env.step(LanguagePolicy(env.spec).act(obs))
        assert env._backend.ticks == 1 and env._backend.closed
        assert env.observation is None and env.task_context is None
