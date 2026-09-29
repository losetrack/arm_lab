"""Public application entry point assembling simulation, algorithms and recording."""
from dataclasses import asdict

from vision_arm_lab.evaluation import EvaluationReport, run_evaluation, validate_policy
from vision_arm_lab.recording.recorder import Recorder, RecordOptions
from vision_arm_lab.configuration import load_config, restore_config
from vision_arm_lab.core.environment import Environment
from vision_arm_lab.task_registry import get_recipe
from vision_arm_lab.simulation.factory import make_expert_locator


def make_environment(config, *, render_mode='offscreen'):
    config = load_config(config)
    recipe = get_recipe(config.task_id)
    task = recipe.make_task(config.task, config.simulation)
    spec = recipe.describe(config.simulation, task.info, config.task.episode_timeout_s)
    backend, read_state = recipe.make_backend(config.simulation, spec.episode_timeout_s, render_mode=render_mode)
    return Environment(backend, task, read_state, spec, render_mode=render_mode)


def make_recorded_environment(metadata, *, render_mode='offscreen'):
    return make_environment(restore_config(metadata), render_mode=render_mode)


def assemble_policy(mode, environment, *, policy_factory=None, locator_factory=None):
    """Keep algorithm selection and actual algorithm metadata at one boundary."""
    if policy_factory is not None:
        return policy_factory(environment.spec), {}

    from vision_arm_lab.algorithms.policies import GraspPolicy, HoldPolicy

    if mode == 'inspect':
        return HoldPolicy(environment.spec), {}
    if environment.spec.task.task_id != 'placement':
        raise ValueError('Built-in grasp policies and locator_factory require the placement task')
    metadata = {}
    if locator_factory is not None:
        locate = locator_factory(environment.spec.task)
    elif mode == 'expert':
        locate = make_expert_locator(environment)
    elif mode == 'vision':
        from vision_arm_lab.algorithms.perception import ColorLocator
        locate = ColorLocator(environment.spec.task)
        metadata['vision'] = asdict(locate.config)
    else:
        raise ValueError(f'Unknown built-in policy: {mode}')
    algorithm = GraspPolicy(environment.spec.task, locate)
    metadata['grasp'] = asdict(algorithm.config)
    return algorithm, metadata


def evaluate(config, *, seeds, policy=None, policy_factory=None, locator_factory=None,
             render_mode='offscreen', steps=None, record='off',
             on_episode=None, on_diagnostic=None, policy_metadata=None) -> EvaluationReport:
    """Assemble one environment and policy, then evaluate sequential episodes.

    Custom factories receive only EnvironmentSpec (or TaskInfo for locators).
    A fresh custom policy is created once per evaluation and reset per episode.
    """
    config = load_config(config)
    seeds = tuple(seeds)
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError('seeds must be a nonempty sequence of nonnegative integers')
    if steps is not None and (type(steps) is not int or steps <= 0):
        raise ValueError('steps must be a positive integer')
    if sum(value is not None for value in (policy, policy_factory, locator_factory)) > 1:
        raise ValueError('Select only one of policy, policy_factory and locator_factory')
    if policy is not None and policy not in ('inspect', 'expert', 'vision'):
        raise ValueError(f'Unknown built-in policy: {policy}')
    options = RecordOptions(mode=record) if isinstance(record, str) else record
    if not isinstance(options, RecordOptions):
        raise TypeError('record must be a mode string or RecordOptions')
    mode = 'custom' if policy_factory is not None else 'custom_locator' if locator_factory is not None else policy or 'vision'
    # Environment ownership includes policy construction and recorder failures.
    with make_environment(config, render_mode=render_mode) as environment:
        algorithm, algorithm_metadata = assemble_policy(
            mode, environment, policy_factory=policy_factory, locator_factory=locator_factory,
        )
        validate_policy(algorithm, environment.spec)
        metadata = {}
        if options.mode != 'off':
            from vision_arm_lab.recording.provenance import fingerprint
            metadata = {
                **fingerprint(), **config.record_metadata(), **algorithm_metadata,
                'action_spec': asdict(environment.spec.action_spec),
                'seeds': seeds, 'policy': mode, 'record_options': asdict(options),
                'policy_metadata': policy_metadata or {},
            }
        return run_evaluation(
            environment, algorithm, seeds=seeds, steps=steps, mode=mode,
            recorder=Recorder(options, metadata),
            on_episode=on_episode, on_diagnostic=on_diagnostic,
        )
