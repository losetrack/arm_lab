"""Simulator-independent episode execution and the public evaluation API."""
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic, sleep
from typing import Callable

from vision_arm_lab.core.config import load_config
from vision_arm_lab.core.contracts import (
    ActionValidationError, PolicyDiagnostics, PolicyFailure, Transition,
)
from vision_arm_lab.recording.recorder import Recorder, RecordOptions
from vision_arm_lab.simulation.factory import make_builtin_policy, make_environment


@dataclass(frozen=True)
class EvaluationReport:
    episodes: tuple[dict, ...]
    summary: dict
    records: Path | None


def validate_policy(policy, spec):
    if policy.action_spec != spec.action_spec:
        raise ValueError('Policy action specification does not match the environment')
    missing = policy.required_inputs - spec.capabilities
    if missing:
        raise ValueError(f'Environment does not provide policy inputs: {sorted(missing)}')
    if not callable(policy.reset) or not callable(policy.act):
        raise TypeError('Policy must implement reset() and act(observation)')


def validate_inputs(policy, observation):
    for name in policy.required_inputs & {'rgb', 'depth', 'calibration'}:
        field = {'rgb': 'rgb', 'depth': 'depth_m', 'calibration': 'intrinsics'}[name]
        if not observation.cameras or any(getattr(camera, field) is None for camera in observation.cameras.values()):
            raise ValueError(f'Actual observation is missing required input: {name}')
        if name == 'calibration' and any(camera.camera_to_world is None for camera in observation.cameras.values()):
            raise ValueError('Actual observation is missing camera_to_world calibration')


def run_episode(environment, policy, seed, steps, *, mode='custom',
                on_transition: Callable | None = None, on_diagnostic: Callable | None = None):
    """Drive one episode. No simulator imports, file I/O or process exit here.

    Unexpected policy/backend failures become explicit results; callbacks are owned
    by the caller, so their exceptions propagate rather than being misclassified.
    """
    if type(steps) is not int or steps <= 0:
        raise ValueError('steps must be a positive integer')
    validate_policy(policy, environment.spec)
    start = monotonic()
    observation = None
    executed = 0
    phase = 'reset'
    status = 'running'
    task_status = None
    failure_reason = None
    error = None
    diagnostic = {}
    previous_diagnostic = None
    try:
        observation = environment.reset(seed)
        task_status = 'running'
        phase = 'policy'
        validate_inputs(policy, observation)
        policy.reset()
        for index in range(1, steps + 1):
            phase = 'policy'
            actions = policy.act(observation)
            if isinstance(policy, PolicyDiagnostics):
                diagnostic = dict(policy.diagnostics())
                if on_diagnostic is not None and diagnostic != previous_diagnostic:
                    phase = 'callback'
                    on_diagnostic(diagnostic)
                    previous_diagnostic = diagnostic
            phase = 'step'
            step = environment.step(actions)
            before, observation = observation, step.observation
            executed = index
            status = step.task_result.status
            task_status = step.task_result.status
            if on_transition is not None:
                phase = 'callback'
                on_transition(Transition(index, before, actions, observation, step.task_result))
            if environment.render_mode == 'window':
                phase = 'render'
                environment.render()
                sleep(environment.spec.action_spec.interval_s)
            if step.done:
                break
        if status == 'running' and mode != 'inspect':
            status = 'step_limit'
    except KeyboardInterrupt:
        status = 'interrupted'
    except PolicyFailure as exc:
        if phase != 'policy':
            raise
        status = 'policy_failure'
        failure_reason = str(exc)
    except Exception as exc:
        if phase == 'callback':
            raise
        status = 'policy_error' if phase == 'policy' or isinstance(exc, ActionValidationError) else 'environment_error'
        error = {'stage': phase, 'type': type(exc).__name__, 'message': str(exc)}
    result = {
        'seed': seed, 'mode': mode, 'status': status, 'steps': executed,
        'task_status': task_status,
        'sim_time_s': observation.timestamp_s if observation is not None else None,
        'wall_time_s': monotonic() - start,
        'observation_source': 'object ground truth and robot sensors' if mode == 'expert' else 'public sensors and declared task priors',
        'policy_diagnostics': diagnostic,
    }
    if error is not None:
        result['error'] = error
    if failure_reason is not None:
        result['failure_reason'] = failure_reason
    return result


def evaluate(config, *, seeds, policy=None, policy_factory=None, locator_factory=None,
             render_mode='offscreen', steps=None, record='off',
             on_episode=None, on_diagnostic=None, policy_metadata=None) -> EvaluationReport:
    """Evaluate sequential episodes with one environment and explicit recording.

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
    results = []
    # Factory creates no physics context until reset. Context ownership covers all
    # later policy, recorder, callback and interruption failure paths.
    with make_environment(config, render_mode=render_mode) as environment:
        if policy_factory is not None:
            algorithm = policy_factory(environment.spec)
        elif locator_factory is not None:
            from vision_arm_lab.algorithms.policies import GraspPolicy
            algorithm = GraspPolicy(environment.spec.task, locator_factory(environment.spec.task))
        else:
            algorithm = make_builtin_policy(mode, environment)
        validate_policy(algorithm, environment.spec)
        metadata = {}
        if options.mode != 'off':
            from vision_arm_lab.recording.provenance import fingerprint, scene_asset_fingerprint
            from vision_arm_lab.algorithms.policies import GraspConfig
            from vision_arm_lab.algorithms.perception import VisionConfig
            metadata = {
                **fingerprint(), 'scene': asdict(config),
                'scene_asset_sha256': scene_asset_fingerprint(config.scene_xml),
                'action_spec': asdict(environment.spec.action_spec),
                'seeds': seeds, 'policy': mode, 'record_options': asdict(options),
                'policy_metadata': policy_metadata or {},
            }
            if mode in ('expert', 'vision', 'custom_locator'):
                metadata['grasp'] = asdict(GraspConfig())
            if mode == 'vision':
                metadata['vision'] = asdict(VisionConfig())
        recorder = Recorder(options, metadata)
        try:
            for index, seed in enumerate(seeds):
                recorder.begin_episode(index, seed, mode)
                result = run_episode(
                    environment, algorithm, seed,
                    steps if steps is not None else round(config.episode_timeout_s * config.control_hz),
                    mode=mode, on_transition=recorder.record_transition,
                    on_diagnostic=on_diagnostic,
                )
                result['environment'] = environment.diagnostics
                result = recorder.end_episode(result)
                results.append(result)
                if on_episode is not None:
                    on_episode(result)
                if result['status'] == 'interrupted':
                    break
        finally:
            recorder.close()
        summary = recorder.finish(results)
        return EvaluationReport(tuple(results), summary, recorder.root)
