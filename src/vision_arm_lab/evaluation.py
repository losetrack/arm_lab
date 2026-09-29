"""Simulator-independent episode execution and the public evaluation API."""
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic, sleep
from typing import Callable

from vision_arm_lab.core.contracts import (
    ActionValidationError, PolicyDiagnostics, PolicyFailure, Transition,
)


@dataclass(frozen=True)
class EvaluationReport:
    episodes: tuple[dict, ...]
    summary: dict
    records: Path | None


def summarize_results(results):
    """Compute evaluation metrics independently of recording and file I/O."""
    successes = [result for result in results if result['status'] == 'success']
    outcomes = {}
    for result in results:
        outcomes[result['status']] = outcomes.get(result['status'], 0) + 1
    return {
        'episodes': len(results), 'successes': len(successes),
        'success_rate': len(successes) / len(results) if results else 0,
        'mean_success_time_s': sum(result['sim_time_s'] for result in successes) / len(successes) if successes else None,
        'outcomes': outcomes,
    }


def validate_policy(policy, spec):
    if policy.action_spec != spec.action_spec:
        raise ValueError('Policy action specification does not match the environment')
    missing = policy.required_inputs - spec.capabilities
    if missing:
        raise ValueError(f'Environment does not provide policy inputs: {sorted(missing)}')
    if not callable(policy.reset) or not callable(policy.act):
        raise TypeError('Policy must implement reset() and act(observation)')


def validate_inputs(policy, observation):
    if 'language' in policy.required_inputs and not observation.language_instruction:
        raise ValueError('Actual observation is missing language instruction')
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
    task_metrics = {}
    task_context = None
    failure_reason = None
    error = None
    diagnostic = {}
    previous_diagnostic = None
    try:
        observation = environment.reset(seed)
        task_context = asdict(observation.task_context) if observation.task_context is not None else None
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
            task_metrics = dict(step.task_result.metrics)
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
        'task_id': environment.spec.task.task_id,
        'task_context': task_context, 'task_metrics': task_metrics,
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


def run_evaluation(environment, policy, *, seeds, steps, mode, recorder,
                   on_episode=None, on_diagnostic=None) -> EvaluationReport:
    """Run assembled components; the caller owns the environment context."""
    validate_policy(policy, environment.spec)
    results = []
    try:
        for index, seed in enumerate(seeds):
            recorder.begin_episode(index, seed, mode)
            result = run_episode(
                environment, policy, seed,
                steps if steps is not None else round(environment.spec.episode_timeout_s / environment.spec.action_spec.interval_s),
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
    summary = recorder.finish(summarize_results(results))
    return EvaluationReport(tuple(results), summary, recorder.root)
