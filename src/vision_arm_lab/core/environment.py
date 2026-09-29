"""Public environment lifecycle; independent of the simulator implementation."""
from typing import Callable, Protocol

from vision_arm_lab.core.contracts import ActionChunk, ActionValidationError, EnvironmentSpec, Observation, StepResult


class Backend(Protocol):
    def reset(self, seed: int) -> Observation: ...
    def step(self, action: ActionChunk) -> Observation: ...
    def render(self) -> None: ...
    def close(self) -> None: ...
    @property
    def diagnostics(self) -> dict: ...


class Environment:
    def __init__(self, backend: Backend, evaluator, read_task_state: Callable,
                 spec: EnvironmentSpec, *, render_mode='offscreen'):
        if render_mode not in ('offscreen', 'window'):
            raise ValueError('render_mode must be offscreen or window')
        self._backend = backend
        self._evaluator = evaluator
        self._read_task_state = read_task_state
        self.spec = spec
        self.render_mode = render_mode
        self.observation = None
        self._done = False

    def reset(self, seed: int = 0) -> Observation:
        if type(seed) is not int or seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        self.observation = None
        self._done = False
        try:
            self._evaluator.reset()
            self.observation = self._backend.reset(seed)
            if self.observation.robot.joint_names != self.spec.joint_names:
                raise ValueError('Robot joint order does not match EnvironmentSpec')
            if tuple(self.observation.cameras) != self.spec.camera_names:
                raise ValueError('Camera names do not match EnvironmentSpec')
            return self.observation
        except BaseException:
            self.close()
            raise

    def step(self, actions: ActionChunk) -> StepResult:
        if self.observation is None:
            raise RuntimeError('Call reset before step')
        if self._done:
            raise RuntimeError('Episode ended; call reset before step')
        try:
            observation = self._backend.step(actions)
            result = self._evaluator.update(
                self._read_task_state(), observation.timestamp_s, actions.actions[0].gripper,
            )
        except ActionValidationError:
            raise  # No physics step took place; the caller can correct the action.
        except BaseException:
            self.close()
            raise
        self.observation = observation
        self._done = result.terminated
        return StepResult(observation, result)

    @property
    def diagnostics(self):
        return dict(self._backend.diagnostics)

    def render(self):
        if self.render_mode != 'window':
            raise RuntimeError('render() requires window mode; read RGB from Observation in offscreen mode')
        self._backend.render()

    def close(self):
        self.observation = None
        self._done = False
        self._backend.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
