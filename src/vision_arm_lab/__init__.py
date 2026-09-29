"""Vision Arm Lab: RGB-D robot manipulation experiments."""

from importlib import import_module

_MODULES = {
    **dict.fromkeys((
        'Action', 'ActionChunk', 'ActionSpec', 'CameraObservation', 'EnvironmentSpec',
        'Observation', 'Policy', 'PolicyFailure', 'RobotState', 'StepResult',
        'TaskInfo', 'TaskContext', 'Task', 'TaskResult', 'Transition',
    ), 'core.contracts'),
    'Environment': 'core.environment', 'make_environment': 'application',
    'EvaluationReport': 'evaluation', 'evaluate': 'application',
    'run_episode': 'evaluation', 'RecordOptions': 'recording.recorder',
}
__all__ = list(_MODULES)


def __getattr__(name):
    # Keep --help and doctor usable even when numerical/rendering dependencies
    # are missing. Public objects are imported only when actually requested.
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module(f'.{_MODULES[name]}', __name__), name)
    globals()[name] = value
    return value
