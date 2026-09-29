"""Dependency checks and opt-in rendering smoke checks; never modifies the host."""
from importlib.metadata import PackageNotFoundError, requires, version
import platform

def doctor(*, config=None, render_mode='offscreen', steps=2):
    checks = []
    report = {'python': platform.python_version(), 'checks': checks}
    try:
        report['project_version'] = version('vision-arm-lab')
        dependencies = requires('vision-arm-lab') or []
    except PackageNotFoundError:
        checks.append({'component': 'project', 'ok': False,
                       'error': 'Install the project with python -m pip install -e . first'})
        return report
    try:
        from packaging.requirements import Requirement
    except ImportError:
        checks.append({'component': 'packaging', 'ok': False, 'error': 'Install the locked dependencies first'})
        return report
    for text in dependencies:
        requirement = Requirement(text)
        if requirement.marker and not requirement.marker.evaluate({'extra': ''}):
            continue
        try:
            installed = version(requirement.name)
            checks.append({'component': requirement.name, 'version': installed,
                           'required': str(requirement.specifier),
                           'ok': installed in requirement.specifier})
        except PackageNotFoundError:
            checks.append({'component': requirement.name, 'ok': False, 'error': 'Not installed'})
    if config is not None and all(check['ok'] for check in checks):
        from vision_arm_lab.application import make_environment
        from vision_arm_lab.algorithms.policies import HoldPolicy
        try:
            if type(steps) is not int or steps <= 0:
                raise ValueError('steps must be a positive integer')
            with make_environment(config, render_mode=render_mode) as environment:
                observation = environment.reset(0)
                policy = HoldPolicy(environment.spec)
                for _ in range(steps):
                    step = environment.step(policy.act(observation))
                    observation = step.observation
                    if render_mode == 'window':
                        environment.render()
                    if step.done:
                        break
                checks.append({'component': 'render', 'ok': True, 'mode': render_mode,
                               **environment.diagnostics, 'sim_time_s': observation.timestamp_s})
        except Exception as exc:
            checks.append({'component': 'render', 'ok': False, 'mode': render_mode,
                           'error': f'{type(exc).__name__}: {exc}'})
    return report
