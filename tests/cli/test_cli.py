"""CLI/API wiring, installation diagnostics and explicit startup failures."""
import json
from pathlib import Path
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from vision_arm_lab.cli import main
from vision_arm_lab.cli.diagnostics import doctor
from vision_arm_lab.cli.runner import import_factory


@pytest.mark.parametrize('module', [
    'vision_arm_lab', 'vision_arm_lab.runner',
    'vision_arm_lab.replay', 'vision_arm_lab.maintenance',
])
def test_module_command_entry_points(module):
    env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[2] / 'src')}
    result = subprocess.run([sys.executable, '-m', module, '--help'],
                            capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout


def test_cli_uses_api_report_and_exit_status(monkeypatch, capsys):
    calls = []
    def evaluate(config, **kwargs):
        calls.append((config, kwargs))
        result = {'status': 'policy_error'}
        kwargs['on_episode'](result)
        return SimpleNamespace(episodes=(result,), summary={'episodes': 1}, records=None)
    monkeypatch.setattr('vision_arm_lab.evaluation.evaluate', evaluate)
    with pytest.raises(SystemExit) as exit:
        main(['evaluate', '--config', 'scene.yaml', '--seed', '3', '--policy', 'expert'])
    assert exit.value.code == 1
    assert calls[0][0] == 'scene.yaml'
    assert calls[0][1]['policy'] == 'expert' and calls[0][1]['seeds'] == [3]
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == {'episodes': 1}


def test_inspect_rejects_algorithm_selection():
    with pytest.raises(SystemExit) as exit:
        main(['inspect', '--config', 'unused', '--seed', '0', '--policy', 'vision'])
    assert exit.value.code == 2


def test_explicit_factory_loading():
    factory = import_factory('vision_arm_lab.algorithms.policies:HoldPolicy')
    assert factory.__name__ == 'HoldPolicy'
    with pytest.raises(ValueError, match='MODULE:CALLABLE'):
        import_factory('bad_reference')


def test_doctor_reports_version_mismatch_without_creating_environment(monkeypatch):
    monkeypatch.setattr('vision_arm_lab.cli.diagnostics.requires', lambda name: ['numpy==1.26.4'])
    monkeypatch.setattr('vision_arm_lab.cli.diagnostics.version', lambda name: '0.2.0' if name == 'vision-arm-lab' else '2.0.0')
    def forbidden(*args, **kwargs):
        raise AssertionError('Do not create a rendering context with mismatched dependencies')
    monkeypatch.setattr('vision_arm_lab.simulation.factory.make_environment', forbidden)
    report = doctor(config='unused')
    assert report['checks'] == [{'component': 'numpy', 'version': '2.0.0', 'required': '==1.26.4', 'ok': False}]
