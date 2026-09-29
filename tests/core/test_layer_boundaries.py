"""Core and algorithms must remain usable without scene construction modules."""
import ast
from pathlib import Path
import subprocess
import sys


def test_core_and_algorithms_do_not_import_simulation():
    package = Path(__file__).resolve().parents[2] / 'src/vision_arm_lab'
    forbidden = ('vision_arm_lab.simulation', 'robosuite', 'mujoco', 'OpenGL',
                 'xml', 'yaml')
    for layer in ('core', 'algorithms'):
        for path in (package / layer).rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                names = ([node.module or ''] if isinstance(node, ast.ImportFrom) else
                         [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
                assert not any(name == prefix or name.startswith(prefix + '.')
                               for name in names for prefix in forbidden), path


def test_algorithms_and_core_import_without_loading_scene_modules():
    script = '''
import sys
from vision_arm_lab import Action, Environment, Observation, run_episode
from vision_arm_lab.algorithms.policies import GraspPolicy
from vision_arm_lab.algorithms.perception import ColorLocator
assert not any(name.startswith('vision_arm_lab.simulation') or
               name.startswith('vision_arm_lab.recording') or
               name.split('.')[0] in ('robosuite', 'mujoco', 'OpenGL', 'yaml')
               for name in sys.modules)
'''
    subprocess.run([sys.executable, '-c', script], check=True)


def test_evaluation_and_simulation_do_not_select_algorithms():
    package = Path(__file__).resolve().parents[2] / 'src/vision_arm_lab'
    files = [package / 'evaluation.py', *(package / 'simulation').rglob('*.py')]
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([node.module or ''] if isinstance(node, ast.ImportFrom) else
                     [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            forbidden = ['vision_arm_lab.algorithms', 'vision_arm_lab.application']
            if path.name == 'evaluation.py':
                forbidden += ['vision_arm_lab.simulation', 'vision_arm_lab.recording', 'yaml', 'xml']
            assert not any(name == prefix or name.startswith(prefix + '.')
                           for name in names for prefix in forbidden), path
