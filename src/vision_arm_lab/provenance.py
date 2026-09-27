"""Source and dependency fingerprints for reproducible runs and replay."""
from hashlib import sha256
from importlib.metadata import distributions
from pathlib import Path
import os
import platform
import subprocess


def fingerprint():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / 'src').rglob('*.py'))
    paths += [root / 'pyproject.toml', root / 'requirements-lock.txt']
    sources = {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest() for p in paths}
    versions = {d.metadata['Name']: d.version for d in distributions()}
    head = subprocess.run(['git', 'rev-parse', '--verify', 'HEAD'], cwd=root, capture_output=True, text=True)
    return {
        'source_sha256': sources, 'packages': versions,
        'git_head': head.stdout.strip() if head.returncode == 0 else None,
        'python': platform.python_version(), 'platform': platform.platform(),
        'render_environment': {key: os.environ.get(key) for key in ('MUJOCO_GL', 'LP_NUM_THREADS')},
    }
