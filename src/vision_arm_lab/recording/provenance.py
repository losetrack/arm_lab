"""Source and dependency fingerprints for reproducible runs and replay."""
from hashlib import sha256
from importlib.metadata import distributions
from pathlib import Path
import os
import platform
import subprocess
import shutil


def fingerprint():
    package = Path(__file__).resolve().parents[1]
    sources = {str(p.relative_to(package)): sha256(p.read_bytes()).hexdigest()
               for p in sorted(package.rglob('*.py'))}
    versions = {d.metadata['Name']: d.version for d in distributions()}
    root = package.parent.parent
    head = None
    if (root / '.git').exists() and shutil.which('git'):
        process = subprocess.run(['git', 'rev-parse', '--verify', 'HEAD'], cwd=root, capture_output=True, text=True)
        if process.returncode == 0:
            head = process.stdout.strip()
    return {
        'source_sha256': sources, 'packages': versions,
        'git_head': head,
        'python': platform.python_version(), 'platform': platform.platform(),
        'render_environment': {key: os.environ.get(key) for key in ('MUJOCO_GL', 'LP_NUM_THREADS')},
    }
