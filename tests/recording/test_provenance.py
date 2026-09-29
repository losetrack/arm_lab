"""Replay must detect changes outside the recording subpackage."""
import json

import pytest

from vision_arm_lab.recording import provenance
from vision_arm_lab.recording.replay import load_episode


def test_replay_rejects_algorithm_source_change(tmp_path, monkeypatch):
    package = tmp_path / 'src' / 'vision_arm_lab'
    recording = package / 'recording'
    recording.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    source = recording / 'provenance.py'
    source.write_text('# recording module\n')
    algorithm = package / 'algorithms' / 'policies.py'
    algorithm.parent.mkdir()
    algorithm.write_text('SPEED = 1\n')
    monkeypatch.setattr(provenance, '__file__', str(source))

    before = provenance.fingerprint()
    assert set(before['source_sha256']) == {
        '__init__.py', 'recording/provenance.py', 'algorithms/policies.py',
    }
    run = tmp_path / 'run'
    episode = run / 'episode_0000'
    episode.mkdir(parents=True)
    (run / 'metadata.json').write_text(json.dumps({'format': 'vision-arm-lab-v1', **before}))
    (episode / 'result.json').write_text(json.dumps({'recording': {'actions_complete': True}}))
    algorithm.write_text('SPEED = 2\n')

    with pytest.raises(ValueError, match='source and dependency versions'):
        load_episode(episode)
