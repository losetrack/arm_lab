"""Replay must detect changes outside the recording subpackage."""
import json

import pytest

from vision_arm_lab.recording import provenance, replay
from vision_arm_lab.configuration import load_config
from vision_arm_lab.recording.replay import load_episode
from vision_arm_lab.simulation.scene_xml import scene_asset_fingerprint


@pytest.mark.parametrize('changed_file', ['algorithm', 'xml'])
def test_replay_rejects_source_change(tmp_path, monkeypatch, changed_file):
    package = tmp_path / 'src' / 'vision_arm_lab'
    recording = package / 'recording'
    recording.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    source = recording / 'provenance.py'
    source.write_text('# recording module\n')
    algorithm = package / 'algorithms' / 'policies.py'
    algorithm.parent.mkdir()
    algorithm.write_text('SPEED = 1\n')
    xml = package / 'simulation' / 'assets' / 'placement.xml'
    xml.parent.mkdir(parents=True)
    xml.write_text('<mujoco model="original"/>')
    monkeypatch.setattr(provenance, '__file__', str(source))

    before = provenance.fingerprint()
    assert set(before['source_sha256']) == {
        '__init__.py', 'recording/provenance.py', 'algorithms/policies.py',
        'simulation/assets/placement.xml',
    }
    run = tmp_path / 'run'
    episode = run / 'episode_0000'
    episode.mkdir(parents=True)
    (run / 'metadata.json').write_text(json.dumps({'format': 'vision-arm-lab-v2', **before}))
    (episode / 'result.json').write_text(json.dumps({'recording': {'actions_complete': True}}))
    if changed_file == 'algorithm':
        algorithm.write_text('SPEED = 2\n')
    else:
        xml.write_text('<mujoco model="modified"/>')

    with pytest.raises(ValueError, match='source and dependency versions'):
        load_episode(episode)


def test_replay_rejects_modified_external_scene_asset(tmp_path, monkeypatch):
    asset = tmp_path / 'texture.png'
    asset.write_bytes(b'original texture')
    xml = f'<mujoco><asset><texture file="{asset}"/></asset></mujoco>'
    metadata = {**load_config('configs/mvp.yaml').record_metadata(), 'scene': {'scene_xml': xml},
                'scene_asset_sha256': scene_asset_fingerprint(xml)}
    monkeypatch.setattr(replay, 'load_episode', lambda directory: (metadata, {}, []))
    asset.write_bytes(b'edited texture')
    with pytest.raises(ValueError, match='scene asset versions'):
        replay.replay_episode(tmp_path)
