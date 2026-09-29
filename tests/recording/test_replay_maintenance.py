from dataclasses import asdict
import json
import sys
import pytest
from vision_arm_lab.core.contracts import ActionSpec
from vision_arm_lab.recording.provenance import fingerprint
from vision_arm_lab.recording.replay import load_episode
from vision_arm_lab.cli.maintenance import main as cleanup


@pytest.fixture
def episode(tmp_path):
    metadata = {'format': 'vision-arm-lab-v2', **fingerprint(), 'action_spec': asdict(ActionSpec())}
    (tmp_path / 'metadata.json').write_text(json.dumps(metadata))
    folder = tmp_path / 'episode_0000'
    folder.mkdir()
    (folder / 'result.json').write_text(json.dumps({'steps': 1, 'recording': {'actions_complete': True}}))
    row = {'step': 1, 'time_s': 0.05, 'delta_position_m': [0,0,0], 'delta_rotation_rad': [0,0,0], 'gripper': -1}
    (folder / 'actions.jsonl').write_text(json.dumps(row) + '\n')
    return folder


def test_replay_rejects_incomplete_and_different_versions(episode):
    assert len(load_episode(episode)[2]) == 1
    path = episode / 'actions.jsonl'
    original = path.read_text()
    path.write_text('')
    with pytest.raises(ValueError, match='incomplete'):
        load_episode(episode)

    path.write_text(original)
    meta_path = episode.parent / 'metadata.json'
    metadata = json.loads(meta_path.read_text())
    metadata['packages']['mujoco'] = 'not-the-recorded-version'
    meta_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='versions'):
        load_episode(episode)


def test_replay_requires_explicit_task_record_format(episode):
    path = episode.parent / 'metadata.json'
    metadata = json.loads(path.read_text())
    metadata['format'] = 'vision-arm-lab-v1'
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='matching old version'):
        load_episode(episode)


def test_cleanup_is_explicit_and_preserves_final_videos(episode, monkeypatch):
    temporary = episode / '.tmp-video-test.mp4'
    retained = episode / 'video.mp4'
    temporary.write_bytes(b'temporary')
    retained.write_bytes(b'historical result')
    monkeypatch.setattr(sys, 'argv', ['maintenance', str(episode.parent)])
    cleanup()
    assert temporary.exists()
    monkeypatch.setattr(sys, 'argv', ['maintenance', str(episode.parent), '--delete'])
    cleanup()
    assert not temporary.exists() and retained.read_bytes() == b'historical result'
