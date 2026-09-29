from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from vision_arm_lab.core.contracts import Action, ActionChunk, ActionSpec
from vision_arm_lab.recording.recorder import Recorder, RecordOptions


def command():
    return ActionChunk((Action(np.zeros(3), np.zeros(3), -1),), ActionSpec())


def observation(time):
    camera = SimpleNamespace(rgb=np.full((32, 32, 3), 120, dtype=np.uint8),
        depth_m=np.ones((32, 32)), intrinsics=np.eye(3), camera_to_world=np.eye(4))
    robot = SimpleNamespace(joint_position_rad=np.zeros(7), joint_velocity_rad_s=np.zeros(7),
        eef_position_m=np.zeros(3), eef_quaternion_xyzw=np.array([1, 0, 0, 0]), gripper_position_m=np.zeros(2))
    return SimpleNamespace(timestamp_s=time, cameras={'fixed': camera}, robot=robot)


def result(status='success', steps=1):
    return {'status': status, 'sim_time_s': 1, 'seed': 0, 'steps': steps}


def test_off_and_debug_without_switches_create_no_extra_data(tmp_path):
    off = Recorder(RecordOptions(output=str(tmp_path / 'off')), {})
    off.begin_episode(0, 0, 'inspect')
    off.step(command(), observation(0.1))
    off.end_episode(result())
    off.finish({'episodes': 1})
    assert not (tmp_path / 'off').exists()
    debug = Recorder(RecordOptions(mode='debug', output=str(tmp_path / 'debug')), {})
    debug.begin_episode(0, 0, 'inspect')
    debug.step(command(), observation(0.1))
    debug.end_episode(result())
    debug.finish({'episodes': 1})
    assert {p.name for p in debug.root.rglob('*') if p.is_file()} == {'metadata.json', 'result.json', 'summary.json'}


def test_budget_stops_actions_but_retains_summary(tmp_path):
    recorder = Recorder(RecordOptions(mode='debug', output=str(tmp_path), actions=True, budget_bytes=6000), {})
    recorder.begin_episode(0, 0, 'inspect')
    for step in range(100):
        recorder.step(command(), observation(step * 0.05))
    saved = recorder.end_episode(result('timeout', steps=100))
    summary = recorder.finish({'episodes': 1, 'outcomes': {'timeout': 1}})
    assert recorder.extra_stopped and not saved['recording']['actions_complete']
    assert summary['episodes'] == 1 and summary['outcomes'] == {'timeout': 1}
    assert (recorder.root / 'summary.json').exists()
    assert sum(p.stat().st_size for p in recorder.root.rglob('*') if p.is_file()) <= 6000


def test_disk_write_failure_is_reported_and_not_raised(tmp_path, monkeypatch):
    recorder = Recorder(RecordOptions(mode='debug', output=str(tmp_path), actions=True), {})
    recorder.begin_episode(0, 0, 'inspect')
    original = Path.open
    def fail(path, *args, **kwargs):
        if path.name == 'actions.jsonl':
            raise OSError('simulated disk full')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', fail)
    recorder.step(command(), observation(0.1))
    saved = recorder.end_episode(result())
    assert not saved['recording']['actions_complete']
    assert any('simulated disk full' in text for text in recorder.warnings)
    assert recorder.finish({'successes': 1})['successes'] == 1


@pytest.mark.parametrize('mode,status,keep', [('all', 'success', True), ('failures', 'success', False), ('failures', 'grasp_failed', True), ('selected', 'success', True)])
def test_streaming_video_selection_and_cleanup(tmp_path, mode, status, keep):
    options = RecordOptions(mode='debug', output=str(tmp_path), video=mode, video_episodes=(0,), video_size=32)
    recorder = Recorder(options, {})
    recorder.begin_episode(0, 0, 'inspect')
    for step in range(1, 5):
        recorder.step(command(), observation(step * 0.1))
    saved = recorder.end_episode(result(status))
    recorder.close()
    assert bool(saved['recording']['video']) == keep
    assert not list(recorder.root.rglob('.tmp-video-*'))
    if keep:
        capture = cv2.VideoCapture(str(recorder.root / saved['recording']['video']))
        try:
            ok, frame = capture.read()
            assert ok and frame.shape == (32, 32, 3)
        finally:
            capture.release()


def test_observations_rate_count_and_small_video_budget(tmp_path):
    recorder = Recorder(RecordOptions(mode='debug', output=str(tmp_path), observations=True, observation_episodes=1, video='all', video_size=32, budget_bytes=7000), {})
    recorder.begin_episode(0, 0, 'inspect')
    for step in range(1, 21):
        recorder.step(command(), observation(step * 0.05))
    recorder.end_episode(result())
    assert recorder.observations_written <= 2
    assert sum(p.stat().st_size for p in recorder.root.rglob('*') if p.is_file()) <= 7000
    recorder.begin_episode(1, 1, 'inspect')
    recorder.step(command(), observation(1))
    recorder.end_episode(result())
    assert recorder.observations_written == 0
    assert not list(recorder.root.rglob('.tmp-video-*'))


def test_recording_flags_require_debug():
    with pytest.raises(ValueError, match='debug'):
        RecordOptions(actions=True)


def test_recording_does_not_replace_execution_steps_or_claim_missing_actions(tmp_path):
    recorder = Recorder(RecordOptions(mode='debug', actions=True, output=str(tmp_path)), {})
    recorder.begin_episode(0, 0, 'custom')
    recorder.step(command(), observation(0.05))
    original = result(steps=2)
    saved = recorder.end_episode(original)
    assert saved['steps'] == original['steps'] == 2
    assert saved['recording']['actions_written'] == 1
    assert not saved['recording']['actions_complete']
    assert 'recording' not in original


@pytest.mark.parametrize('actions', [False, True])
def test_debug_without_images_accepts_robot_only_observations(tmp_path, actions):
    recorder = Recorder(RecordOptions(mode='debug', actions=actions, output=str(tmp_path)), {})
    recorder.begin_episode(0, 0, 'custom')
    obs = observation(0.05)
    obs.cameras = {}
    recorder.step(command(), obs)
    saved = recorder.end_episode(result())
    assert saved['recording']['actions_written'] == int(actions)
    assert saved['recording']['observations_written'] == 0
    assert saved['recording']['video'] is None


def test_video_exhausts_budget_without_stopping_steps(tmp_path):
    recorder = Recorder(RecordOptions(mode='debug', output=str(tmp_path), video='all', video_size=64, budget_bytes=6000), {})
    recorder.begin_episode(0, 0, 'inspect')
    rng = np.random.default_rng(0)
    for step in range(1, 51):
        obs = observation(step * 0.1)
        obs.cameras['fixed'].rgb = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        recorder.step(command(), obs)
    saved = recorder.end_episode(result('timeout', steps=50))
    recorder.finish({'episodes': 1})
    assert saved['steps'] == 50
    assert saved['recording']['video_incomplete']
    assert not list(recorder.root.rglob('.tmp-video-*'))
    assert sum(p.stat().st_size for p in recorder.root.rglob('*') if p.is_file()) <= 6000


def test_selected_video_indices_and_count_cap(tmp_path):
    recorder = Recorder(RecordOptions(mode='debug', output=str(tmp_path), video='selected', video_episodes=(1, 2), video_max=1, video_size=32), {})
    kept = []
    for index in range(3):
        recorder.begin_episode(index, index, 'inspect')
        recorder.step(command(), observation(0.1))
        kept.append(bool(recorder.end_episode(result())['recording']['video']))
    assert kept == [False, True, False]
    assert recorder.video_count == 1
