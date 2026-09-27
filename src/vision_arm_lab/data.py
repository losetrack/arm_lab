"""Opt-in recording with a shared byte budget, including streaming video output."""
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import subprocess
from threading import RLock, Thread
from uuid import uuid4

import cv2
import numpy as np


@dataclass(frozen=True)
class RecordOptions:
    mode: str = 'off'
    output: str = 'runs'
    budget_bytes: int = 128 * 1024 * 1024
    actions: bool = False
    observations: bool = False
    observation_hz: float = 2.0
    observation_episodes: int = 2
    video: str = 'off'
    video_episodes: tuple[int, ...] = ()
    video_max: int = 3
    video_fps: int = 10
    video_size: int = 256

    def __post_init__(self):
        if self.mode not in ('off', 'summary', 'debug'):
            raise ValueError('Unknown recording mode')
        if (self.actions or self.observations or self.video != 'off') and self.mode != 'debug':
            raise ValueError('Additional recording requires debug mode')
        if self.video not in ('off', 'all', 'failures', 'selected'):
            raise ValueError('Unknown video selection')
        if self.video == 'selected' and not self.video_episodes:
            raise ValueError('Selected videos require zero-based episode indices')
        if self.budget_bytes <= 0 or not 0 < self.observation_hz <= 20:
            raise ValueError('Budget must be positive; observation rate must be in (0, 20]')
        if not 1 <= self.video_fps <= 20 or self.video_size < 2 or self.video_size % 2:
            raise ValueError('Video fps must be 1..20; size must be positive and even')
        if self.video_max < 0 or self.observation_episodes < 0:
            raise ValueError('Recording counts must be nonnegative')


class VideoStream:
    """FFmpeg encodes; a bounded pipe reader applies the recorder's byte cap."""
    def __init__(self, recorder, path, size, fps):
        import imageio_ffmpeg
        self.recorder = recorder
        self.path = path
        self.failed = False
        self.frames = 0
        self.process = subprocess.Popen([
            imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{size}x{size}',
            '-r', str(fps), '-i', 'pipe:0', '-an', '-threads', '1',
            '-c:v', 'mpeg4', '-q:v', '5', '-pix_fmt', 'yuv420p',
            '-movflags', 'frag_keyframe+empty_moov', '-f', 'mp4', 'pipe:1',
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.thread = Thread(target=self._drain, daemon=True)
        self.thread.start()

    def _drain(self):
        try:
            while data := self.process.stdout.read(65536):
                if not self.failed and not self.recorder._write(self.path, data, append=True, extra=True):
                    self.failed = True
                # Continue draining after budget exhaustion to avoid pipe deadlock.
        except OSError as exc:
            self.failed = True
            self.recorder.warn(f'video read failed: {exc}')
        finally:
            self.process.stdout.close()

    def write(self, rgb):
        if self.failed:
            return
        try:
            self.process.stdin.write(np.ascontiguousarray(rgb).tobytes())
            self.process.stdin.flush()
            self.frames += 1
        except (OSError, ValueError) as exc:
            self.failed = True
            self.recorder.warn(f'video input failed: {exc}')

    def close(self):
        try:
            self.process.stdin.close()
        except OSError as exc:
            self.failed = True
            self.recorder.warn(f'video close failed: {exc}')
        try:
            code = self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            code = self.process.wait()
            self.recorder.warn('video encoder exceeded shutdown timeout')
        self.thread.join(timeout=10)
        if self.thread.is_alive() or code != 0:
            self.failed = True
            self.recorder.warn(f'video encoder did not finish cleanly (exit {code})')
        return not self.failed and self.frames > 0


class Recorder:
    def __init__(self, options, metadata):
        self.options = options
        self.root = None
        self.lock = RLock()
        self.used = 0
        self.reserve = min(256 * 1024, options.budget_bytes // 2)
        self.extra_stopped = False
        self.warnings = []
        self.stream = None
        self.video_count = 0
        self.index = None
        if options.mode != 'off':
            name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8]
            root = Path(options.output) / name
            try:
                root.mkdir(parents=True, exist_ok=False)
                self.root = root
                self._json(root / 'metadata.json', {'format': 'vision-arm-lab-v1', **metadata})
            except OSError as exc:
                self.warn(f'cannot create recording directory: {exc}')

    def warn(self, message):
        self.warnings.append(message)
        print(f'[recording] {message}', flush=True)

    def _write(self, path, payload, *, append=False, extra=False):
        if self.root is None:
            return False
        with self.lock:
            if extra and self.extra_stopped:
                return False
            try:
                old_size = path.stat().st_size if path.exists() and not append else 0
                limit = self.options.budget_bytes - (self.reserve if extra else 0)
                if self.used - old_size + len(payload) > limit:
                    if not self.extra_stopped:
                        self.warn('budget reached; additional recording stopped, evaluation continues')
                    self.extra_stopped = True
                    return False
                path.parent.mkdir(parents=True, exist_ok=True)
                # Account before writing so a partial disk write cannot escape the budget.
                self.used += len(payload) - old_size
                with path.open('ab' if append else 'wb') as stream:
                    stream.write(payload)
                return True
            except OSError as exc:
                self.extra_stopped = True
                self.warn(f'cannot save {path.name}: {exc}')
                return False

    def _json(self, path, value):
        return self._write(path, json.dumps(value, indent=2).encode())

    def _remove(self, path):
        with self.lock:
            try:
                if path.exists():
                    size = path.stat().st_size
                    path.unlink()
                    self.used -= size
            except OSError as exc:
                self.warn(f'cannot clean own temporary file {path}: {exc}')

    def begin_episode(self, index, seed, mode):
        self.index = index
        self.seed = seed
        self.mode = mode
        self.steps = 0
        self.actions_written = 0
        self.observations_written = 0
        self.observations_incomplete = False
        self.video_incomplete = False
        self.next_observation = 1 / self.options.observation_hz
        self.next_video = 1 / self.options.video_fps
        self.episode_path = None if self.root is None else self.root / f'episode_{index:04d}'
        self.video_selected = self.options.video != 'off' and self.video_count < self.options.video_max
        if self.options.video == 'selected':
            self.video_selected &= index in self.options.video_episodes

    def step(self, chunk, observation):
        self.steps += 1
        if self.root is None or self.options.mode != 'debug':
            return
        options = self.options
        if options.actions:
            action = chunk.actions[0]
            payload = json.dumps({
                'step': self.steps, 'time_s': observation.timestamp_s,
                'delta_position_m': action.delta_position_m.tolist(),
                'delta_rotation_rad': action.delta_rotation_rad.tolist(), 'gripper': action.gripper,
            }).encode() + b'\n'
            if self._write(self.episode_path / 'actions.jsonl', payload, append=True, extra=True):
                self.actions_written += 1
        camera = next(iter(observation.cameras.values()))
        if options.observations and self.index < options.observation_episodes and observation.timestamp_s + 1e-9 >= self.next_observation:
            self.next_observation += 1 / options.observation_hz
            if self.extra_stopped:
                self.observations_incomplete = True
            else:
                payload = BytesIO()
                np.savez_compressed(payload, rgb=camera.rgb, depth_m=camera.depth_m,
                    intrinsics=camera.intrinsics, camera_to_world=camera.camera_to_world,
                    timestamp_s=observation.timestamp_s,
                    joint_position_rad=observation.robot.joint_position_rad,
                    joint_velocity_rad_s=observation.robot.joint_velocity_rad_s,
                    eef_position_m=observation.robot.eef_position_m,
                    eef_quaternion_xyzw=observation.robot.eef_quaternion_xyzw,
                    gripper_position_m=observation.robot.gripper_position_m)
                if self._write(self.episode_path / f'observation_{self.steps:05d}.npz', payload.getvalue(), extra=True):
                    self.observations_written += 1
                else:
                    self.observations_incomplete = True
        if self.video_selected and observation.timestamp_s + 1e-9 >= self.next_video:
            self.next_video += 1 / options.video_fps
            if self.extra_stopped:
                self.video_incomplete = True
                return
            if self.stream is None:
                path = self.episode_path / f'.tmp-video-{uuid4().hex}.mp4'
                try:
                    self.stream = VideoStream(self, path, options.video_size, options.video_fps)
                except OSError as exc:
                    self.video_incomplete = True
                    self.video_selected = False
                    self.warn(f'cannot start video encoder: {exc}')
                    return
            rgb = cv2.resize(camera.rgb, (options.video_size, options.video_size), interpolation=cv2.INTER_AREA)
            self.stream.write(rgb)

    def end_episode(self, result):
        video_path = None
        if self.stream is not None:
            complete = self.stream.close() and not self.video_incomplete
            keep = complete and (self.options.video != 'failures' or result['status'] != 'success')
            if keep:
                destination = self.episode_path / 'video.mp4'
                try:
                    self.stream.path.rename(destination)
                    self.video_count += 1
                    video_path = str(destination.relative_to(self.root))
                except OSError as exc:
                    self.warn(f'cannot retain video: {exc}')
                    self._remove(self.stream.path)
                    self.video_incomplete = True
            else:
                self._remove(self.stream.path)
                self.video_incomplete |= not complete
            self.stream = None
        recording = {
            'actions_complete': self.options.actions and self.steps > 0 and self.steps == self.actions_written,
            'actions_written': self.actions_written, 'observations_written': self.observations_written,
            'observations_incomplete': self.observations_incomplete,
            'video_incomplete': self.video_incomplete, 'video': video_path,
        }
        result = {**result, 'steps': self.steps, 'recording': recording}
        if self.root is not None:
            self._json(self.episode_path / 'result.json', result)
        return result

    def finish(self, results):
        successes = [r for r in results if r['status'] == 'success']
        reasons = {}
        for result in results:
            reasons[result['status']] = reasons.get(result['status'], 0) + 1
        summary = {
            'episodes': len(results), 'successes': len(successes),
            'success_rate': len(successes) / len(results) if results else 0,
            'mean_success_time_s': float(np.mean([r['sim_time_s'] for r in successes])) if successes else None,
            'outcomes': reasons, 'recording': self.options.mode,
            'warnings': self.warnings, 'budget_bytes': self.options.budget_bytes,
        }
        if self.root is not None:
            self._json(self.root / 'summary.json', summary)
        return summary

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self._remove(self.stream.path)
            self.stream = None
