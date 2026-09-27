"""Single-environment scene inspection and grasp evaluation."""
import argparse
from dataclasses import asdict
import json
from time import monotonic, sleep
import numpy as np
import yaml
from vision_arm_lab.config import load_config
from vision_arm_lab.contracts import Action, ActionChunk, ActionSpec
from vision_arm_lab.data import Recorder, RecordOptions
from vision_arm_lab.policies import GraspConfig, GraspPolicy, PolicyFailure
from vision_arm_lab.perception import ColorLocator, VisionConfig
from vision_arm_lab.provenance import fingerprint


def run_episode(backend, seed, mode, steps, *, recorder=None, verbose=False):
    start = monotonic()
    obs = backend.reset(seed)
    policy = None
    if mode == 'expert':
        policy = GraspPolicy(backend.config, lambda observation: backend.read_privileged_state().position_m)
    elif mode == 'vision':
        # No backend or privileged callback is given to the vision locator.
        policy = GraspPolicy(backend.config, ColorLocator(backend.config))
    action = ActionChunk((Action(np.zeros(3), np.zeros(3), -1),), backend.action_spec)
    status = 'running'
    for _ in range(steps):
        if policy is not None:
            try:
                previous = policy.phase
                action = policy.act(obs)
                if verbose and previous != policy.phase:
                    print(json.dumps(policy.events[-1]), flush=True)
            except PolicyFailure as exc:
                status = str(exc)
                break
        obs, result = backend.step(action)
        status = result.status
        if recorder is not None:
            recorder.step(action, obs)
        if backend.window:
            backend.render()
            sleep(1 / backend.config.control_hz)
        if result.terminated:
            break
    if status == 'running' and mode != 'inspect':
        status = 'step_limit'
    from OpenGL import GL
    return {
        'seed': seed, 'mode': mode, 'status': status,
        'sim_time_s': obs.timestamp_s, 'wall_time_s': monotonic() - start,
        'observation_source': 'object ground truth and robot sensors' if mode == 'expert' else 'RGB-D and robot sensors',
        'final_phase': policy.phase if policy else None,
        'phases': policy.events if policy else [],
        'gl_renderer': GL.glGetString(GL.GL_RENDERER).decode(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    seeds = parser.add_mutually_exclusive_group(required=True)
    seeds.add_argument('--seed', type=int, nargs='+')
    seeds.add_argument('--seed-file')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--policy', choices=('inspect', 'expert', 'vision'), default='inspect')
    parser.add_argument('--window', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--record', choices=('off', 'summary', 'debug'), default='off')
    parser.add_argument('--output', default='runs')
    parser.add_argument('--budget-mb', type=float, default=128)
    parser.add_argument('--actions', action='store_true')
    parser.add_argument('--observations', action='store_true')
    parser.add_argument('--observation-hz', type=float, default=2)
    parser.add_argument('--observation-episodes', type=int, default=2)
    parser.add_argument('--video', choices=('off', 'all', 'failures', 'selected'), default='off')
    parser.add_argument('--video-episodes', type=int, nargs='+', default=[])
    parser.add_argument('--video-max', type=int, default=3)
    parser.add_argument('--video-fps', type=int, default=10)
    parser.add_argument('--video-size', type=int, default=256)
    args = parser.parse_args()
    if args.steps is not None and args.steps <= 0:
        parser.error('--steps must be positive')
    config = load_config(args.config)
    if args.seed_file:
        with open(args.seed_file) as stream:
            run_seeds = yaml.safe_load(stream)['seeds']
    else:
        run_seeds = args.seed
    options = RecordOptions(
        mode=args.record, output=args.output, budget_bytes=int(args.budget_mb * 1024 * 1024),
        actions=args.actions, observations=args.observations,
        observation_hz=args.observation_hz, observation_episodes=args.observation_episodes,
        video=args.video, video_episodes=tuple(args.video_episodes), video_max=args.video_max,
        video_fps=args.video_fps, video_size=args.video_size,
    )
    metadata = {}
    if args.record != 'off':
        metadata = {
            **fingerprint(), 'scene': asdict(config), 'action_spec': asdict(ActionSpec()),
            'grasp': asdict(GraspConfig()), 'vision': asdict(VisionConfig()),
            'seeds': run_seeds, 'policy': args.policy, 'record_options': asdict(options),
        }
    recorder = Recorder(options, metadata)
    from vision_arm_lab.backends.robosuite import RobosuiteBackend
    backend = RobosuiteBackend(config, window=args.window)
    results = []
    try:
        for index, seed in enumerate(run_seeds):
            recorder.begin_episode(index, seed, args.policy)
            start = monotonic()
            interrupted = False
            try:
                result = run_episode(backend, seed, args.policy,
                    args.steps or round(config.episode_timeout_s * config.control_hz),
                    recorder=recorder, verbose=args.verbose)
            except (Exception, KeyboardInterrupt) as exc:
                interrupted = isinstance(exc, KeyboardInterrupt)
                result = {
                    'seed': seed, 'mode': args.policy,
                    'status': 'interrupted' if interrupted else 'environment_error',
                    'error': f'{type(exc).__name__}: {exc}',
                    'sim_time_s': backend.observation.timestamp_s if backend.observation else None,
                    'wall_time_s': monotonic() - start,
                }
            result = recorder.end_episode(result)
            results.append(result)
            print(json.dumps(result), flush=True)
            if interrupted:
                break
    finally:
        backend.close()
        recorder.close()
    summary = recorder.finish(results)
    print(json.dumps(summary), flush=True)
    if recorder.root is not None:
        print(f'Records: {recorder.root}', flush=True)
    if any(result['status'] == 'interrupted' for result in results):
        raise SystemExit(130)
    if any(result['status'] == 'environment_error' for result in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
