"""Command-line adapter for the public evaluation API."""
import argparse
import importlib
import json


def add_arguments(parser):
    parser.add_argument('--config', required=True)
    seeds = parser.add_mutually_exclusive_group(required=True)
    seeds.add_argument('--seed', type=int, nargs='+')
    seeds.add_argument('--seed-file')
    parser.add_argument('--steps', type=int)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--policy', choices=('inspect', 'expert', 'vision'))
    selection.add_argument('--policy-factory', metavar='MODULE:CALLABLE')
    selection.add_argument('--locator-factory', metavar='MODULE:CALLABLE')
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


def import_factory(reference):
    module, separator, name = reference.partition(':')
    if not separator or not module or not name:
        raise ValueError('Factory must use MODULE:CALLABLE syntax')
    factory = getattr(importlib.import_module(module), name)
    if not callable(factory):
        raise ValueError(f'{reference} is not callable')
    return factory


def execute(args, *, default_policy='inspect'):
    from vision_arm_lab.recording.recorder import RecordOptions
    from vision_arm_lab.evaluation import evaluate
    if args.seed_file:
        import yaml
        with open(args.seed_file) as stream:
            seeds = yaml.safe_load(stream)['seeds']
    else:
        seeds = args.seed
    options = RecordOptions(
        mode=args.record, output=args.output, budget_bytes=int(args.budget_mb * 1024 * 1024),
        actions=args.actions, observations=args.observations,
        observation_hz=args.observation_hz, observation_episodes=args.observation_episodes,
        video=args.video, video_episodes=tuple(args.video_episodes), video_max=args.video_max,
        video_fps=args.video_fps, video_size=args.video_size,
    )
    custom = args.policy_factory or args.locator_factory
    report = evaluate(
        args.config, seeds=seeds, policy=args.policy or (None if custom else default_policy),
        policy_factory=import_factory(args.policy_factory) if args.policy_factory else None,
        locator_factory=import_factory(args.locator_factory) if args.locator_factory else None,
        render_mode='window' if args.window else 'offscreen', steps=args.steps, record=options,
        on_episode=lambda result: print(json.dumps(result), flush=True),
        on_diagnostic=(lambda value: print(json.dumps({'policy_diagnostics': value}), flush=True)) if args.verbose else None,
        policy_metadata={'factory': custom} if custom else None,
    )
    print(json.dumps(report.summary), flush=True)
    if report.records is not None:
        print(f'Records: {report.records}', flush=True)
    statuses = {result['status'] for result in report.episodes}
    if 'interrupted' in statuses:
        return 130
    return 1 if statuses & {'environment_error', 'policy_error'} else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        code = execute(args)
    except (ValueError, TypeError, OSError, KeyError, ImportError, AttributeError) as exc:
        parser.exit(2, f'Configuration/startup error: {exc}\n')
    raise SystemExit(code)


if __name__ == '__main__':
    main()
