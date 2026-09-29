"""Installed command-line entry points sharing the public Python API."""
import argparse
import json

from vision_arm_lab.cli.runner import add_arguments, execute


def main(argv=None):
    parser = argparse.ArgumentParser(prog='vision-arm', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('evaluate', 'inspect'):
        add_arguments(commands.add_parser(name))
    check = commands.add_parser('doctor', help='Check installation; optionally create a rendering context')
    check.add_argument('--config', help='Explicitly enable a rendering smoke check for this scene')
    check.add_argument('--window', action='store_true')
    check.add_argument('--steps', type=int, default=2)
    replay = commands.add_parser('replay')
    replay.add_argument('episode')
    replay.add_argument('--window', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'doctor':
            from vision_arm_lab.cli.diagnostics import doctor
            if args.window and not args.config:
                raise ValueError('--window requires --config for the rendering check')
            report = doctor(config=args.config, render_mode='window' if args.window else 'offscreen', steps=args.steps)
            print(json.dumps(report, indent=2))
            code = 0 if all(check['ok'] for check in report['checks']) else 1
        elif args.command == 'replay':
            from vision_arm_lab.recording.replay import replay_episode
            print(json.dumps(replay_episode(args.episode, render_mode='window' if args.window else 'offscreen')))
            code = 0
        else:
            if args.command == 'inspect':
                if args.policy_factory or args.locator_factory or args.policy not in (None, 'inspect'):
                    raise ValueError('inspect only supports the hold policy; use evaluate for algorithms')
                args.policy = 'inspect'
            code = execute(args, default_policy='vision')
    except (ValueError, TypeError, OSError, KeyError, ImportError, AttributeError) as exc:
        parser.exit(2, f'Configuration/startup error: {exc}\n')
    raise SystemExit(code)
