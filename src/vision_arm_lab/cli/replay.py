"""Replay a complete action log under its recorded scene and software versions."""
import argparse
import json

from vision_arm_lab.recording.replay import replay_episode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode')
    parser.add_argument('--window', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps(replay_episode(args.episode, render_mode='window' if args.window else 'offscreen')))
