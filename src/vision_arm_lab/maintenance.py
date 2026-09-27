"""Explicitly list or delete only this recorder's interrupted video temporaries."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    parser.add_argument('--delete', action='store_true')
    args = parser.parse_args()
    root = Path(args.run)
    if json.loads((root / 'metadata.json').read_text()).get('format') != 'vision-arm-lab-v1':
        raise ValueError('Not a recognized recording directory')
    for path in sorted(root.glob('episode_[0-9][0-9][0-9][0-9]/.tmp-video-*.mp4')):
        print(path)
        if args.delete:
            path.unlink()


if __name__ == '__main__':
    main()
