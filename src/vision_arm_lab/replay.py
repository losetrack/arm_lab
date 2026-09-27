"""Replay a complete action log under its recorded scene and software versions."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import sleep
import numpy as np
from vision_arm_lab.config import SceneConfig
from vision_arm_lab.contracts import Action, ActionChunk, ActionSpec
from vision_arm_lab.provenance import fingerprint


def load_episode(directory):
    directory = Path(directory)
    metadata = json.loads((directory.parent / 'metadata.json').read_text())
    result = json.loads((directory / 'result.json').read_text())
    if metadata.get('format') != 'vision-arm-lab-v1' or not result['recording']['actions_complete']:
        raise ValueError('Replay requires a complete action record')
    current = fingerprint()
    if metadata['packages'] != current['packages'] or metadata['source_sha256'] != current['source_sha256']:
        raise ValueError('Replay requires the recorded source and dependency versions')
    spec = json.loads(json.dumps(asdict(ActionSpec())))
    if metadata['action_spec'] != spec:
        raise ValueError('Unsupported recorded action specification')
    rows = [json.loads(line) for line in (directory / 'actions.jsonl').read_text().splitlines()]
    if len(rows) != result['steps'] or not rows:
        raise ValueError('Action log is incomplete')
    for i, row in enumerate(rows, 1):
        if row['step'] != i or not np.isclose(row['time_s'], i * spec['interval_s'], rtol=0, atol=1e-7):
            raise ValueError('Action sequence or timestamps are incomplete')
    return metadata, result, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode')
    parser.add_argument('--window', action='store_true')
    args = parser.parse_args()
    metadata, recorded, rows = load_episode(args.episode)
    from vision_arm_lab.backends.robosuite import RobosuiteBackend
    backend = RobosuiteBackend(SceneConfig(**metadata['scene']), window=args.window)
    try:
        backend.reset(recorded['seed'])
        for row in rows:
            command = ActionChunk((Action(np.array(row['delta_position_m']), np.array(row['delta_rotation_rad']), row['gripper']),), backend.action_spec)
            observation, result = backend.step(command)
            if args.window:
                backend.render()
                sleep(backend.action_spec.interval_s)
        print(json.dumps({'replayed_steps': len(rows), 'sim_time_s': observation.timestamp_s, 'task_status': result.status}))
    finally:
        backend.close()


if __name__ == '__main__':
    main()
