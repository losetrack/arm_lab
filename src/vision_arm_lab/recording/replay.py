"""Replay a complete action log under its recorded scene and software versions."""
from dataclasses import asdict
import json
from pathlib import Path
from time import sleep
import numpy as np
from vision_arm_lab.core.config import SceneConfig
from vision_arm_lab.core.contracts import Action, ActionChunk, ActionSpec
from vision_arm_lab.recording.provenance import fingerprint, scene_asset_fingerprint
from vision_arm_lab.simulation.factory import make_environment


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


def replay_episode(directory, *, render_mode='offscreen'):
    metadata, recorded, rows = load_episode(directory)
    if metadata['scene_asset_sha256'] != scene_asset_fingerprint(metadata['scene']['scene_xml']):
        raise ValueError('Replay requires the recorded scene asset versions')
    with make_environment(SceneConfig(**metadata['scene']), render_mode=render_mode) as environment:
        environment.reset(recorded['seed'])
        for row in rows:
            command = ActionChunk((Action(np.array(row['delta_position_m']), np.array(row['delta_rotation_rad']), row['gripper']),), environment.spec.action_spec)
            step = environment.step(command)
            if render_mode == 'window':
                environment.render()
                sleep(environment.spec.action_spec.interval_s)
        return {'replayed_steps': len(rows), 'sim_time_s': step.observation.timestamp_s,
                'task_status': step.task_result.status}
