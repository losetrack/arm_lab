"""Application configuration: task rules, scene settings and evaluation seeds."""
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

from vision_arm_lab.task_registry import get_recipe


@dataclass(frozen=True)
class EnvironmentConfig:
    task_id: str
    simulation: object
    task: object
    development_seeds: tuple[int, ...]

    def __post_init__(self):
        recipe = get_recipe(self.task_id)
        if not isinstance(self.task, recipe.config_type):
            raise ValueError('Task configuration does not match task_id')
        object.__setattr__(self, 'development_seeds', tuple(self.development_seeds))
        # Shared physical facts originate in the scene, not independent overrides.
        for name in recipe.scene_fields:
            if getattr(self.task, name) != getattr(self.simulation, name):
                raise ValueError(f'{name} must match scene configuration')

    @property
    def spec(self):
        recipe = get_recipe(self.task_id)
        task = recipe.make_task(self.task, self.simulation)
        return recipe.describe(self.simulation, task.info, self.task.episode_timeout_s)

    def record_metadata(self):
        recipe = get_recipe(self.task_id)
        return {
            **recipe.scene_metadata(self.simulation),
            'task': {'id': self.task_id, 'version': recipe.version, 'config': asdict(self.task)},
            'evaluation': {'development_seeds': self.development_seeds},
        }


def load_config(path: str | Path | EnvironmentConfig) -> EnvironmentConfig:
    if isinstance(path, EnvironmentConfig):
        return path
    path = Path(path)
    values = yaml.safe_load(path.read_text())
    if not isinstance(values, dict):
        raise ValueError('Environment configuration must be a YAML mapping')
    task_id = values.pop('task_id', 'placement')
    recipe = get_recipe(task_id)
    seeds = values.pop('development_seeds')
    # Keep the existing flat YAML while giving each component its own config.
    names = {field.name for field in fields(recipe.config_type)}
    rules = {name: values.pop(name) for name in names - recipe.scene_fields if name in values}
    try:
        simulation = recipe.load_scene(values, path.parent)
        scene_values = asdict(simulation)
        task = recipe.config_type(**{name: scene_values[name] for name in recipe.scene_fields}, **rules)
        return EnvironmentConfig(task_id, simulation, task, seeds)
    except TypeError as exc:
        raise ValueError(f'Invalid environment configuration: {exc}') from exc


def restore_config(metadata):
    task = metadata['task']
    recipe = get_recipe(task['id'])
    if task['version'] != recipe.version:
        raise ValueError('Replay requires the recorded task version')
    return EnvironmentConfig(
        task['id'], recipe.restore_scene(metadata), recipe.config_type(**task['config']),
        metadata['evaluation']['development_seeds'],
    )
