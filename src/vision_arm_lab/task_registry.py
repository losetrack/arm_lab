"""Application-owned recipes connecting pure tasks to simulation adapters."""
from dataclasses import dataclass
from typing import Callable

from vision_arm_lab.tasks.placement import PlacementConfig, PlacementTask
from vision_arm_lab.simulation.config import load_simulation_config, restore_simulation_config, simulation_metadata
from vision_arm_lab.simulation.factory import make_backend, describe_environment


@dataclass(frozen=True)
class TaskRecipe:
    version: int
    config_type: type
    scene_fields: frozenset[str]
    make_task: Callable
    load_scene: Callable
    restore_scene: Callable
    scene_metadata: Callable
    make_backend: Callable
    describe: Callable


RECIPES = {
    'placement': TaskRecipe(
        PlacementTask.version, PlacementConfig,
        frozenset({'cube_side_m', 'table_height_m', 'target_center_m', 'target_size_m'}),
        lambda config, scene: PlacementTask(config, scene.camera_name),
        load_simulation_config, restore_simulation_config, simulation_metadata,
        make_backend, describe_environment,
    ),
}


def get_recipe(task_id):
    if not isinstance(task_id, str) or task_id not in RECIPES:
        raise ValueError(f'Unknown task: {task_id!r}')
    return RECIPES[task_id]
