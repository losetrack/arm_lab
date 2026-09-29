"""Reuse robosuite assets and reset logic for the confirmed placement scene."""

import numpy as np
from copy import deepcopy
import xml.etree.ElementTree as ET
from robosuite.environments.manipulation.lift import Lift
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.base import MujocoXML
from robosuite.models.objects import MujocoObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler

from vision_arm_lab.core.scene_xml import resolve_asset_file


class PlacementArena(MujocoXML):
    """Adapt an already resolved XML snapshot to robosuite's model merger."""
    def __init__(self, xml, table_height):
        self.root = ET.fromstring(xml)
        self.table_offset = np.array([0.0, 0.0, table_height])
        for section in ('worldbody', 'asset', 'actuator', 'sensor', 'tendon', 'equality', 'contact'):
            setattr(self, section, self.create_default_element(section))
        for node in self.asset.findall('*[@file]'):
            node.set('file', str(resolve_asset_file(node.get('file'))))


class PlacementCube(MujocoObject):
    """Use the XML body's actual geoms and joints, preserving their names."""
    def __init__(self, body, side):
        super().__init__(duplicate_collision_geoms=False)
        self._name = 'cube'
        self._obj = body
        self.side = side
        self._get_object_properties()

    def exclude_from_prefixing(self, inp):
        return True  # Scene XML already contains the stable cube_* names.

    @property
    def bottom_offset(self):
        return np.array([0.0, 0.0, -self.side / 2])

    @property
    def top_offset(self):
        return -self.bottom_offset

    @property
    def horizontal_radius(self):
        return self.side / np.sqrt(2)


class CubePlacement(Lift):
    def __init__(self, scene_config, **kwargs):
        self.scene_config = scene_config
        super().__init__(
            table_full_size=scene_config.table_size_m,
            table_friction=scene_config.friction,
            **kwargs,
        )

    def _load_model(self):
        # Reuse robot loading, without creating Lift's textured random-size cube.
        ManipulationEnv._load_model(self)
        c = self.scene_config
        self.table_offset = np.array([0.0, 0.0, c.table_height_m])
        robot = self.robots[0].robot_model
        robot.set_base_xpos(robot.base_xpos_offset["table"](c.table_size_m[0]))
        arena = PlacementArena(c.scene_xml, c.table_height_m)
        cube_body = arena.worldbody.find("body[@name='cube_main']")
        arena.worldbody.remove(cube_body)
        self.cube = PlacementCube(cube_body, c.cube_side_m)
        # Bound every possible spawn using the square's circumscribed radius;
        # reject overlapping configurations rather than resampling difficult seeds.
        radius = c.cube_side_m / np.sqrt(2)
        spawn_low = np.array([c.spawn_x_m[0], c.spawn_y_m[0]]) - radius
        spawn_high = np.array([c.spawn_x_m[1], c.spawn_y_m[1]]) + radius
        target_low = np.asarray(c.target_center_m) - np.asarray(c.target_size_m) / 2
        target_high = np.asarray(c.target_center_m) + np.asarray(c.target_size_m) / 2
        gap = np.maximum(np.maximum(target_low - spawn_high, spawn_low - target_high), 0)
        if np.linalg.norm(gap) < c.initial_target_gap_m:
            raise ValueError("Spawn range does not guarantee the configured target gap")
        self.placement_initializer = UniformRandomSampler(
            name="CubeSampler", mujoco_objects=self.cube,
            x_range=c.spawn_x_m, y_range=c.spawn_y_m,
            rotation=np.deg2rad(c.spawn_yaw_deg), rotation_axis="z",
            ensure_object_boundary_in_range=False, ensure_valid_placement=True,
            reference_pos=self.table_offset, z_offset=c.spawn_clearance_m, rng=self.rng,
        )
        self.model = ManipulationTask(arena, [robot], self.cube)
        # robosuite merges bodies/assets but not global MJCF options. Apply the
        # scene's explicit global sections after assembling the Panda model.
        merged = {'worldbody', 'asset', 'actuator', 'sensor', 'tendon', 'equality', 'contact'}
        for section in arena.root:
            if section.tag not in merged:
                existing = self.model.root.find(section.tag)
                if existing is not None:
                    self.model.root.remove(existing)
                self.model.root.append(deepcopy(section))

    def reward(self, action=None):
        # Lift's reward/success definition must not leak into placement evaluation.
        return 0.0

    def _check_success(self):
        return False  # PlacementEvaluator owns success and the stable-time history.
