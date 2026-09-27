"""Reuse robosuite assets and reset logic for the confirmed placement scene."""

import numpy as np
from xml.etree.ElementTree import SubElement
from robosuite.environments.manipulation.lift import Lift
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string, new_site
from robosuite.utils.placement_samplers import UniformRandomSampler


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
        arena = TableArena(c.table_size_m, c.friction, self.table_offset)
        arena.set_origin([0, 0, 0])
        camera = arena.worldbody.find(f"camera[@name='{c.camera_name}']")
        camera.set("pos", array_to_string(c.camera_position_m))
        quat = np.asarray(c.camera_quaternion_wxyz, dtype=float)
        camera.set("quat", array_to_string(quat / np.linalg.norm(quat)))
        camera.set("fovy", str(c.camera_fovy_deg))
        arena.worldbody.append(new_site(
            name="target_region", type="box",
            pos=[*c.target_center_m, c.table_height_m + 0.0005],
            size=[*(np.asarray(c.target_size_m) / 2), 0.0005],
            rgba=[0, 1, 0, 0.35],
        ))
        self.cube = BoxObject(
            name="cube", size=[c.cube_side_m / 2] * 3,
            density=c.cube_mass_kg / c.cube_side_m ** 3,
            friction=c.friction, rgba=[1, 0, 0, 1], rng=self.rng,
        )
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
        self.model.create_default_element("option").set("timestep", str(1 / c.physics_hz))
        SubElement(self.model.create_default_element("visual"), "quality", offsamples=str(c.offscreen_samples))

    def reward(self, action=None):
        # Lift's reward/success definition must not leak into placement evaluation.
        return 0.0

    def _check_success(self):
        return False  # PlacementEvaluator owns success and the stable-time history.
