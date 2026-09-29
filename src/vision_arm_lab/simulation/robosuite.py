"""Panda controller configuration and explicit sensor allowlist for robosuite."""

import numpy as np
from robosuite import macros
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
    get_real_depth_map,
)

from vision_arm_lab.core.contracts import CameraObservation, Observation, RobotState
from vision_arm_lab.simulation.control import PandaActionAdapter


def panda_controller_config():
    config = load_composite_controller_config(robot="Panda")
    arm = config["body_parts"]["right"]
    arm.update(
        type="OSC_POSE", input_ref_frame="world", input_type="delta",
        input_min=-1, input_max=1,
        output_min=[-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        output_max=[0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
    )
    return config


def adapt_observation(env, raw, camera_name: str) -> Observation:
    """Copy declared sensors only; never retain env or the raw observation dict.

    Call immediately after reset/step at the confirmed 20 Hz camera/control rate.
    Object poses, segmentation and aggregate robosuite states are not copied.
    """
    if macros.IMAGE_CONVENTION != "opengl":
        raise ValueError("Adapter requires robosuite's OpenGL image convention")
    rgb = raw[f"{camera_name}_image"][::-1].copy()
    depth = get_real_depth_map(env.sim, raw[f"{camera_name}_depth"])[::-1, :, 0].copy()
    height, width = rgb.shape[:2]
    timestamp = float(env.sim.data.time)
    intrinsics = get_camera_intrinsic_matrix(env.sim, camera_name, height, width)
    # MuJoCo samples pixel centers. In our OpenCV coordinates the top-left
    # pixel center is (0, 0), so the principal point is (W-1, H-1) / 2.
    intrinsics[:2, 2] -= 0.5
    camera = CameraObservation(
        rgb=rgb, depth_m=depth,
        intrinsics=intrinsics,
        camera_to_world=get_camera_extrinsic_matrix(env.sim, camera_name),
        timestamp_s=timestamp,
    )
    robot = RobotState(
        joint_names=tuple(env.robots[0].robot_joints),
        joint_position_rad=np.array(raw["robot0_joint_pos"], copy=True),
        joint_velocity_rad_s=np.array(raw["robot0_joint_vel"], copy=True),
        eef_position_m=np.array(raw["robot0_eef_pos"], copy=True),
        eef_quaternion_xyzw=np.array(raw["robot0_eef_quat_site"], copy=True),
        gripper_position_m=np.array(raw["robot0_gripper_qpos"], copy=True),
    )
    return Observation(cameras={camera_name: camera}, robot=robot, timestamp_s=timestamp)


class RobosuiteBackend:
    """Physics and sensors only. Scene creation is injected by the factory."""

    def __init__(self, scene_factory, *, camera_name, adapter, window=False):
        self.scene_factory = scene_factory
        self.camera_name = camera_name
        self.adapter = adapter
        self.window = window
        self.env = None
        self.observation = None

    def reset(self, seed: int) -> Observation:
        self.close()
        try:
            self.env = self.scene_factory(seed)
            raw = self.env.reset()
            self.observation = adapt_observation(self.env, raw, self.camera_name)
            return self.observation
        except BaseException:
            self.close()
            raise

    def step(self, chunk):
        if self.env is None or self.observation is None:
            raise RuntimeError("Call reset before step")
        command = self.adapter.encode(chunk, self.observation.robot.eef_quaternion_xyzw)
        raw, _, _, _ = self.env.step(command)
        self.observation = adapt_observation(self.env, raw, self.camera_name)
        return self.observation

    @property
    def diagnostics(self):
        if self.env is None:
            return {'backend': 'robosuite', 'gl_renderer': None}
        from OpenGL import GL
        renderer = GL.glGetString(GL.GL_RENDERER)
        return {'backend': 'robosuite', 'gl_renderer': renderer.decode() if renderer else None}

    def render(self):
        if self.env is None or not self.window:
            raise RuntimeError("Window rendering requires reset with window=True")
        self.env.render()

    def close(self):
        env, self.env = self.env, None
        self.observation = None
        if env is not None:
            env.close()
