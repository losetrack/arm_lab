"""Panda controller configuration and explicit sensor allowlist for robosuite."""

import numpy as np
from robosuite import macros
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
    get_real_depth_map,
)

from vision_arm_lab.contracts import CameraObservation, Observation, RobotState
from vision_arm_lab.control import PandaActionAdapter
from vision_arm_lab.tasks.placement import ObjectState, PlacementEvaluator


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
    capabilities = frozenset({"rgb", "depth", "calibration", "robot_state", "reset", "step"})

    def __init__(self, config, *, window=False):
        self.config = config
        self.window = window
        self.env = None
        self.observation = None
        self.adapter = PandaActionAdapter(config.grasp_quaternion_xyzw)
        self.action_spec = self.adapter.spec
        self.evaluator = PlacementEvaluator(config)

    def reset(self, seed: int) -> Observation:
        from vision_arm_lab.backends.scene import CubePlacement

        # Recreate the single environment so every reset(seed) resets all RNGs,
        # controller goals, gripper state, and evaluator history together.
        self.close()
        c = self.config
        self.env = CubePlacement(
            scene_config=c, robots="Panda", controller_configs=panda_controller_config(),
            has_renderer=self.window, has_offscreen_renderer=True,
            use_camera_obs=True, use_object_obs=False,
            camera_names=c.camera_name, camera_heights=c.camera_size_px[0],
            camera_widths=c.camera_size_px[1], camera_depths=True,
            control_freq=c.control_hz, horizon=round(c.episode_timeout_s * c.control_hz),
            initialization_noise=None, seed=seed, hard_reset=False,
        )
        try:
            raw = self.env.reset()
            self.evaluator.reset()
            self.observation = adapt_observation(self.env, raw, c.camera_name)
            return self.observation
        except Exception:
            self.close()
            raise

    def step(self, chunk):
        if self.env is None or self.observation is None:
            raise RuntimeError("Call reset before step")
        if self.evaluator.result.terminated:
            raise RuntimeError("Episode ended; call reset before step")
        command = self.adapter.encode(chunk, self.observation.robot.eef_quaternion_xyzw)
        raw, _, _, _ = self.env.step(command)
        self.observation = adapt_observation(self.env, raw, self.config.camera_name)
        result = self.evaluator.update(
            self.read_privileged_state(), self.observation.timestamp_s, chunk.actions[0].gripper,
        )
        return self.observation, result

    def read_privileged_state(self) -> ObjectState:
        """Explicit expert/evaluation channel; never passed to visual policies."""
        if self.env is None:
            raise RuntimeError("Call reset before reading privileged state")
        env = self.env
        body_id = env.cube_body_id
        return ObjectState(
            position_m=env.sim.data.body_xpos[body_id].copy(),
            rotation=env.sim.data.body_xmat[body_id].reshape(3, 3).copy(),
            linear_velocity_m_s=env.sim.data.get_body_xvelp(env.cube.root_body),
            angular_velocity_rad_s=env.sim.data.get_body_xvelr(env.cube.root_body),
            gripper_contact=env.check_contact(env.robots[0].gripper["right"], env.cube),
        )

    def render(self):
        if self.env is None or not self.window:
            raise RuntimeError("Window rendering requires reset with window=True")
        self.env.render()

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
        self.observation = None
