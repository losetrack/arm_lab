"""M0 diagnostic using upstream Lift/Panda defaults, not MVP task parameters.

Print results to stdout; do not create experiment files. Select MUJOCO_GL
before launching Python (egl for offscreen, glfw for a desktop window).
"""

import argparse
import json
import time

import numpy as np
import robosuite as suite
from OpenGL import GL
from robosuite.utils.camera_utils import get_real_depth_map
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", action="store_true")
    parser.add_argument("--t03", action="store_true", help="Check approved action and observation adapters")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--image-size", type=int, required=True)
    args = parser.parse_args()
    if args.image_size <= 0:
        parser.error("--image-size must be positive")

    options = {}
    if args.t03:
        from vision_arm_lab.backends.robosuite import adapt_observation, panda_controller_config
        from vision_arm_lab.contracts import Action, ActionChunk, ActionSpec
        from vision_arm_lab.control import PandaActionAdapter

        options.update(controller_configs=panda_controller_config(), use_object_obs=False)
        # Diagnostic target: exactly downward, matching the upstream reset yaw.
        # The MVP scene's grasp yaw remains a separate scene configuration choice.
        adapter = PandaActionAdapter([np.sqrt(0.5), np.sqrt(0.5), 0, 0])

    env = suite.make(
        "Lift",
        robots="Panda",
        has_renderer=args.window,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names="agentview",
        camera_heights=args.image_size,
        camera_widths=args.image_size,
        camera_depths=True,
        initialization_noise=None,
        seed=args.seed,
        control_freq=20,
        **options,
    )
    try:
        obs = env.reset()
        low, high = env.action_spec
        assert low.shape == high.shape == (7,), (low, high)
        action = np.zeros(7)
        count = 0

        def check_image():
            rgb = obs["agentview_image"]
            raw_depth = obs["agentview_depth"]
            assert rgb.shape == (args.image_size, args.image_size, 3)
            assert rgb.dtype == np.uint8 and np.ptp(rgb) > 0
            assert raw_depth.shape == (args.image_size, args.image_size, 1)
            assert np.all((raw_depth >= 0) & (raw_depth <= 1)), (
                count, float(raw_depth.min()), float(raw_depth.max()),
                GL.glGetString(GL.GL_RENDERER),
            )
            depth = get_real_depth_map(env.sim, raw_depth)
            assert np.isfinite(depth).all() and np.all(depth > 0)
            assert np.ptp(depth) > 0
            return rgb, depth

        check_image()

        def advance(command, steps):
            nonlocal obs, count
            assert np.all(command >= low) and np.all(command <= high)
            for _ in range(steps):
                if args.t03:
                    public = adapt_observation(env, obs, "agentview")
                    # Convert the original diagnostic pulse to physical units.
                    chunk = ActionChunk((Action(command[:3] * 0.05, np.zeros(3), int(command[-1])),), ActionSpec())
                    encoded = adapter.encode(chunk, public.robot.eef_quaternion_xyzw)
                else:
                    encoded = command
                obs, _, done, _ = env.step(encoded)
                count += 1
                assert not done, "Unexpected episode termination"
                assert np.isfinite(obs["robot0_joint_pos"]).all()
                check_image()
                if args.window:
                    env.render()
                    assert env.viewer.viewer.is_running(), "Window closed during check"
                    time.sleep(1 / env.control_freq)

        # Diagnostic pulses: one second per phase at the upstream control rate.
        steps = round(env.control_freq)
        action[-1] = -1
        advance(action, steps)
        opened = obs["robot0_gripper_qpos"].copy()
        displacements = []
        for axis in range(3):
            start = obs["robot0_eef_pos"].copy()
            action[axis] = 0.05
            advance(action, steps)
            action[axis] = 0
            delta = obs["robot0_eef_pos"] - start
            assert delta[axis] > 0.001, (axis, delta)
            displacements.append(delta.tolist())
        action[-1] = 1
        advance(action, steps)
        closed = obs["robot0_gripper_qpos"].copy()
        assert np.linalg.norm(closed) < np.linalg.norm(opened), (opened, closed)

        rgb, depth = check_image()
        t03_result = {}
        if args.t03:
            public = adapt_observation(env, obs, "agentview")
            camera = public.cameras["agentview"]
            np.testing.assert_array_equal(camera.rgb, rgb[::-1])
            np.testing.assert_array_equal(camera.depth_m, depth[::-1, :, 0])
            assert public.timestamp_s == camera.timestamp_s == env.sim.data.time
            site = env.robots[0].eef_site_id["right"]
            measured = Rotation.from_quat(public.robot.eef_quaternion_xyzw)
            np.testing.assert_allclose(measured.as_matrix(), env.sim.data.site_xmat[site].reshape(3, 3), atol=1e-6)
            orientation_error = float((adapter.target * measured.inv()).magnitude())
            assert orientation_error < 0.02, orientation_error  # diagnostic tolerance, radians
            assert np.isclose(env.sim.model.opt.timestep, 0.002)
            ray_error = check_calibrated_depth(env, camera)
            t03_result = {"orientation_error_rad": orientation_error, "max_ray_depth_error_m": ray_error}
        assert np.isclose(env.sim.data.time, count / env.control_freq)
        print(json.dumps({
            "status": "passed",
            "window": args.window,
            "t03": t03_result,
            "gl_renderer": GL.glGetString(GL.GL_RENDERER).decode(),
            "seed": args.seed,
            "steps": count,
            "control_hz": env.control_freq,
            "physics_dt": env.sim.model.opt.timestep,
            "sim_time": env.sim.data.time,
            "action_low": low.tolist(),
            "action_high": high.tolist(),
            "xyz_pulse_displacements_m": displacements,
            "gripper_open_qpos": opened.tolist(),
            "gripper_closed_qpos": closed.tolist(),
            "rgb_shape": list(rgb.shape),
            "depth_range_m": [float(depth.min()), float(depth.max())],
        }, indent=2))
    finally:
        env.close()


def check_calibrated_depth(env, camera):
    """Diagnostic-only geometry query; no privileged data enters Observation."""
    import mujoco

    height, width = camera.depth_m.shape
    camera_origin = camera.camera_to_world[:3, 3].copy()
    errors = []
    for row in range(height // 4, 3 * height // 4, max(1, height // 8)):
        for col in range(width // 4, 3 * width // 4, max(1, width // 8)):
            ray_camera = np.linalg.solve(camera.intrinsics, [col, row, 1])
            ray_world = camera.camera_to_world[:3, :3] @ ray_camera
            norm = np.linalg.norm(ray_world)
            hit = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(
                env.sim.model._model, env.sim.data._data, camera_origin,
                ray_world / norm, np.array([0, 1, 1, 1, 1, 1], dtype=np.uint8),
                1, -1, hit,
            )
            if hit[0] >= 0 and env.sim.model.geom_id2name(int(hit[0])) == "table_visual":
                errors.append(abs(float(camera.depth_m[row, col]) - distance / norm))
    assert len(errors) >= 3, f"Too few table ray samples: {len(errors)}"
    # The default 4x MSAA depth resolve samples subpixel positions rather than
    # the center ray (measured at about 3 mm here at 128 px). This checks gross
    # calibration errors, not submillimeter accuracy or task success.
    assert max(errors) < 0.005, errors
    return max(errors)


if __name__ == "__main__":
    main()
