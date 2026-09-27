from types import SimpleNamespace

import numpy as np


def test_sensor_allowlist_pixel_convention_and_owned_arrays(monkeypatch):
    from vision_arm_lab.backends import robosuite as backend

    # An extra privileged key must never be requested or propagated.
    class Raw(dict):
        def __getitem__(self, key):
            assert key != "cube_pos", "Privileged object state was read"
            return super().__getitem__(key)

    raw = Raw({
        "agentview_image": np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
        "agentview_depth": np.arange(6, dtype=float).reshape(2, 3, 1),
        "robot0_joint_pos": np.zeros(7), "robot0_joint_vel": np.zeros(7),
        "robot0_eef_pos": np.array([0, 0, 1.0]),
        "robot0_eef_quat_site": np.array([1.0, 0, 0, 0]),
        "robot0_eef_quat": np.array([0, 0, 0, 1.0]),
        "robot0_gripper_qpos": np.array([0.04, -0.04]),
        "cube_pos": object(),
    })
    env = SimpleNamespace(
        sim=SimpleNamespace(data=SimpleNamespace(time=0.05)),
        robots=[SimpleNamespace(robot_joints=tuple(f"joint{i}" for i in range(7)))],
    )
    monkeypatch.setattr(backend.macros, "IMAGE_CONVENTION", "opengl")
    monkeypatch.setattr(backend, "get_real_depth_map", lambda sim, depth: depth + 1)
    monkeypatch.setattr(backend, "get_camera_extrinsic_matrix", lambda *args: np.eye(4))
    monkeypatch.setattr(backend, "get_camera_intrinsic_matrix", lambda *args: np.array([[2., 0, 1.5], [0, 2., 1.], [0, 0, 1.]]))
    obs = backend.adapt_observation(env, raw, "agentview")
    camera = obs.cameras["agentview"]
    np.testing.assert_array_equal(camera.rgb, raw["agentview_image"][::-1])
    np.testing.assert_array_equal(camera.depth_m, [[4, 5, 6], [1, 2, 3]])
    np.testing.assert_array_equal(camera.intrinsics[:2, 2], [1, 0.5])
    np.testing.assert_array_equal(obs.robot.eef_quaternion_xyzw, [1, 0, 0, 0])
    assert obs.timestamp_s == camera.timestamp_s == 0.05
    assert set(vars(obs)) == {"cameras", "robot", "timestamp_s", "language_instruction"}
    raw["agentview_image"][:] = 0
    raw["robot0_eef_pos"][:] = 0
    assert camera.rgb.any()
    np.testing.assert_array_equal(obs.robot.eef_position_m, [0, 0, 1])
