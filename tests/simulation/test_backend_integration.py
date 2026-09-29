"""MuJoCo checks, explicitly enabled with pytest -m integration."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vision_arm_lab.simulation.config import load_config
from vision_arm_lab.core.contracts import Action, ActionChunk

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def backend():
    from vision_arm_lab import make_environment
    value = make_environment(ROOT / "configs/mvp.yaml")
    yield value
    value.close()


def hold(backend):
    return ActionChunk((Action(np.zeros(3), np.zeros(3), -1),), backend.spec.action_spec)


def test_scene_seeds_sensors_and_no_experiment_files(backend, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first = None
    poses = set()
    seeds = load_config(ROOT / 'configs/mvp.yaml').development_seeds
    for seed in seeds + [0]:
        observation = backend.reset(seed)
        env = backend._backend.env  # White-box scene integration assertions only.
        truth = backend._read_task_state()
        assert -0.10 <= truth.position_m[0] <= 0.10
        assert -0.15 <= truth.position_m[1] <= -0.05
        np.testing.assert_allclose(truth.position_m[2], 0.83, atol=1e-12)
        yaw = Rotation.from_matrix(truth.rotation).as_euler("xyz", degrees=True)[2]
        assert -15 <= yaw <= 15
        pose = np.r_[truth.position_m, truth.rotation.ravel()]
        poses.add(tuple(pose))
        if seed == 0:
            if first is None:
                first = pose
            else:
                np.testing.assert_array_equal(first, pose)
        np.testing.assert_allclose(env.sim.model.body_mass[env.cube_body_id], 0.1)
        assert env.sim.model.vis.quality.offsamples == 0
        assert env.sim._render_context_offscreen.con.offSamples == 0
        assert "target_region" in env.sim.model.site_names
        assert "target_region" not in env.sim.model.geom_names
        assert env.sim.model.cam_fovy[env.sim.model.camera_name2id("agentview")] == 45
        camera = observation.cameras["agentview"]
        assert camera.rgb.shape == (256, 256, 3)
        assert camera.depth_m.shape == (256, 256)
        assert np.isfinite(camera.depth_m).all() and np.all(camera.depth_m > 0)
        assert np.ptp(camera.rgb) > 0
        red, green, blue = camera.rgb.astype(float).transpose(2, 0, 1)
        assert np.any((red > 1.5 * green) & (red > 1.5 * blue) & (red > 80))
        assert np.any((green > 1.5 * red) & (green > 1.5 * blue) & (green > 80))
        step = backend.step(hold(backend))
        assert step.observation.timestamp_s == pytest.approx(0.05)
        assert step.task_result.status == "running"
    assert len(poses) == len(seeds)
    assert list(tmp_path.iterdir()) == []


def test_physical_placement_success_reset_and_timeout(backend):
    backend.reset(0)
    env = backend._backend.env
    # Test-only privileged placement, not an expert policy or a grasp success.
    env.sim.data.set_joint_qpos(env.cube.joints[0], [0.1, 0.15, 0.82, 1, 0, 0, 0])
    env.sim.data.set_joint_qvel(env.cube.joints[0], np.zeros(6))
    env.sim.forward()
    for _ in range(60):
        result = backend.step(hold(backend)).task_result
        if result.terminated:
            break
    assert result.status == "success"
    assert result.stable_s >= 1
    with pytest.raises(RuntimeError, match="Episode ended"):
        backend.step(hold(backend))
    backend.reset(0)
    assert backend._evaluator.result.status == "running"
    # Shorten only this test's horizon to check termination wiring. Unit tests
    # separately cover the production 60-second threshold exactly.
    backend._evaluator.config = replace(backend._evaluator.config, episode_timeout_s=0.1)
    backend.step(hold(backend))
    result = backend.step(hold(backend)).task_result
    assert result.status == "timeout"
    backend.close()
    with pytest.raises(RuntimeError, match="reset"):
        backend.step(hold(backend))
