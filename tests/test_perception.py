from types import SimpleNamespace
import numpy as np
import pytest
from vision_arm_lab.perception import ColorLocator
from vision_arm_lab.policies import PolicyFailure


def scene():
    return SimpleNamespace(camera_name='fixed', table_height_m=0.8, cube_side_m=0.04)


def observation():
    rgb = np.zeros((5, 5, 3), dtype=np.uint8)
    rgb[1:4, 1:4, 0] = 255
    transform = np.eye(4)
    transform[:3, 3] = [0.1, -0.1, 0]
    camera = SimpleNamespace(rgb=rgb, depth_m=np.full((5, 5), 0.84),
        intrinsics=np.array([[100., 0, 2], [0, 100., 2], [0, 0, 1.]]),
        camera_to_world=transform)
    return SimpleNamespace(cameras={'fixed': camera})


def test_localizes_using_only_declared_observation():
    np.testing.assert_allclose(ColorLocator(scene())(observation()), [0.1, -0.1, 0.82])


@pytest.mark.parametrize('bad', [np.nan, np.inf, 0, 4, 0.70])
def test_invalid_depth_or_wrong_surface_fails(bad):
    obs = observation()
    obs.cameras['fixed'].depth_m[:] = bad
    with pytest.raises(PolicyFailure, match='localization_failed'):
        ColorLocator(scene())(obs)


def test_missing_depth_and_absent_color_fail():
    obs = observation()
    obs.cameras['fixed'].rgb[:] = 0
    with pytest.raises(PolicyFailure):
        ColorLocator(scene())(obs)
    obs.cameras['fixed'].depth_m = None
    with pytest.raises(PolicyFailure, match='missing_rgbd'):
        ColorLocator(scene())(obs)
