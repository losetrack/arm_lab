"""Color/depth localization with declared camera, table and cube-size priors."""
from dataclasses import dataclass
import cv2
import numpy as np
from vision_arm_lab.core.contracts import PolicyFailure

from vision_arm_lab.tasks.placement import PlacementTaskInfo


@dataclass(frozen=True)
class VisionConfig:
    hue_low_max: int = 10
    hue_high_min: int = 170
    saturation_min: int = 120
    value_min: int = 60
    min_pixels: int = 5
    depth_min_m: float = 0.1
    depth_max_m: float = 3.0
    top_tolerance_m: float = 0.005


class ColorLocator:
    required_inputs = frozenset({'rgb', 'depth', 'calibration'})
    required_priors = frozenset({'table_height', 'cube_size', 'color_range'})

    def __init__(self, task: PlacementTaskInfo, config=VisionConfig()):
        self.task = task
        self.config = config

    def __call__(self, observation):
        camera = observation.cameras[self.task.camera_name]
        if camera.rgb is None or camera.depth_m is None:
            raise PolicyFailure('missing_rgbd')
        c = self.config
        hsv = cv2.cvtColor(camera.rgb, cv2.COLOR_RGB2HSV)
        mask = ((hsv[..., 0] <= c.hue_low_max) | (hsv[..., 0] >= c.hue_high_min))
        mask &= (hsv[..., 1] >= c.saturation_min) & (hsv[..., 2] >= c.value_min)
        mask &= np.isfinite(camera.depth_m) & (camera.depth_m >= c.depth_min_m) & (camera.depth_m <= c.depth_max_m)
        rows, cols = np.nonzero(mask)
        if len(rows) < c.min_pixels:
            raise PolicyFailure('localization_failed')
        rays = np.linalg.solve(camera.intrinsics, np.vstack((cols, rows, np.ones(len(rows)))))
        points_camera = rays * camera.depth_m[rows, cols]
        points = (camera.camera_to_world[:3, :3] @ points_camera).T + camera.camera_to_world[:3, 3]
        # Table and the fixed cube height distinguish the top face from side
        # faces and the red visualization site above the workspace.
        top_z = self.task.table_height_m + self.task.cube_side_m
        points = points[np.abs(points[:, 2] - top_z) <= c.top_tolerance_m]
        if len(points) < c.min_pixels:
            raise PolicyFailure('localization_failed')
        xy = np.median(points[:, :2], axis=0)
        return np.r_[xy, self.task.table_height_m + self.task.cube_side_m / 2]
