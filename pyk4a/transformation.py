from typing import Optional

import numpy as np

import k4a_module

from .calibration import Calibration


def depth_image_to_color_camera(depth: np.ndarray, calibration: Calibration, thread_safe: bool) -> Optional[np.ndarray]:
    """
    Transform depth color_image to color color_image space
    Return empty result if transformation failed
    """
    return k4a_module.transformation_depth_image_to_color_camera(
        calibration.transformation_handle, thread_safe, depth, calibration.color_resolution,
    )


def depth_image_to_point_cloud(depth: np.ndarray, calibration: Calibration, thread_safe: bool) -> Optional[np.ndarray]:
    """
    Transform depth color_image to point cloud
    Return empty result if transformation failed
    """
    return k4a_module.transformation_depth_image_to_point_cloud(
        calibration.transformation_handle, thread_safe, depth, True
    )


def color_image_to_depth_camera(
    color: np.ndarray, depth: np.ndarray, calibration: Calibration, thread_safe: bool
) -> Optional[np.ndarray]:
    """
    Transform color color_image to depth color_image space
    Return empty result if transformation failed
    """
    return k4a_module.transformation_color_image_to_depth_camera(
        calibration.transformation_handle, thread_safe, depth, color
    )
