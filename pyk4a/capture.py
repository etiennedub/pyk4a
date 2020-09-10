from typing import Optional

import numpy as np

import k4a_module

from .calibration import Calibration
from .config import ImageFormat
from .transformation import color_image_to_depth_camera, depth_image_to_color_camera, depth_image_to_point_cloud


class PyK4ACapture:
    def __init__(
        self, calibration: Calibration, capture_handle: object, color_format: ImageFormat, thread_safe: bool = True
    ):
        self._calibration: Calibration = calibration
        self._capture_handle: object = capture_handle  # built-in PyCapsule
        self.thread_safe = thread_safe
        self._color_format = color_format

        self._color: Optional[np.ndarray] = None
        self._depth: Optional[np.ndarray] = None
        self._ir: Optional[np.ndarray] = None
        self._depth_point_cloud: Optional[np.ndarray] = None
        self._transformed_depth: Optional[np.ndarray] = None
        self._transformed_depth_point_cloud: Optional[np.ndarray] = None
        self._transformed_color: Optional[np.ndarray] = None

    @property
    def color(self) -> Optional[np.ndarray]:
        if self._color is None:
            self._color = k4a_module.capture_get_color_image(self._capture_handle, self.thread_safe)
        return self._color

    @property
    def depth(self) -> Optional[np.ndarray]:
        if self._depth is None:
            self._depth = k4a_module.capture_get_depth_image(self._capture_handle, self.thread_safe)
        return self._depth

    @property
    def ir(self) -> Optional[np.ndarray]:
        if self._ir is None:
            self._ir = k4a_module.capture_get_ir_image(self._capture_handle, self.thread_safe)
        return self._ir

    @property
    def transformed_depth(self) -> Optional[np.ndarray]:
        if self._transformed_depth is None and self.depth is not None:
            self._transformed_depth = depth_image_to_color_camera(self._depth, self._calibration, self.thread_safe)
        return self._transformed_depth

    @property
    def depth_point_cloud(self) -> Optional[np.ndarray]:
        if self._depth_point_cloud is None and self.depth is not None:
            self._depth_point_cloud = depth_image_to_point_cloud(self._depth, self._calibration, self.thread_safe)
        return self._depth_point_cloud

    @property
    def transformed_depth_point_cloud(self) -> Optional[np.ndarray]:
        if self._transformed_depth_point_cloud is None and self.transformed_depth is not None:
            self._depth_point_cloud = depth_image_to_point_cloud(
                self.transformed_depth, self._calibration, self.thread_safe
            )
        return self._transformed_depth_point_cloud

    @property
    def transformed_color(self) -> Optional[np.ndarray]:
        if self._transformed_color is None and self.depth is not None and self.color is not None:
            if self._color_format != ImageFormat.COLOR_BGRA32:
                raise RuntimeError(
                    "color color_image must be of color_format K4A_IMAGE_FORMAT_COLOR_BGRA32 for "
                    "transformation_color_image_to_depth_camera"
                )
            self._transformed_color = color_image_to_depth_camera(
                self.color, self.depth, self._calibration, self.thread_safe
            )
        return self._transformed_color
