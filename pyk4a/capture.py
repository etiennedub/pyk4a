from typing import Optional

import numpy as np

import k4a_module

from .calibration import Calibration
from .config import ImageFormat
from .transformation import color_image_to_depth_camera, depth_image_to_color_camera, depth_image_to_point_cloud


class PyK4ACapture:
    def __init__(
        self, calibration: Calibration, capture_handle: object, color_format: ImageFormat, thread_safe: bool = True,
        body_tracker: Optional[object] = None
    ):
        self._calibration: Calibration = calibration
        self._capture_handle: object = capture_handle  # built-in PyCapsule
        self._body_tracker: Optional[None] = body_tracker # built-in PyCapsule
        self.thread_safe = thread_safe
        self._color_format = color_format

        self._color: Optional[np.ndarray] = None
        self._depth: Optional[np.ndarray] = None
        self._ir: Optional[np.ndarray] = None
        self._depth_point_cloud: Optional[np.ndarray] = None
        self._transformed_depth: Optional[np.ndarray] = None
        self._transformed_depth_point_cloud: Optional[np.ndarray] = None
        self._transformed_color: Optional[np.ndarray] = None
        self._body_skeleton: Optional[np.ndarray] = None
        self._body_index_map: Optional[np.ndarray] = None

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

    @property
    def body_skeleton(self) -> Optional[np.ndarray]:
        """
        np array of floats
        (n_bodies, n_joints, n_data) == body_skeleton.shape

        data for a joint follows this order(
            position_x,
            position_y,
            position_z,
            orientation_w,
            orientation_x,
            orientation_y,
            orientation_z,
            confidence_level,
            position_image_0,
            position_image_1,
        )
        """
        assert self._body_tracker is not None
        if self._body_skeleton is None:
            self._body_skeleton, self._body_index_map = k4a_module.capture_get_body_tracking(
                self._body_tracker, self._capture_handle, self.thread_safe
            )
        return self._body_skeleton

    @property
    def body_index_map(self) -> Optional[np.ndarray]:
        assert self._body_tracker is not None
        if self.body_index_map is None:
            self._body_skeleton, self._body_index_map = k4a_module.capture_get_body_tracking(
                self._body_tracker, self._capture_handle, self.thread_safe
            )
        return self._body_index_map
