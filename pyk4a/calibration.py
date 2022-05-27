from enum import IntEnum
from typing import Optional, Tuple

import numpy as np

from .config import ColorResolution, DepthMode
from .errors import K4AException, _verify_error
from .module import k4a_module


class CalibrationType(IntEnum):
    UNKNOWN = -1  # Unknown
    DEPTH = 0  # Depth Camera
    COLOR = 1  # Color Sensor
    GYRO = 2  # Gyroscope
    ACCEL = 3  # Accelerometer
    NUM = 4  # Number of types excluding unknown type


class Calibration:
    def __init__(
        self, handle: object, depth_mode: DepthMode, color_resolution: ColorResolution, thread_safe: bool = True
    ):
        self._calibration_handle = handle
        self._transformation_handle: Optional[object] = None
        self.thread_safe = thread_safe
        self._depth_mode = depth_mode
        self._color_resolution = color_resolution
        self._raw: Optional[str] = None

    @classmethod
    def from_raw(
        cls, value: str, depth_mode: DepthMode, color_resolution: ColorResolution, thread_safe: bool = True
    ) -> "Calibration":
        res, handle = k4a_module.calibration_get_from_raw(thread_safe, value, depth_mode, color_resolution)
        _verify_error(res)
        return Calibration(
            handle=handle, depth_mode=depth_mode, color_resolution=color_resolution, thread_safe=thread_safe
        )

    @property
    def depth_mode(self) -> DepthMode:
        return self._depth_mode

    @property
    def color_resolution(self) -> ColorResolution:
        return self._color_resolution

    def _convert_3d_to_3d(
        self,
        source_point_3d: Tuple[float, float, float],
        source_camera: CalibrationType,
        target_camera: CalibrationType,
    ) -> Tuple[float, float, float]:
        """
        Transform a 3d point of a source coordinate system into a 3d
        point of the target coordinate system.
        :param source_point_3d  The 3D coordinates in millimeters representing a point in source_camera.
        :param source_camera    The current camera.
        :param target_camera    The target camera.
        :return                 The 3D coordinates in millimeters representing a point in target camera.
        """
        res, target_point_3d = k4a_module.calibration_3d_to_3d(
            self._calibration_handle,
            self.thread_safe,
            source_point_3d,
            source_camera,
            target_camera,
        )

        _verify_error(res)
        return target_point_3d

    def depth_to_color_3d(self, point_3d: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return self._convert_3d_to_3d(point_3d, CalibrationType.DEPTH, CalibrationType.COLOR)

    def color_to_depth_3d(self, point_3d: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return self._convert_3d_to_3d(point_3d, CalibrationType.COLOR, CalibrationType.DEPTH)

    def _convert_2d_to_3d(
        self,
        source_pixel_2d: Tuple[float, float],
        source_depth: float,
        source_camera: CalibrationType,
        target_camera: CalibrationType,
    ) -> Tuple[float, float, float]:
        """
        Transform a 3d point of a source coordinate system into a 3d
        point of the target coordinate system.
        :param source_pixel_2d    The 2D coordinates in px of source_camera color_image.
        :param source_depth       Depth in mm
        :param source_camera      The current camera.
        :param target_camera      The target camera.
        :return                   The 3D coordinates in mm representing a point in target camera.
        """
        res, valid, target_point_3d = k4a_module.calibration_2d_to_3d(
            self._calibration_handle,
            self.thread_safe,
            source_pixel_2d,
            source_depth,
            source_camera,
            target_camera,
        )

        _verify_error(res)
        if valid == 0:
            raise ValueError(f"Coordinates {source_pixel_2d} are not valid in the calibration model")

        return target_point_3d

    def convert_2d_to_3d(
        self,
        coordinates: Tuple[float, float],
        depth: float,
        source_camera: CalibrationType,
        target_camera: Optional[CalibrationType] = None,
    ):
        """
        Transform a 2d pixel to a 3d point of the target coordinate system.
        """
        if target_camera is None:
            target_camera = source_camera
        return self._convert_2d_to_3d(coordinates, depth, source_camera, target_camera)

    def _convert_3d_to_2d(
        self,
        source_point_3d: Tuple[float, float, float],
        source_camera: CalibrationType,
        target_camera: CalibrationType,
    ) -> Tuple[float, float]:
        """
        Transform a 3d point of a source coordinate system into a 3d
        point of the target coordinate system.
        :param source_point_3d    The 3D coordinates in mm of source_camera.
        :param source_camera      The current camera.
        :param target_camera      The target camera.
        :return                   The 3D coordinates in mm representing a point in target camera.
        """
        res, valid, target_px_2d = k4a_module.calibration_3d_to_2d(
            self._calibration_handle,
            self.thread_safe,
            source_point_3d,
            source_camera,
            target_camera,
        )

        _verify_error(res)
        if valid == 0:
            raise ValueError(f"Coordinates {source_point_3d} are not valid in the calibration model")

        return target_px_2d

    def convert_3d_to_2d(
        self,
        coordinates: Tuple[float, float, float],
        source_camera: CalibrationType,
        target_camera: Optional[CalibrationType] = None,
    ):
        """
        Transform a 3d point to a 2d pixel of the target coordinate system.
        """
        if target_camera is None:
            target_camera = source_camera
        return self._convert_3d_to_2d(coordinates, source_camera, target_camera)

    @property
    def transformation_handle(self) -> object:
        if not self._transformation_handle:
            handle = k4a_module.transformation_create(self._calibration_handle, self.thread_safe)
            if not handle:
                raise K4AException("Cannot create transformation handle")
            self._transformation_handle = handle
        return self._transformation_handle

    def get_camera_matrix(self, camera: CalibrationType) -> np.ndarray:
        """
        Get the camera matrix (in OpenCV compatible format) for the color or depth camera
        """
        if camera not in [CalibrationType.COLOR, CalibrationType.DEPTH]:
            raise ValueError("Camera matrix only available for color and depth cameras.")
        params = k4a_module.calibration_get_intrinsics(self._calibration_handle, self.thread_safe, camera)
        if len(params) != 14:
            raise ValueError("Unknown camera calibration type")

        cx, cy, fx, fy = params[:4]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def get_distortion_coefficients(self, camera: CalibrationType) -> np.ndarray:
        """
        Get the distortion coefficients (in OpenCV compatible format) for the color or depth camera
        """
        if camera not in [CalibrationType.COLOR, CalibrationType.DEPTH]:
            raise ValueError("Distortion coefficients only available for color and depth cameras.")
        params = k4a_module.calibration_get_intrinsics(self._calibration_handle, self.thread_safe, camera)
        if len(params) != 14:
            raise ValueError("Unknown camera calibration type")

        return np.array([params[4], params[5], params[13], params[12], *params[6:10]])
