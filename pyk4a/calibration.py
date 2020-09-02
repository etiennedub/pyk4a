from enum import IntEnum
from typing import Optional, Tuple

import k4a_module
from pyk4a.config import Config

from .pyk4a import K4AException, K4ATimeoutException, PyK4A, Result


class CalibrationType(IntEnum):
    UNKNOWN = -1  # Unknown
    DEPTH = 0  # Depth Camera
    COLOR = 1  # Color Sensor
    GYRO = 2  # Gyroscope
    ACCEL = 3  # Accelerometer
    NUM = 4  # Number of types excluding unknown type


class Calibration:
    def __init__(
        self, device: PyK4A, config: Config, source_calibration: CalibrationType, target_calibration: CalibrationType,
    ):
        if isinstance(device, PyK4A):
            self.device = device
        else:
            raise K4AException("Calibration instance created without a device of the proper class")
        self.source_calibration = source_calibration
        self.target_calibration = target_calibration
        self.config = config

    def convert_3d_to_3d(
        self,
        source_point_3d: Tuple[float, float, float],
        source_camera: Optional[CalibrationType] = None,
        target_camera: Optional[CalibrationType] = None,
    ) -> Tuple[float, float, float]:
        """
        Transform a 3d point of a source coordinate system into a 3d
        point of the target coordinate system.
        """
        # Device needs to be running for the functions to work
        if self.device.is_running:
            if source_camera is None:
                source_camera = self.source_calibration
            if target_camera is None:
                target_camera = self.target_calibration
            res, target_point_3d = k4a_module.calibration_3d_to_3d(
                self.device._device_id, self.device.thread_safe, source_point_3d, source_camera, target_camera,
            )

            self._verify_error(res)
            return target_point_3d
        else:
            raise K4AException("Device not running. Please connect to the device (device.connect())")

    def convert_2d_to_3d(
        self,
        source_pixel_2d: Tuple[float, float, float],
        depth: float,
        source_camera: Optional[CalibrationType] = None,
        target_camera: Optional[CalibrationType] = None,
    ) -> Tuple[int, Tuple[float, float, float]]:
        """
        Transform a 2d pixel to a 3d point of the target coordinate system.
        """
        # Device needs to be running for the functions to work
        if self.device.is_running:
            if source_camera is None:
                source_camera = self.source_calibration
            if target_camera is None:
                target_camera = self.target_calibration
            res, valid, target_point_3d = k4a_module.calibration_2d_to_3d(
                self.device._device_id, self.device.thread_safe, source_pixel_2d, depth, source_camera, target_camera,
            )
            self._verify_error(res)
            return valid, target_point_3d
        else:
            raise K4AException("Device not running. Please connect to the device (device.connect())")

    @staticmethod
    def _verify_error(res) -> None:
        res = Result(res)
        if res == Result.Failed:
            raise K4AException()
        elif res == Result.Timeout:
            raise K4ATimeoutException()
