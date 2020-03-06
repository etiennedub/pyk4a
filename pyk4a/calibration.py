from enum import IntEnum
from typing import List, Tuple
import k4a_module
from .pyk4a import Result, K4AException, K4ATimeoutException, PyK4A
from pyk4a.config import Config


class CalibrationType(IntEnum):
    UNKNOWN = -1       # Unknown
    DEPTH = 0          # Depth Camera
    COLOR = 1          # Color Sensor
    GYRO = 2           # Gyroscope
    ACCEL = 3          # Accelerometer
    NUM = 4            # Number of types excluding unknown type


class Calibration:
    def __init__(self, device: PyK4A, config: Config):
        if isinstance(device, PyK4A):
            self.device = device
        else:
            raise K4AException(f'Calibration instance created without '
                               f'a device of the proper class')
        self.config = config

    def convert_3d_to_3d(self,
                         source_point_3d: List,
                         source_camera: CalibrationType,
                         target_camera: CalibrationType) -> Tuple[int, List]:
        """
        Transform a 3d point of a source coordinate system into a 3d
        point of the target coordinate system.
        """
        # Device needs to be running for the functions to work
        if self.device.is_running:
            res, x_targ, y_targ, z_targ = k4a_module.calibration_3d_to_3d(
                source_point_3d[0],
                source_point_3d[1],
                source_point_3d[2],
                source_camera,
                target_camera,
                *self.config.unpack())

            self._verify_error(res)
            return [x_targ, y_targ, z_targ]
        else:
            raise K4AException(f'Device not running. Please connect '
                               f'to the device (device.connect())')

    def convert_2d_to_3d(self,
                         source_pixel_2d: List,
                         depth: float,
                         source_camera: CalibrationType,
                         target_camera: CalibrationType) -> Tuple[int, List]:
        """
        Transform a 2d pixel to a 3d point of the target coordinate system.
        """
        # Device needs to be running for the functions to work
        if self.device.is_running:
            res, valid, x_targ, y_targ, z_targ = k4a_module.calibration_2d_to_3d(
                source_pixel_2d[0],
                source_pixel_2d[1],
                depth,
                source_camera,
                target_camera,
                *self.config.unpack())
            self._verify_error(res)
            return valid, [x_targ, y_targ, z_targ]
        else:
            raise K4AException(f'Device not running. Please connect '
                               f'to the device (device.connect())')

    @staticmethod
    def _verify_error(res):
        res = Result(res)
        if res == Result.Failed:
            raise K4AException()
        elif res == Result.Timeout:
            raise K4ATimeoutException()
