from enum import IntEnum
import k4a_module
import numpy as np
from pyk4a import Result, K4AException, K4ATimeoutException


class CalibrationType(IntEnum):
    CALIBRATION_UNKNOWN = -1       # Unknown
    CALIBRATION_DEPTH = 0          # Depth Camera
    CALIBRATION_COLOR = 1          # Color Sensor
    CALIBRATION_GYRO = 2           # Gyroscope
    CALIBRATION_ACCEL = 3          # Accelerometer
    CALIBRATION_NUM = 4            # Number of types excluding unknown type


class Calibration:
    def __init__(self):
        self.calibration = k4a_module.device_get_calibration()

    def convert_3d_to_3d(self,
                         source_point_3d: np,
                         source_camera,
                         target_camera):
        """
        Transform a 3d point of a source coordinate system into a 3d point of the target coordinate system.
        """
        res = k4a_module.calibration_3d_to_3d(
            self.calibration,
            source_point_3d,
            source_camera,
            target_camera)
        self._verify_error(res)
        return res

    @staticmethod
    def _verify_error(res):
        res = Result(res)
        if res == Result.Failed:
            raise K4AException()
        elif res == Result.Timeout:
            raise K4ATimeoutException()
