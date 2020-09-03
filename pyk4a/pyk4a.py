import sys
from enum import Enum
from typing import Any, Optional, Tuple

import numpy as np

import k4a_module
from pyk4a.config import ColorControlCommand, ColorControlMode, ColorFormat, Config


if sys.version_info < (3, 8):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict


# k4a_wait_result_t
class Result(Enum):
    Success = 0
    Failed = 1
    Timeout = 2


class K4AException(Exception):
    pass


class K4ATimeoutException(K4AException):
    pass


class PyK4A:
    TIMEOUT_WAIT_INFINITE = -1

    def __init__(self, config: Optional[Config] = None, device_id: int = 0, thread_safe: bool = True):
        self._device_id = device_id
        self._config: Config = config if (config is not None) else Config()
        self.thread_safe = thread_safe
        self._device_handle: Optional[object] = None
        self.is_running = False

    def __del__(self):
        if self.is_running:
            self.disconnect()
        elif self.opened:
            self.close()

    @property
    def opened(self) -> bool:
        return self._device_handle is not None

    def open(self):
        """
        Open device
        You must open device before querying any information
        """
        if self.opened:
            raise K4AException("Device already opened")
        self._device_open()

    def close(self):
        """
        Close device
        """
        self._validate_is_opened()
        self._device_close()

    def connect(self):
        """
        Open device if device not opened, then start cameras and IMU
        All-in-one function
        :return:
        """
        if not self.opened:
            self.open()
        self._start_cameras()
        self._start_imu()
        self.is_running = True

    def disconnect(self):
        """
        Stop cameras, IMU, ... and close device
        :return:
        """
        self._stop_imu()
        self._stop_cameras()
        self._device_close()
        self.is_running = False

    def save_calibration_json(self, path: Any):
        calibration = k4a_module.device_get_calibration(self._device_id, self.thread_safe)
        with open(path, "w") as f:
            f.write(calibration)

    def load_calibration_json(self, path: Any):
        with open(path, "r") as f:
            calibration = f.read()
        res = k4a_module.calibration_set_from_raw(
            self._device_id, self.thread_safe, calibration, *self._config.unpack()
        )
        self._verify_error(res)

    def _device_open(self):
        res, handle = k4a_module.device_open(self._device_id, self.thread_safe)
        self._verify_error(res)
        self._device_handle = handle

    def _device_close(self):
        res = k4a_module.device_close(self._device_handle, self.thread_safe)
        self._verify_error(res)
        self._device_handle = None

    def _start_cameras(self):
        res = k4a_module.device_start_cameras(self._device_handle, self.thread_safe, *self._config.unpack())
        self._verify_error(res)

    def _start_imu(self):
        res = k4a_module.device_start_imu(self._device_handle, self.thread_safe)
        self._verify_error(res)

    def _stop_cameras(self):
        res = k4a_module.device_stop_cameras(self._device_handle, self.thread_safe)
        self._verify_error(res)

    def _stop_imu(self):
        res = k4a_module.device_stop_imu(self._device_handle, self.thread_safe)
        self._verify_error(res)

    def get_capture(self, timeout=TIMEOUT_WAIT_INFINITE,) -> "PyK4ACapture":
        """
        Fetch a capture from the device and return a PyK4ACapture object. Images are
        lazily fetched.

        Arguments:
            :param timeout: Timeout in ms. Default is infinite.

        Returns:
            :return capture containing requested images and infos if they are available
                in the current capture. There are no guarantees that the returned
                object will contain all the requested images.

        If using any ColorFormat other than ColorFormat.BGRA32, the color image must be
        decoded. See example/color_formats.py
        """
        res, capture_capsule = k4a_module.device_get_capture(self._device_id, self.thread_safe, timeout)
        self._verify_error(res)

        capture = PyK4ACapture(device=self, capture_capsule=capture_capsule)
        return capture

    def get_imu_sample(self, timeout: int = TIMEOUT_WAIT_INFINITE) -> Optional["ImuSample"]:
        res, imu_sample = k4a_module.device_get_imu_sample(self._device_id, self.thread_safe, timeout)
        self._verify_error(res)
        return imu_sample

    @property
    def sync_jack_status(self) -> Tuple[bool, bool]:
        self._validate_is_opened()
        res, jack_in, jack_out = k4a_module.device_get_sync_jack(self._device_handle, self.thread_safe)
        self._verify_error(res)
        return jack_in == 1, jack_out == 1

    def _get_color_control(self, cmd: ColorControlCommand) -> Tuple[int, ColorControlMode]:
        self._validate_is_opened()
        res, mode, value = k4a_module.device_get_color_control(self._device_handle, self.thread_safe, cmd)
        self._verify_error(res)
        return value, ColorControlMode(mode)

    def _set_color_control(self, cmd: ColorControlCommand, value: int, mode=ColorControlMode.MANUAL):
        self._validate_is_opened()
        res = k4a_module.device_set_color_control(self._device_handle, self.thread_safe, cmd, mode, value)
        self._verify_error(res)

    @property
    def brightness(self) -> int:
        return self._get_color_control(ColorControlCommand.BRIGHTNESS)[0]

    @brightness.setter
    def brightness(self, value: int):
        self._set_color_control(ColorControlCommand.BRIGHTNESS, value)

    @property
    def contrast(self) -> int:
        return self._get_color_control(ColorControlCommand.CONTRAST)[0]

    @contrast.setter
    def contrast(self, value: int):
        self._set_color_control(ColorControlCommand.CONTRAST, value)

    @property
    def saturation(self) -> int:
        return self._get_color_control(ColorControlCommand.SATURATION)[0]

    @saturation.setter
    def saturation(self, value: int):
        self._set_color_control(ColorControlCommand.SATURATION, value)

    @property
    def sharpness(self) -> int:
        return self._get_color_control(ColorControlCommand.SHARPNESS)[0]

    @sharpness.setter
    def sharpness(self, value: int):
        self._set_color_control(ColorControlCommand.SHARPNESS, value)

    @property
    def backlight_compensation(self) -> int:
        return self._get_color_control(ColorControlCommand.BACKLIGHT_COMPENSATION)[0]

    @backlight_compensation.setter
    def backlight_compensation(self, value: int):
        self._set_color_control(ColorControlCommand.BACKLIGHT_COMPENSATION, value)

    @property
    def gain(self) -> int:
        return self._get_color_control(ColorControlCommand.GAIN)[0]

    @gain.setter
    def gain(self, value: int):
        self._set_color_control(ColorControlCommand.GAIN, value)

    @property
    def powerline_frequency(self) -> int:
        return self._get_color_control(ColorControlCommand.POWERLINE_FREQUENCY)[0]

    @powerline_frequency.setter
    def powerline_frequency(self, value: int):
        self._set_color_control(ColorControlCommand.POWERLINE_FREQUENCY, value)

    @property
    def exposure(self) -> int:
        # sets mode to manual
        return self._get_color_control(ColorControlCommand.EXPOSURE_TIME_ABSOLUTE)[0]

    @exposure.setter
    def exposure(self, value: int):
        self._set_color_control(ColorControlCommand.EXPOSURE_TIME_ABSOLUTE, value)

    @property
    def exposure_mode_auto(self) -> bool:
        return self._get_color_control(ColorControlCommand.EXPOSURE_TIME_ABSOLUTE)[1] == ColorControlMode.AUTO

    @exposure_mode_auto.setter
    def exposure_mode_auto(self, mode_auto: bool, value: int = 2500):
        mode = ColorControlMode.AUTO if mode_auto else ColorControlMode.MANUAL
        self._set_color_control(ColorControlCommand.EXPOSURE_TIME_ABSOLUTE, value=value, mode=mode)

    @property
    def whitebalance(self) -> int:
        # sets mode to manual
        return self._get_color_control(ColorControlCommand.WHITEBALANCE)[0]

    @whitebalance.setter
    def whitebalance(self, value: int):
        self._set_color_control(ColorControlCommand.WHITEBALANCE, value)

    @property
    def whitebalance_mode_auto(self) -> bool:
        return self._get_color_control(ColorControlCommand.WHITEBALANCE)[1] == ColorControlMode.AUTO

    @whitebalance_mode_auto.setter
    def whitebalance_mode_auto(self, mode_auto: bool, value: int = 2500):
        mode = ColorControlMode.AUTO if mode_auto else ColorControlMode.MANUAL
        self._set_color_control(ColorControlCommand.WHITEBALANCE, value=value, mode=mode)

    def _get_color_control_capabilities(self, cmd: ColorControlCommand) -> Optional["ColorControlCapabilities"]:
        self._validate_is_opened()
        res, capabilities = k4a_module.device_get_color_control_capabilities(self._device_handle, self.thread_safe, cmd)
        self._verify_error(res)
        return capabilities

    def reset_color_control_to_default(self):
        for cmd in ColorControlCommand:
            capability = self._get_color_control_capabilities(cmd)
            self._set_color_control(cmd, capability["default_value"], capability["default_mode"])

    @staticmethod
    def _verify_error(res):
        res = Result(res)
        if res == Result.Failed:
            raise K4AException()
        elif res == Result.Timeout:
            raise K4ATimeoutException()

    def _validate_is_opened(self):
        if not self.opened:
            raise K4AException("Device is not opened")


class PyK4ACapture:
    def __init__(self, device: PyK4A, capture_capsule: object):
        # capture is a PyCapsule containing pointer to k4a_capture_t.
        # use properties instead of attributes
        self.device: PyK4A = device
        self._color: Optional[np.ndarray] = None
        self._depth: Optional[np.ndarray] = None
        self._ir: Optional[np.ndarray] = None
        self._depth_point_cloud: Optional[np.ndarray] = None
        self._transformed_depth: Optional[np.ndarray] = None
        self._transformed_depth_point_cloud: Optional[np.ndarray] = None
        self._transformed_color: Optional[np.ndarray] = None
        self._cap: object = capture_capsule  # built-in PyCapsule

    @property
    def color(self) -> Optional[np.ndarray]:
        if self._color is None:
            self._color = k4a_module.capture_get_color_image(self.device.thread_safe, self._cap)
        return self._color

    @property
    def ir(self) -> Optional[np.ndarray]:
        if self._ir is None:
            self._ir = k4a_module.capture_get_ir_image(self.device.thread_safe, self._cap)
        return self._ir

    @property
    def depth(self) -> Optional[np.ndarray]:
        if self._depth is None:
            self._depth = k4a_module.capture_get_depth_image(self.device.thread_safe, self._cap)
        return self._depth

    @property
    def transformed_depth(self) -> Optional[np.ndarray]:
        if self._transformed_depth is None and self.depth is not None:
            self._transformed_depth = k4a_module.transformation_depth_image_to_color_camera(
                self.device._device_id, self.device.thread_safe, self.depth, self.device._config.color_resolution,
            )
        return self._transformed_depth

    @property
    def depth_point_cloud(self) -> Optional[np.ndarray]:
        if self._depth_point_cloud is None and self.depth is not None:
            self._depth_point_cloud = k4a_module.transformation_depth_image_to_point_cloud(
                self.device._device_id, self.device.thread_safe, self.depth, True
            )
        return self._depth_point_cloud

    @property
    def transformed_depth_point_cloud(self) -> Optional[np.ndarray]:
        if self._transformed_depth_point_cloud is None and self.transformed_depth is not None:
            self._transformed_depth_point_cloud = k4a_module.transformation_depth_image_to_point_cloud(
                self.device._device_id, self.device.thread_safe, self.transformed_depth, False
            )
        return self._transformed_depth_point_cloud

    @property
    def transformed_color(self) -> Optional[np.ndarray]:
        if self._transformed_color is None and self.depth is not None and self.color is not None:
            if self.device._config.color_format != ColorFormat.BGRA32:
                raise RuntimeError(
                    "color image must be of format K4A_IMAGE_FORMAT_COLOR_BGRA32 for "
                    "transformation_color_image_to_depth_camera"
                )
            self._transformed_color = k4a_module.transformation_color_image_to_depth_camera(
                self.device._device_id, self.device.thread_safe, self.depth, self.color
            )
        return self._transformed_color


class ImuSample(TypedDict):
    temperature: float
    acc_sample: Tuple[float, float, float]
    acc_timestamp: int
    gyro_sample: Tuple[float, float, float]
    gyro_timestamp: int


class ColorControlCapabilities(TypedDict):
    color_control_command: ColorControlCommand
    supports_auto: bool
    min_value: int
    max_value: int
    step_value: int
    default_value: int
    default_mode: ColorControlMode
