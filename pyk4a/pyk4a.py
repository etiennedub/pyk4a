from typing import Tuple, Optional
import k4a_module
from enum import Enum
import numpy as np
from dataclasses import dataclass
from typing import Union

from pyk4a.config import Config, ColorControlMode, ColorControlCommand, ColorFormat


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

    def __init__(self, config=None, device_id=0, thread_safe: bool = True):
        self._device_id = device_id
        self._config = config if (config is not None) else Config()
        self._thread_safe = thread_safe
        self.is_running = False

    def __del__(self):
        if self.is_running:
            self.disconnect()

    @property
    def thread_safe(self) -> bool:
        return self._thread_safe

    def connect(self):
        self._device_open()
        self._start_cameras()
        self.is_running = True

    def disconnect(self):
        self._stop_imu()
        self._stop_cameras()
        self._device_close()
        self.is_running = False

    def save_calibration_json(self, path):
        calibration = k4a_module.device_get_calibration(self._device_id, self._thread_safe)
        with open(path, 'w') as f:
            f.write(calibration)

    def load_calibration_json(self, path):
        with open(path, 'r') as f:
            calibration = f.read()
        res = k4a_module.calibration_set_from_raw(self._device_id, self._thread_safe, calibration, *self._config.unpack())
        self._verify_error(res)

    def _device_open(self):
        thread_safe = 1 if self._thread_safe else 0
        res = k4a_module.device_open(self._device_id, self._thread_safe)
        self._verify_error(res)

    def _device_close(self):
        res = k4a_module.device_close(self._device_id, self._thread_safe)
        self._verify_error(res)

    def _start_cameras(self):
        res = k4a_module.device_start_cameras(self._device_id, self._thread_safe, *self._config.unpack())
        self._verify_error(res)

    def _start_imu(self):
        res = k4a_module.device_start_imu(self._device_id, self._thread_safe)
        self._verify_error(res)

    def _stop_cameras(self):
        res = k4a_module.device_stop_cameras(self._device_id, self._thread_safe)
        self._verify_error(res)

    def _stop_imu(self):
        res = k4a_module.device_stop_imu(self._device_id, self._thread_safe)
        self._verify_error(res)

    def get_capture(self, timeout=TIMEOUT_WAIT_INFINITE, ):
        r"""Fetch a capture from the device and return a PyK4ACapture object. Images are lazily fetched.

        Arguments:
            :param capture_request: CaptureRequest containing requested images and information to be fetched if
                they are available in the current frame.
            :param timeout: Timeout in ms. Default is infinite.

        Returns:
            :return CaptureResult containing requested images and infos if they are available in the current
                capture. There are no guarantees that the returned object will contain all the requested images.

        See default request in CaptureRequest
        If using any ColorFormat other than ColorFormat.BGRA32, the color image must be decoded.
            See example/color_formats.py
        """
        res, capture_capsule = k4a_module.device_get_capture(self._device_id, self._thread_safe, timeout)
        self._verify_error(res)

        capture = PyK4ACapture(device=self, capture_capsule=capture_capsule)
        return capture

    def get_imu_sample(self, timeout=TIMEOUT_WAIT_INFINITE):
        res, imu_sample = k4a_module.device_get_imu_sample(self._device_id, self._thread_safe, PyK4A.TIMEOUT_WAIT_INFINITE)
        self._verify_error(res)
        (temperature, acc_sample, acc_timestamp, gyro_sample, gyro_timestamp) = imu_sample
        return {
            "temperature": temperature,
            "acc_sample": acc_sample,
            "acc_timestamp": acc_timestamp,
            "gyro_sample": gyro_sample,
            "gyro_timestamp": gyro_timestamp
        }

    @property
    def sync_jack_status(self) -> Tuple[bool, bool]:
        res, jack_in, jack_out = k4a_module.device_get_sync_jack(self._device_id, self._thread_safe)
        self._verify_error(res)
        return jack_in == 1, jack_out == 1

    def _get_color_control(self, cmd: ColorControlCommand) -> Tuple[int, ColorControlMode]:
        res, mode, value = k4a_module.device_get_color_control(self._device_id, self._thread_safe, cmd)
        self._verify_error(res)
        return value, ColorControlMode(mode)

    def _set_color_control(self, cmd: ColorControlCommand, value: int, mode=ColorControlMode.MANUAL):
        res = k4a_module.device_set_color_control(self._device_id, self._thread_safe, cmd, mode, value)
        self._verify_error(res)

    @property
    def brightness(self) -> int:
        return self._get_color_control(ColorControlCommand.BRIGHTNESS)[0]

    @property
    def contrast(self) -> int:
        return self._get_color_control(ColorControlCommand.CONTRAST)[0]

    @property
    def saturation(self) -> int:
        return self._get_color_control(ColorControlCommand.SATURATION)[0]

    @property
    def sharpness(self) -> int:
        return self._get_color_control(ColorControlCommand.SHARPNESS)[0]

    @property
    def backlight_compensation(self) -> int:
        return self._get_color_control(ColorControlCommand.BACKLIGHT_COMPENSATION)[0]

    @property
    def gain(self) -> int:
        return self._get_color_control(ColorControlCommand.GAIN)[0]

    @property
    def powerline_frequency(self) -> int:
        return self._get_color_control(ColorControlCommand.POWERLINE_FREQUENCY)[0]

    @property
    def exposure(self) -> int:
        # sets mode to manual
        return self._get_color_control(ColorControlCommand.EXPOSURE_TIME_ABSOLUTE)[0]

    @property
    def exposure_mode_auto(self) -> bool:
        return self._get_color_control(ColorControlCommand.EXPOSURE_TIME_ABSOLUTE)[1] == ColorControlMode.AUTO

    @property
    def whitebalance(self) -> int:
        # sets mode to manual
        return self._get_color_control(ColorControlCommand.WHITEBALANCE)[0]

    @property
    def whitebalance_mode_auto(self) -> bool:
        return self._get_color_control(ColorControlCommand.WHITEBALANCE)[1] == ColorControlMode.AUTO

    @brightness.setter
    def brightness(self, value: int):
        self._set_color_control(ColorControlCommand.BRIGHTNESS, value)

    @contrast.setter
    def contrast(self, value: int):
        self._set_color_control(ColorControlCommand.CONTRAST, value)

    @saturation.setter
    def saturation(self, value: int):
        self._set_color_control(ColorControlCommand.SATURATION, value)

    @sharpness.setter
    def sharpness(self, value: int):
        self._set_color_control(ColorControlCommand.SHARPNESS, value)

    @backlight_compensation.setter
    def backlight_compensation(self, value: int):
        self._set_color_control(ColorControlCommand.BACKLIGHT_COMPENSATION, value)

    @gain.setter
    def gain(self, value: int):
        self._set_color_control(ColorControlCommand.GAIN, value)

    @powerline_frequency.setter
    def powerline_frequency(self, value: int):
        self._set_color_control(ColorControlCommand.POWERLINE_FREQUENCY, value)

    @exposure.setter
    def exposure(self, value: int):
        self._set_color_control(ColorControlCommand.EXPOSURE_TIME_ABSOLUTE, value)

    @exposure_mode_auto.setter
    def exposure_mode_auto(self, mode_auto: bool, value=2500):
        mode = ColorControlMode.AUTO if mode_auto else ColorControlMode.MANUAL
        self._set_color_control(ColorControlCommand.EXPOSURE_TIME_ABSOLUTE, value=value, mode=mode)

    @whitebalance.setter
    def whitebalance(self, value: int):
        self._set_color_control(ColorControlCommand.WHITEBALANCE, value)

    @whitebalance_mode_auto.setter
    def whitebalance_mode_auto(self, mode_auto: bool, value=2500):
        mode = ColorControlMode.AUTO if mode_auto else ColorControlMode.MANUAL
        self._set_color_control(ColorControlCommand.WHITEBALANCE, value=value, mode=mode)

    def _get_color_control_capabilities(self, cmd: ColorControlCommand) -> (bool, int, int, int, int, int):
        ret = k4a_module.device_get_color_control_capabilities(self._device_id, self._thread_safe, cmd)
        (res, supports_auto, min_value, max_value, step_value, default_value, default_mode) = ret
        self._verify_error(res)
        return {
            "color_control_command": cmd,
            "supports_auto": supports_auto == 1,
            "min_value": min_value,
            "max_value": max_value,
            "step_value": step_value,
            "default_value": default_value,
            "default_mode": default_mode,
        }

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


class PyK4ACapture:
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
                self.device._device_id, self.device.thread_safe, self.depth, self.device._config.color_resolution, )
        return self._transformed_depth

    @property
    def transformed_color(self) -> Optional[np.ndarray]:
        if self._transformed_color is None and self.depth is not None and self.color is not None:
            if self.device._config.color_format != ColorFormat.BGRA32:
                raise RuntimeError("color image must be of format K4A_IMAGE_FORMAT_COLOR_BGRA32 for "
                                   "transformation_color_image_to_depth_camera")
            self._transformed_color = k4a_module.transformation_color_image_to_depth_camera(
                self.device._device_id, self.device.thread_safe, self.depth, self.color
            )
        return self._transformed_color

    def __init__(self, device: PyK4A, capture_capsule: object):
        # capture is a PyCapsule containing pointer to k4a_capture_t.
        # use properties instead of attributes
        self.device: PyK4A = device
        self._color: Optional[np.ndarray] = None
        self._depth: Optional[np.ndarray] = None
        self._ir: Optional[np.ndarray] = None
        self._transformed_depth: Optional[np.ndarray] = None
        self._transformed_color: Optional[np.ndarray] = None
        self._cap: object = capture_capsule  # built-in PyCapsule
