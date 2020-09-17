import sys
from typing import Any, Optional, Tuple

import k4a_module

from .calibration import Calibration
from .capture import PyK4ACapture
from .config import ColorControlCommand, ColorControlMode, Config
from .errors import K4AException, _verify_error


if sys.version_info < (3, 8):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict


class PyK4A:
    TIMEOUT_WAIT_INFINITE = -1

    def __init__(self, config: Optional[Config] = None, device_id: int = 0, thread_safe: bool = True):
        self._device_id = device_id
        self._config: Config = config if (config is not None) else Config()
        self.thread_safe = thread_safe
        self._device_handle: Optional[object] = None
        self._calibration: Optional[Calibration] = None
        self.is_running = False

    def start(self):
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

    def stop(self):
        """
        Stop cameras, IMU, ... and close device
        :return:
        """
        self._stop_imu()
        self._stop_cameras()
        self._device_close()
        self.is_running = False

    def __del__(self):
        if self.is_running:
            self.stop()
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
        self._validate_is_opened()
        self._device_close()

    def save_calibration_json(self, path: Any):
        with open(path, "w") as f:
            f.write(self.calibration_raw)

    def load_calibration_json(self, path: Any):
        with open(path, "r") as f:
            calibration = f.read()
        self.calibration_raw = calibration

    def _device_open(self):
        res, handle = k4a_module.device_open(self._device_id, self.thread_safe)
        _verify_error(res)
        self._device_handle = handle

    def _device_close(self):
        res = k4a_module.device_close(self._device_handle, self.thread_safe)
        _verify_error(res)
        self._device_handle = None

    def _start_cameras(self):
        res = k4a_module.device_start_cameras(self._device_handle, self.thread_safe, *self._config.unpack())
        _verify_error(res)

    def _start_imu(self):
        res = k4a_module.device_start_imu(self._device_handle, self.thread_safe)
        _verify_error(res)

    def _stop_cameras(self):
        res = k4a_module.device_stop_cameras(self._device_handle, self.thread_safe)
        _verify_error(res)

    def _stop_imu(self):
        res = k4a_module.device_stop_imu(self._device_handle, self.thread_safe)
        _verify_error(res)

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

        If using any ImageFormat other than ImageFormat.COLOR_BGRA32, the color color_image must be
        decoded. See example/color_formats.py
        """
        self._validate_is_opened()
        res, capture_capsule = k4a_module.device_get_capture(self._device_handle, self.thread_safe, timeout)
        _verify_error(res)

        capture = PyK4ACapture(
            calibration=self.calibration,
            capture_handle=capture_capsule,
            color_format=self._config.color_format,
            thread_safe=self.thread_safe,
        )
        return capture

    def get_imu_sample(self, timeout: int = TIMEOUT_WAIT_INFINITE) -> Optional["ImuSample"]:
        self._validate_is_opened()
        res, imu_sample = k4a_module.device_get_imu_sample(self._device_handle, self.thread_safe, timeout)
        _verify_error(res)
        return imu_sample

    @property
    def calibration_raw(self) -> str:
        self._validate_is_opened()
        raw = k4a_module.device_get_raw_calibration(self._device_handle, self.thread_safe)
        return raw

    @calibration_raw.setter
    def calibration_raw(self, value: str):
        self._validate_is_opened()
        self._calibration = Calibration.from_raw(
            value, self._config.depth_mode, self._config.color_resolution, self.thread_safe
        )

    @property
    def sync_jack_status(self) -> Tuple[bool, bool]:
        self._validate_is_opened()
        res, jack_in, jack_out = k4a_module.device_get_sync_jack(self._device_handle, self.thread_safe)
        _verify_error(res)
        return jack_in == 1, jack_out == 1

    def _get_color_control(self, cmd: ColorControlCommand) -> Tuple[int, ColorControlMode]:
        self._validate_is_opened()
        res, mode, value = k4a_module.device_get_color_control(self._device_handle, self.thread_safe, cmd)
        _verify_error(res)
        return value, ColorControlMode(mode)

    def _set_color_control(self, cmd: ColorControlCommand, value: int, mode=ColorControlMode.MANUAL):
        self._validate_is_opened()
        res = k4a_module.device_set_color_control(self._device_handle, self.thread_safe, cmd, mode, value)
        _verify_error(res)

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
        _verify_error(res)
        return capabilities

    def reset_color_control_to_default(self):
        for cmd in ColorControlCommand:
            capability = self._get_color_control_capabilities(cmd)
            self._set_color_control(cmd, capability["default_value"], capability["default_mode"])

    @property
    def calibration(self) -> Calibration:
        self._validate_is_opened()
        if not self._calibration:
            res, calibration_handle = k4a_module.device_get_calibration(
                self._device_handle, self.thread_safe, self._config.depth_mode, self._config.color_resolution
            )
            _verify_error(res)
            self._calibration = Calibration(
                handle=calibration_handle,
                depth_mode=self._config.depth_mode,
                color_resolution=self._config.color_resolution,
                thread_safe=self.thread_safe,
            )
        return self._calibration

    def _validate_is_opened(self):
        if not self.opened:
            raise K4AException("Device is not opened")


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
