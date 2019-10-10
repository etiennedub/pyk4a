from typing import Tuple
import k4a_module
from enum import Enum

from pyk4a.config import Config, ColorControlMode, ColorControlCommand


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

    def __init__(self, config=Config(), device_id=0):
        self._device_id = device_id
        self._config = config
        self.is_running = False

    def __del__(self):
        if self.is_running:
            self.disconnect()

    def connect(self):
        self._device_open()
        self._start_cameras()
        self.is_running = True

    def disconnect(self):
        self._stop_cameras()
        self._device_close()
        self.is_running = False

    def _device_open(self):
        res = k4a_module.device_open(self._device_id)
        self._verify_error(res)

    def _device_close(self):
        res = k4a_module.device_close()
        self._verify_error(res)

    def _start_cameras(self):
        res = k4a_module.device_start_cameras(*self._config.unpack())
        self._verify_error(res)

    def _stop_cameras(self):
        res = k4a_module.device_stop_cameras()
        self._verify_error(res)

    def get_capture(self, timeout=TIMEOUT_WAIT_INFINITE, color_only=False, transform_depth_to_color=True):
        res = k4a_module.device_get_capture(timeout)
        self._verify_error(res)

        color = k4a_module.device_get_color_image()
        if color_only:
            return color

        depth = k4a_module.device_get_depth_image(transform_depth_to_color)

        return color, depth

    @property
    def sync_jack_status(self) -> Tuple[bool, bool]:
        res, jack_in, jack_out = k4a_module.device_get_sync_jack()
        self._verify_error(res)
        return jack_in == 1, jack_out == 1

    def _get_color_control(self, cmd: ColorControlCommand) -> Tuple[int, ColorControlMode]:
        res, mode, value = k4a_module.device_get_color_control(cmd)
        self._verify_error(res)
        return value, ColorControlMode(mode)

    def _set_color_control(self, cmd: ColorControlCommand, value: int, mode=ColorControlMode.MANUAL):
        res = k4a_module.device_set_color_control(cmd, mode, value)
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
    def whitebalance(self, value: int, mode=ColorControlMode.MANUAL):
        self._set_color_control(ColorControlCommand.WHITEBALANCE, value)

    @whitebalance_mode_auto.setter
    def whitebalance_mode_auto(self, mode_auto: bool, value=2500):
        mode = ColorControlMode.AUTO if mode_auto else ColorControlMode.MANUAL
        self._set_color_control(ColorControlCommand.WHITEBALANCE, value=value, mode=mode)

    @staticmethod
    def _verify_error(res):
        res = Result(res)
        if res == Result.Failed:
            raise K4AException()
        elif res == Result.Timeout:
            raise K4ATimeoutException()


if __name__ == "__main__":
    k4a = PyK4A(Config())
    k4a.connect()
    print("Connected")
    jack_in, jack_out = k4a.get_sync_jack()
    print("Jack status : in -> {} , out -> {}".format(jack_in, jack_out))
    for _ in range(10):
        color, depth = k4a.device_get_capture(color_only=False)
        print(color.shape, depth.shape)
    k4a.disconnect()
    print("Disconnected")
