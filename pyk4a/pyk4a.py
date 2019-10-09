from typing import Tuple
import k4a_module
from enum import Enum

from pyk4a.config import Config, ColorControlMode, ColorControlCommand

# k4a_result_t
class Result(Enum):
    Success = 0
    Failed = 1

class PyK4A:
    TIMEOUT_WAIT_INFINITE = -1
    def __init__(self, config : Config, device_id=0):
        self._device_id = 0
        self._config = config
        self.is_running = False

    def __del__(self):
        if self.is_running:
            self.disconnect()

    def connect(self):
        self._device_open()
        self._device_start_cameras()
        self.is_running = True

    def disconnect(self):
        self._device_stop_cameras()
        self._device_close()
        self.is_running = False

    def _device_open(self):
        res = k4a_module.device_open(self._device_id)
        self._verify_error(res)

    def _device_close(self):
        res = k4a_module.device_close()
        self._verify_error(res)

    def _device_start_cameras(self):
        res = k4a_module.device_start_cameras(*self._config.unpack())
        self._verify_error(res)

    def _device_stop_cameras(self):
        res = k4a_module.device_stop_cameras()
        self._verify_error(res)

    def device_get_capture(self, timeout=TIMEOUT_WAIT_INFINITE,
            color_only=False, transform_depth_to_color=True):
        res = k4a_module.device_get_capture(timeout)
        self._verify_error(res)

        color = k4a_module.device_get_color_image()
        if color_only:
            return color

        depth = k4a_module.device_get_depth_image(transform_depth_to_color)

        return color, depth

    def device_get_sync_jack(self):
        res, jack_in, jack_out = k4a_module.device_get_sync_jack()
        self._verify_error(res)
        return jack_in == 1, jack_out == 1

    def _get_color_control(self, cmd: ColorControlCommand) -> Tuple[int, ColorControlMode]:
        res, mode, value = k4a_module.device_get_color_control(cmd)
        self._verify_error(res)
        return value, ColorControlMode(mode)

    def _set_color_control(self, cmd: ColorControlCommand,
                                 value: int,
                                 mode=ColorControlMode.MANUAL):
        res = k4a_module.device_set_color_control(cmd, mode, value)
        self._verify_error(res)

    def set_exposure(self, value: int):
        self._set_color_control(ColorControlCommand.EXPOSURE_TIME_ABSOLUTE, value)

    def set_brightness(self, value: int):
        self._set_color_control(ColorControlCommand.BRIGHTNESS, value)

    def set_contrast(self, value: int):
        self._set_color_control(ColorControlCommand.CONTRAST, value)

    def set_saturation(self, value: int):
        self._set_color_control(ColorControlCommand.SATURATION, value)

    def set_sharpness(self, value: int):
        self._set_color_control(ColorControlCommand.SHARPNESS, value)

    def set_whitebalance(self, value: int, mode=ColorControlMode.MANUAL):
        self._set_color_control(ColorControlCommand.WHITEBALANCE, value, mode=mode)

    def set_backlight_compensation(self, value: int):
        self._set_color_control(ColorControlCommand.BACKLIGHT_COMPENSATION, value)

    def set_gain(self, value: int):
        self._set_color_control(ColorControlCommand.GAIN, value)

    def set_powerline_frequency(self, value: int):
        self._set_color_control(ColorControlCommand.POWERLINE_FREQUENCY, value)

    def get_exposure(self) -> Tuple[int, ColorControlMode]:
        return self._get_color_control(ColorControlCommand.EXPOSURE_TIME_ABSOLUTE)

    def get_brightness(self) -> Tuple[int, ColorControlMode]:
        return self._get_color_control(ColorControlCommand.BRIGHTNESS)

    def get_contrast(self) -> Tuple[int, ColorControlMode]:
        return self._get_color_control(ColorControlCommand.CONTRAST)

    def get_saturation(self) -> Tuple[int, ColorControlMode]:
        return self._get_color_control(ColorControlCommand.SATURATION)

    def get_sharpness(self) -> Tuple[int, ColorControlMode]:
        return self._get_color_control(ColorControlCommand.SHARPNESS)

    def get_whitebalance(self) -> Tuple[int, ColorControlMode]:
        return self._get_color_control(ColorControlCommand.WHITEBALANCE)

    def get_backlight_compensation(self) -> Tuple[int, ColorControlMode]:
        return self._get_color_control(ColorControlCommand.BACKLIGHT_COMPENSATION)

    def get_gain(self) -> Tuple[int, ColorControlMode]:
        return self._get_color_control(ColorControlCommand.GAIN)

    def get_powerline_frequency(self) -> Tuple[int, ColorControlMode]:
        return self._get_color_control(ColorControlCommand.POWERLINE_FREQUENCY)

    @staticmethod
    def _verify_error(res):
        if Result(res) == Result.Failed:
            raise RuntimeError('Function return status : {}'.format(res))

if __name__ == "__main__":
    k4a = PyK4A(Config())
    k4a.connect()
    print("Connected")
    jack_in, jack_out = k4a.device_get_sync_jack()
    print("Jack status : in -> {} , out -> {}".format(jack_in, jack_out))
    for _ in range(10):
        color, depth = k4a.device_get_capture(color_only=False)
        print(color.shape, depth.shape)
    k4a.disconnect()
    print("Disconnected")
