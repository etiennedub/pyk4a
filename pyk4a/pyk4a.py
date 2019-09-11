import k4a_module
from enum import Enum

from pyk4a.config import Config

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
