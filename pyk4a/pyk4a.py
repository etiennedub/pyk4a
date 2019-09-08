import k4a_module
from enum import Enum

# k4a_result_t
class Result(Enum):
    Success = 0
    Failed = 1

class PyK4A:
    def device_open(self, device_id=0):
        res, self._device = k4a_module.device_open(device_id)
        self._verify_error(res)

    def device_close(self):
        k4a_module.device_close(self._device)

    def device_get_sync_jack(self):
        res, jack_in, jack_out = k4a_module.device_get_sync_jack(self._device)
        self._verify_error(res)
        return jack_in == 1, jack_out == 1

    @staticmethod
    def _verify_error(res):
        if Result(res) == Result.Failed:
            raise RuntimeError('Function return status : {}'.format(res))

if __name__ == "__main__":
    k4a = PyK4A()
    k4a.device_open()
    print(k4a.device_get_sync_jack())
    # k4a.device_close()
