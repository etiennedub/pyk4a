from dataclasses import dataclass
from typing import Any, Mapping, Tuple

import pytest

from pyk4a.pyk4a import Result


@dataclass(frozen=True)
class DeviceMeta:
    id: int
    jack_in: bool = False
    jack_out: bool = False


DEVICE_METAS: Mapping[int, DeviceMeta] = {0: DeviceMeta(id=0)}


@pytest.fixture()
def patch_module_device(monkeypatch):
    class DeviceHandle:
        def __init__(self, device_id: int):
            self._meta: DeviceMeta = DEVICE_METAS[device_id]
            self._opened = True

        def close(self) -> int:
            assert self._opened
            self._opened = False
            return Result.Success.value

        def get_sync_jack(self) -> Tuple[bool, bool]:
            return Result.Success.value, self._meta.jack_in, self._meta.jack_out

        def device_get_color_control(self, cmd: int):
            return Result.Success.value, 1, 123  # Why not?

        def device_set_color_control(self, cmd: int, value: int, mode: int):
            return Result.Success.value

    def _device_open(device_id: int, thread_safe: bool):
        if device_id not in DEVICE_METAS:
            return Result.Failed.value, None
        capsule = DeviceHandle(device_id)
        return Result.Success.value, capsule

    def _device_close(capsule: DeviceHandle, thread_safe: bool):
        capsule.close()
        return Result.Success

    def _device_get_sync_jack(capsule: DeviceHandle, thread_safe: bool):
        return capsule.get_sync_jack()

    def _device_get_color_control(capsule: DeviceHandle, thread_safe: bool, cmd: int):
        return capsule.device_get_color_control(cmd)

    def _device_set_color_control(capsule: DeviceHandle, thread_safe: bool, cmd: int, value: int, mode: int):
        return capsule.device_set_color_control(cmd, value, mode)

    monkeypatch.setattr("k4a_module.device_open", _device_open)
    monkeypatch.setattr("k4a_module.device_close", _device_close)
    monkeypatch.setattr("k4a_module.device_get_sync_jack", _device_get_sync_jack)
    monkeypatch.setattr("k4a_module.device_get_color_control", _device_get_color_control)
    monkeypatch.setattr("k4a_module.device_set_color_control", _device_set_color_control)


@pytest.fixture()
def device_id_good(patch_module_device: Any) -> int:
    return 0


@pytest.fixture()
def device_id_not_exists(patch_module_device: Any) -> int:
    return 99
