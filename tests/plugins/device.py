from dataclasses import dataclass
from typing import Any, Mapping

import pytest

from pyk4a.pyk4a import Result


@dataclass(frozen=True)
class DeviceMeta:
    id: int


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

    def _device_open(device_id: int, thread_safe: bool):
        if device_id not in DEVICE_METAS:
            return Result.Failed.value, None
        capsule = DeviceHandle(device_id)
        return Result.Success.value, capsule

    def _device_close(capsule: DeviceHandle, thread_safe: bool):
        capsule.close()
        return Result.Success

    monkeypatch.setattr("k4a_module.device_open", _device_open)
    monkeypatch.setattr("k4a_module.device_close", _device_close)


@pytest.fixture()
def device_id_good(patch_module_device: Any) -> int:
    return 0


@pytest.fixture()
def device_id_not_exists(patch_module_device: Any) -> int:
    return 99
