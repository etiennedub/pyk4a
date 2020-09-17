from typing import Iterator

import pytest

from pyk4a import K4AException, PyK4A


DEVICE_ID = 0
DEVICE_ID_NOT_EXISTS = 99


@pytest.fixture()
def device_id() -> int:
    return DEVICE_ID


@pytest.fixture()
def device_id_not_exists() -> int:
    return DEVICE_ID_NOT_EXISTS


@pytest.fixture()
def device(device_id: int) -> Iterator[PyK4A]:
    device = PyK4A(device_id=device_id)
    yield device

    if device._device_handle:
        # close all
        try:
            device._stop_imu()
        except K4AException:
            pass
        try:
            device._stop_cameras()
        except K4AException:
            pass
        try:
            device.close()
        except K4AException:
            pass


@pytest.fixture()
def device_not_exists(device_id_not_exists: int) -> Iterator[PyK4A]:
    device = PyK4A(device_id=device_id_not_exists)
    yield device
