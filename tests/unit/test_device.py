from typing import Iterator

import pytest

from pyk4a import K4AException, PyK4A


DEVICE_ID = 0
DEVICE_ID_NOT_EXISTS = 99


@pytest.fixture()
def device(device_id_good: int) -> Iterator[PyK4A]:
    device = PyK4A(device_id=device_id_good)
    yield device
    # autoclose
    try:
        if device.opened:
            device.close()
    except K4AException:
        pass


@pytest.fixture()
def device_not_exists(device_id_not_exists: int) -> Iterator[PyK4A]:
    device = PyK4A(device_id=device_id_not_exists)
    yield device
    # autoclose
    try:
        if device.opened:
            device.close()
    except K4AException:
        pass


class TestOpenClose:
    @staticmethod
    def test_open_none_existing_device(device_not_exists: PyK4A):
        with pytest.raises(K4AException):
            device_not_exists.open()

    @staticmethod
    def test_open_existing_device(device: PyK4A):
        device.open()

    @staticmethod
    def test_open_twice(device: PyK4A):
        device.open()
        with pytest.raises(K4AException, match=r"Device already opened"):
            device.open()
