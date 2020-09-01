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


class TestOpenClose:
    @staticmethod
    def test_open_none_existing_device(device_id_not_exists: int):
        device = PyK4A(device_id=device_id_not_exists)
        with pytest.raises(K4AException):
            device.open()

    @staticmethod
    @pytest.mark.device
    def test_open_existing_device(device_id: int):
        device = PyK4A(device_id=device_id)
        device.open()
        device.close()
