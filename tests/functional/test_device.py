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


class TestProperties:
    @staticmethod
    @pytest.mark.device
    def test_sync_jack_status(device_id: int):
        device = PyK4A(device_id=device_id)
        device.open()
        jack_in, jack_out = device.sync_jack_status
        assert isinstance(jack_in, bool)
        assert isinstance(jack_out, bool)

    @staticmethod
    @pytest.mark.device
    def test_get_color_control(device_id: int):
        device = PyK4A(device_id=device_id)
        device.open()
        brightness = device.brightness
        assert isinstance(brightness, int)

    @staticmethod
    @pytest.mark.device
    def test_set_color_control(device_id: int):
        device = PyK4A(device_id=device_id)
        device.open()
        device.brightness = 123

    @staticmethod
    @pytest.mark.device
    def test_reset_color_control_to_default(device_id: int):
        device = PyK4A(device_id=device_id)
        device.open()
        device.reset_color_control_to_default()
        assert device.brightness == 128  # default value 128
        device.brightness = 123
        assert device.brightness == 123
        device.reset_color_control_to_default()
        assert device.brightness == 128

    @staticmethod
    @pytest.mark.device
    def test_start_stop_cameras(device_id: int):
        device = PyK4A(device_id=device_id)
        device.open()
        device._start_cameras()
        device._stop_cameras()

    @staticmethod
    @pytest.mark.device
    def test_start_stop_imu(device_id: int):
        device = PyK4A(device_id=device_id)
        device.open()
        device._start_cameras()  # imu will not work without cameras
        device._start_imu()
        device._stop_imu()
