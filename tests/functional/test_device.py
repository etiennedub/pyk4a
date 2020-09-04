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


class TestOpenClose:
    @staticmethod
    def test_open_none_existing_device(device_not_exists: PyK4A):
        with pytest.raises(K4AException):
            device_not_exists.open()

    @staticmethod
    @pytest.mark.device
    def test_open_close_existing_device(device: PyK4A):
        device.open()
        device.close()


class TestProperties:
    @staticmethod
    @pytest.mark.device
    def test_sync_jack_status(device: PyK4A):
        device.open()
        jack_in, jack_out = device.sync_jack_status
        assert isinstance(jack_in, bool)
        assert isinstance(jack_out, bool)

    @staticmethod
    @pytest.mark.device
    def test_get_color_control(device: PyK4A):
        device.open()
        brightness = device.brightness
        assert isinstance(brightness, int)

    @staticmethod
    @pytest.mark.device
    def test_set_color_control(device: PyK4A):
        device.open()
        device.brightness = 123

    @staticmethod
    @pytest.mark.device
    def test_reset_color_control_to_default(device: PyK4A):
        device.open()
        device.reset_color_control_to_default()
        assert device.brightness == 128  # default value 128
        device.brightness = 123
        assert device.brightness == 123
        device.reset_color_control_to_default()
        assert device.brightness == 128

    @staticmethod
    @pytest.mark.device
    def test_get_calibration(device: PyK4A):
        device.open()
        calibration = device.calibration
        assert calibration._handle


class TestCameras:
    @staticmethod
    @pytest.mark.device
    def test_start_stop_cameras(device: PyK4A):
        device.open()
        device._start_cameras()
        device._stop_cameras()

    @staticmethod
    @pytest.mark.device
    def test_capture(device: PyK4A):
        device.open()
        device._start_cameras()
        capture = device.get_capture()
        assert capture.color is not None


class TestIMU:
    @staticmethod
    @pytest.mark.device
    def test_start_stop_imu(device: PyK4A):
        device.open()
        device._start_cameras()  # imu will not work without cameras
        device._start_imu()
        device._stop_imu()
        device._stop_cameras()

    @staticmethod
    @pytest.mark.device
    def test_get_imu_sample(device: PyK4A):
        device.open()
        device._start_cameras()
        device._start_imu()
        sample = device.get_imu_sample()
        assert sample is not None
