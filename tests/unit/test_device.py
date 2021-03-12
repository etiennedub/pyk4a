from typing import Iterator

import pytest

from pyk4a import K4AException, PyK4A, connected_device_count


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
        with pytest.raises(K4AException, match="Device already opened"):
            device.open()


class TestProperties:
    @staticmethod
    def test_sync_jack_status_on_closed_device(device: PyK4A):
        with pytest.raises(K4AException, match="Device is not opened"):
            assert device.sync_jack_status

    @staticmethod
    def test_sync_jack_status(device: PyK4A):
        device.open()
        jack_in, jack_out = device.sync_jack_status
        assert isinstance(jack_in, bool)
        assert isinstance(jack_out, bool)

    @staticmethod
    def test_color_property_on_closed_device(device: PyK4A):
        with pytest.raises(K4AException, match="Device is not opened"):
            assert device.brightness

    @staticmethod
    def test_color_property(device: PyK4A):
        device.open()
        assert device.brightness == 128
        assert device.contrast == 5
        assert device.saturation == 32
        assert device.sharpness == 2
        assert device.backlight_compensation == 0
        assert device.gain == 128
        assert device.powerline_frequency == 2
        assert device.exposure == 16670
        assert device.exposure_mode_auto is True
        assert device.whitebalance == 4500
        assert device.whitebalance_mode_auto is True

    @staticmethod
    def test_color_property_setter_on_closed_device(device: PyK4A):
        with pytest.raises(K4AException, match="Device is not opened"):
            device.brightness = 123

    @staticmethod
    def test_color_property_setter_incorrect_value(device: PyK4A):
        with pytest.raises(K4AException):
            device.contrast = 5000

    @staticmethod
    def test_color_property_setter(device: PyK4A):
        device.open()
        device.brightness = 123
        assert device.brightness == 123
        device.contrast = 4
        assert device.contrast == 4
        device.saturation = 20
        assert device.saturation == 20
        device.sharpness = 1
        assert device.sharpness == 1
        device.backlight_compensation = 1
        assert device.backlight_compensation == 1
        device.gain = 123
        assert device.gain == 123
        device.powerline_frequency = 1
        assert device.powerline_frequency == 1
        device.exposure = 17000
        assert device.exposure == 17000
        device.exposure_mode_auto = False
        assert device.exposure_mode_auto is False
        device.whitebalance = 5000
        assert device.whitebalance == 5000
        device.whitebalance_mode_auto = False
        assert device.whitebalance_mode_auto is False

    @staticmethod
    def test_reset_color_control_to_default_on_closed_device(device: PyK4A):
        with pytest.raises(K4AException, match="Device is not opened"):
            device.reset_color_control_to_default()

    @staticmethod
    def test_reset_color_control_to_default(device: PyK4A):
        device.open()
        device.brightness = 123
        assert device.brightness == 123
        device.reset_color_control_to_default()
        assert device.brightness == 128

    @staticmethod
    def test_calibration(device: PyK4A):
        device.open()
        device._start_cameras()
        calibration = device.calibration
        assert calibration

    @staticmethod
    def test_serial(device: PyK4A):
        device.open()
        serial = device.serial
        assert serial == "123456789"


class TestCameras:
    @staticmethod
    def test_capture_on_closed_device(device: PyK4A):
        with pytest.raises(K4AException, match="Device is not opened"):
            device.get_capture()

    @staticmethod
    def test_get_capture(device: PyK4A):
        device.open()
        device._start_cameras()
        capture = device.get_capture()
        assert capture is not None


class TestIMU:
    @staticmethod
    def test_get_imu_sample_on_closed_device(device: PyK4A):
        with pytest.raises(K4AException, match="Device is not opened"):
            device.get_imu_sample()

    @staticmethod
    def test_get_imu_sample(device: PyK4A):
        device.open()
        device._start_cameras()
        device._start_imu()
        sample = device.get_imu_sample()
        assert sample


class TestCalibrationRaw:
    @staticmethod
    def test_calibration_raw_on_closed_device(device: PyK4A):
        with pytest.raises(K4AException, match="Device is not opened"):
            assert device.calibration_raw

    @staticmethod
    def test_calibration_raw(device: PyK4A):
        device.open()
        assert device.calibration_raw


class TestInstalledCount:
    @staticmethod
    def test_count(patch_module_device):
        count = connected_device_count()
        assert count == 1
