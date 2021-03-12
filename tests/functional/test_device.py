import pytest

from pyk4a import K4AException, PyK4A, connected_device_count


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
        assert calibration._calibration_handle

    @staticmethod
    @pytest.mark.device
    def test_serial(device: PyK4A):
        device.open()
        serial = device.serial
        assert len(serial) > 5


class TestCameras:
    @staticmethod
    @pytest.mark.device
    @pytest.mark.opengl
    def test_start_stop_cameras(device: PyK4A):
        device.open()
        device._start_cameras()
        device._stop_cameras()

    @staticmethod
    @pytest.mark.device
    @pytest.mark.opengl
    def test_capture(device: PyK4A):
        device.open()
        device._start_cameras()
        capture = device.get_capture()
        assert capture._capture_handle is not None


class TestIMU:
    @staticmethod
    @pytest.mark.device
    @pytest.mark.opengl
    def test_start_stop_imu(device: PyK4A):
        device.open()
        device._start_cameras()  # imu will not work without cameras
        device._start_imu()
        device._stop_imu()
        device._stop_cameras()

    @staticmethod
    @pytest.mark.device
    @pytest.mark.opengl
    def test_get_imu_sample(device: PyK4A):
        device.open()
        device._start_cameras()
        device._start_imu()
        sample = device.get_imu_sample()
        assert sample is not None


class TestCalibrationRaw:
    @staticmethod
    @pytest.mark.device
    def test_calibration_raw(device: PyK4A):
        device.open()
        raw = device.calibration_raw
        import sys

        print(raw, file=sys.stderr)
        assert raw


class TestInstalledCount:
    @staticmethod
    def test_count():
        count = connected_device_count()
        assert count >= 0
