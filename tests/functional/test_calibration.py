import pytest

from pyk4a import Calibration, ColorResolution, DepthMode, K4AException


@pytest.fixture()
def calibration(calibration_raw) -> Calibration:
    return Calibration.from_raw(
        calibration_raw, depth_mode=DepthMode.NFOV_UNBINNED, color_resolution=ColorResolution.RES_720P
    )


class TestCalibration:
    @staticmethod
    def test_from_raw_incorrect_data(calibration_raw):
        with pytest.raises(K4AException):
            Calibration.from_raw(
                "none-calibration-json-string", depth_mode=DepthMode.NFOV_UNBINNED, color_resolution=ColorResolution.OFF
            )

    @staticmethod
    def test_from_raw(calibration_raw):
        calibration = Calibration.from_raw(
            calibration_raw, depth_mode=DepthMode.NFOV_UNBINNED, color_resolution=ColorResolution.OFF
        )
        assert calibration

    @staticmethod
    @pytest.mark.opengl
    def test_creating_transfromation_handle(calibration: Calibration):
        transformation = calibration.transformation_handle
        assert transformation is not None
