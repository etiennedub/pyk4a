import numpy as np
import pytest

from pyk4a import Calibration, ColorResolution, DepthMode


@pytest.fixture()
def calibration(calibration_raw) -> Calibration:
    return Calibration.from_raw(
        calibration_raw, depth_mode=DepthMode.NFOV_UNBINNED, color_resolution=ColorResolution.RES_720P
    )


class TestCalibration:
    @staticmethod
    def test_properties(calibration_raw):
        calibration = Calibration.from_raw(
            calibration_raw, depth_mode=DepthMode.NFOV_2X2BINNED, color_resolution=ColorResolution.RES_1536P
        )
        assert calibration.depth_mode == DepthMode.NFOV_2X2BINNED
        assert calibration.color_resolution == ColorResolution.RES_1536P

    def test_color_to_depth_3d(self, calibration: Calibration):
        point_color = (1000, 1500, 2000)
        point_depth = (1031.4664306640625, 1325.8529052734375, 2117.6611328125)
        converted = calibration.color_to_depth_3d(point_color)
        assert np.allclose(converted, point_depth)

    def test_depth_to_color_3d(self, calibration: Calibration):
        point_color = (1000, 1500, 2000)
        point_depth = (1031.4664306640625, 1325.8529052734375, 2117.6611328125)
        converted = calibration.depth_to_color_3d(point_depth)
        assert np.allclose(converted, point_color)
