import numpy as np
import pytest

from pyk4a import Calibration, CalibrationType, ColorResolution, DepthMode


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
        point_color = 1000.0, 1500.0, 2000.0
        point_depth = 1031.4664306640625, 1325.8529052734375, 2117.6611328125
        converted = calibration.color_to_depth_3d(point_color)
        assert np.allclose(converted, point_depth)

    def test_depth_to_color_3d(self, calibration: Calibration):
        point_color = 1000.0, 1500.0, 2000.0
        point_depth = 1031.4664306640625, 1325.8529052734375, 2117.6611328125
        converted = calibration.depth_to_color_3d(point_depth)
        assert np.allclose(converted, point_color)

    def test_2d_to_3d_without_target_camera(self, calibration: Calibration):
        point = 250.0, 300.0
        depth = 250.0
        point3d = -154.5365753173828, -26.12171173095703, 250.0
        converted = calibration.convert_2d_to_3d(point, depth, CalibrationType.COLOR)
        assert np.allclose(converted, point3d)

    def test_2d_to_3d_with_target_camera(self, calibration: Calibration):
        point = 250.0, 300.0
        depth = 250.0
        point3d = -122.29087829589844, -45.17741394042969, 243.19528198242188
        converted = calibration.convert_2d_to_3d(point, depth, CalibrationType.COLOR, CalibrationType.DEPTH)
        assert np.allclose(converted, point3d)

    def test_3d_to_2d_without_target_camera(self, calibration: Calibration):
        point3d = -154.5365753173828, -26.12171173095703, 250.0
        point = 250.0, 300.0
        converted = calibration.convert_3d_to_2d(point3d, CalibrationType.COLOR)
        assert np.allclose(converted, point)

    def test_3d_to_2d_with_target_camera(self, calibration: Calibration):
        point3d = -122.29087829589844, -45.17741394042969, 243.19528198242188
        point = 250.0, 300.0
        converted = calibration.convert_3d_to_2d(point3d, CalibrationType.DEPTH, CalibrationType.COLOR)
        assert np.allclose(converted, point)
