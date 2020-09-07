from pyk4a import Calibration, ColorResolution, DepthMode


class TestCalibration:
    @staticmethod
    def test_properties(calibration_raw):
        calibration = Calibration.from_raw(
            calibration_raw, depth_mode=DepthMode.NFOV_2X2BINNED, color_resolution=ColorResolution.RES_1536P
        )
        assert calibration.depth_mode == DepthMode.NFOV_2X2BINNED
        assert calibration.color_resolution == ColorResolution.RES_1536P
