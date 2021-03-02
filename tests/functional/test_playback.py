import pytest

from pyk4a import K4AException, PyK4APlayback, SeekOrigin
from pyk4a.config import FPS, ColorResolution, DepthMode, ImageFormat, WiredSyncMode
from pyk4a.playback import Configuration


RECORD_LENGTH = 463945
RECORD_CALIBRATION_JSON = r'{"CalibrationInformation":{"Cameras":[{"Intrinsics":{"ModelParameterCount":14,"ModelParameters":[0.51296979188919067,0.51692354679107666,0.4927808940410614,0.49290284514427185,0.54793977737426758,-0.020971370860934258,-0.0026522316038608551,0.88967907428741455,0.086130209267139435,-0.013910384848713875,0,0,-7.2104157879948616E-5,2.0557534298859537E-5],"ModelType":"CALIBRATION_LensDistortionModelBrownConrady"},"Location":"CALIBRATION_CameraLocationD0","Purpose":"CALIBRATION_CameraPurposeDepth","MetricRadius":1.7399997711181641,"Rt":{"Rotation":[1,0,0,0,1,0,0,0,1],"Translation":[0,0,0]},"SensorHeight":1024,"SensorWidth":1024,"Shutter":"CALIBRATION_ShutterTypeUndefined","ThermalAdjustmentParams":{"Params":[0,0,0,0,0,0,0,0,0,0,0,0]}},{"Intrinsics":{"ModelParameterCount":14,"ModelParameters":[0.49984714388847351,0.50661760568618774,0.47796511650085449,0.63742393255233765,0.33012130856513977,-2.4893965721130371,1.453670859336853,0.20891207456588745,-2.302915096282959,1.3751275539398193,0,0,-0.00012944525224156678,-0.00016178244550246745],"ModelType":"CALIBRATION_LensDistortionModelBrownConrady"},"Location":"CALIBRATION_CameraLocationPV0","Purpose":"CALIBRATION_CameraPurposePhotoVideo","MetricRadius":0,"Rt":{"Rotation":[0.99999922513961792,0.0011817576596513391,-0.00044814043212682009,-0.0011387512786313891,0.99628061056137085,0.0861603245139122,0.0005482942215166986,-0.086159750819206238,0.9962812066078186],"Translation":[-0.032083429396152496,-0.0022053730208426714,0.0038836926687508821]},"SensorHeight":3072,"SensorWidth":4096,"Shutter":"CALIBRATION_ShutterTypeUndefined","ThermalAdjustmentParams":{"Params":[0,0,0,0,0,0,0,0,0,0,0,0]}}],"InertialSensors":[{"BiasTemperatureModel":[-0.00038840249180793762,0,0,0,0.0066333282738924026,0,0,0,0.0011737601598724723,0,0,0],"BiasUncertainty":[9.9999997473787516E-5,9.9999997473787516E-5,9.9999997473787516E-5],"Id":"CALIBRATION_InertialSensorId_LSM6DSM","MixingMatrixTemperatureModel":[0.99519604444503784,0,0,0,0.0019702909048646688,0,0,0,-0.00029000110225751996,0,0,0,0.0019510588608682156,0,0,0,1.00501549243927,0,0,0,0.0030893723014742136,0,0,0,-0.00028924690559506416,0,0,0,0.0031117112375795841,0,0,0,0.99779671430587769,0,0,0],"ModelTypeMask":16,"Noise":[0.00095000001601874828,0.00095000001601874828,0.00095000001601874828,0,0,0],"Rt":{"Rotation":[0.0018387800082564354,0.11238107085227966,-0.993663489818573,-0.99999493360519409,-0.002381625585258007,-0.0021198526956140995,-0.0026047658175230026,0.99366235733032227,0.11237611621618271],"Translation":[0,0,0]},"SecondOrderScaling":[0,0,0,0,0,0,0,0,0],"SensorType":"CALIBRATION_InertialSensorType_Gyro","TemperatureBounds":[5,60],"TemperatureC":0},{"BiasTemperatureModel":[0.1411108672618866,0,0,0,0.0053673363290727139,0,0,0,-0.42933177947998047,0,0,0],"BiasUncertainty":[0.0099999997764825821,0.0099999997764825821,0.0099999997764825821],"Id":"CALIBRATION_InertialSensorId_LSM6DSM","MixingMatrixTemperatureModel":[0.99946749210357666,0,0,0,3.0261040592449717E-5,0,0,0,-0.00310571794398129,0,0,0,3.0574858101317659E-5,0,0,0,0.98919963836669922,0,0,0,0.00054294260917231441,0,0,0,-0.0031195830088108778,0,0,0,0.00053976889466866851,0,0,0,0.995025634765625,0,0,0],"ModelTypeMask":56,"Noise":[0.010700000450015068,0.010700000450015068,0.010700000450015068,0,0,0],"Rt":{"Rotation":[0.0019369830843061209,0.1084766760468483,-0.994097113609314,-0.99999403953552246,-0.0026388836558908224,-0.0022364303003996611,-0.0028659072704613209,0.994095504283905,0.10847091674804688],"Translation":[-0.0509832426905632,0.0034723200369626284,0.0013277813559398055]},"SecondOrderScaling":[0,0,0,0,0,0,0,0,0],"SensorType":"CALIBRATION_InertialSensorType_Accelerometer","TemperatureBounds":[5,60],"TemperatureC":0}],"Metadata":{"SerialId":"001514394512","FactoryCalDate":"11/9/2019 10:32:12 AM GMT","Version":{"Major":1,"Minor":2},"DeviceName":"AzureKinect-PV","Notes":"PV0_max_radius_invalid"}}}'  # noqa: E501
RECORD_CONFIGURATION = Configuration(
    color_format=ImageFormat.COLOR_MJPG,
    color_resolution=ColorResolution.RES_720P,
    depth_mode=DepthMode.NFOV_UNBINNED,
    camera_fps=FPS.FPS_5,
    color_track_enabled=True,
    depth_track_enabled=True,
    ir_track_enabled=True,
    imu_track_enabled=True,
    depth_delay_off_color_usec=0,
    wired_sync_mode=WiredSyncMode.STANDALONE,
    subordinate_delay_off_master_usec=0,
    start_timestamp_offset_usec=336277,
)


class TestInit:
    @staticmethod
    def test_not_existing_path():
        playback = PyK4APlayback(path="/some/not-exists.file")
        with pytest.raises(K4AException):
            playback.open()


class TestPropertyLength:
    @staticmethod
    def test_correct_value(playback: PyK4APlayback):
        playback.open()
        assert playback.length == RECORD_LENGTH


class TestPropertyCalibrationRaw:
    @staticmethod
    def test_correct_value(playback: PyK4APlayback):
        playback.open()
        assert playback.calibration_raw == RECORD_CALIBRATION_JSON


class TestPropertyConfiguration:
    @staticmethod
    def test_readness(playback: PyK4APlayback):
        playback.open()
        assert playback.configuration == RECORD_CONFIGURATION


class TestCalibration:
    @staticmethod
    def test_readness(playback: PyK4APlayback):
        playback.open()
        calibration = playback.calibration
        assert calibration


class TestSeek:
    # playback asset has only one capture inside

    @staticmethod
    def test_seek_from_start(playback: PyK4APlayback):
        # TODO fetch capture/data and validate time
        playback.open()
        playback.get_next_capture()
        playback.seek(playback.configuration["start_timestamp_offset_usec"], origin=SeekOrigin.BEGIN)
        capture = playback.get_next_capture()
        assert capture.color is not None
        with pytest.raises(EOFError):
            playback.get_previouse_capture()

    @staticmethod
    def test_seek_from_end(playback: PyK4APlayback):
        # TODO fetch capture/data and validate time
        playback.open()
        playback.seek(0, origin=SeekOrigin.END)
        capture = playback.get_previouse_capture()
        assert capture.color is not None
        with pytest.raises(EOFError):
            playback.get_next_capture()

    @staticmethod
    def test_seek_by_device_time(playback: PyK4APlayback):
        # TODO fetch capture/data and validate time
        playback.open()
        playback.seek(1, origin=SeekOrigin.DEVICE_TIME)  # TODO add correct timestamp from datablock here
        capture = playback.get_next_capture()
        assert capture.color is not None


class TestGetCapture:
    @staticmethod
    def test_get_next_capture(playback: PyK4APlayback):
        playback.open()
        capture = playback.get_next_capture()
        assert capture is not None
        assert capture.depth is not None
        assert capture.color is not None
        assert capture.depth_timestamp_usec == 800222
        assert capture.color_timestamp_usec == 800222
        assert capture.ir_timestamp_usec == 800222
        assert capture._calibration is not None  # Issue #81

    @staticmethod
    def test_get_previouse_capture(playback: PyK4APlayback):
        playback.open()
        playback.seek(0, origin=SeekOrigin.END)
        capture = playback.get_previouse_capture()
        assert capture is not None
        assert capture.depth is not None
        assert capture.color is not None
        assert capture.depth_timestamp_usec == 800222
        assert capture.color_timestamp_usec == 800222
        assert capture.ir_timestamp_usec == 800222
        assert capture._calibration is not None  # Issue #81
