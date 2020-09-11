from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import pytest

from pyk4a.config import FPS, ColorResolution, DepthMode, ImageFormat, WiredSyncMode
from pyk4a.playback import SeekOrigin
from pyk4a.results import BufferResult, Result, StreamResult
from tests.plugins.calibration import CalibrationHandle


@dataclass(frozen=True)
class PlaybackMeta:
    filename: str
    length: int
    calibration_json_str: Optional[str] = None
    configuration: Optional[Tuple[int, ...]] = None


PLAYBACK_METAS: Mapping[str, PlaybackMeta] = {
    "file1.mkv": PlaybackMeta(
        filename="file1.mkv",
        length=1234,
        calibration_json_str=r'{"CalibrationInformation":{"Cameras":[{"Intrinsics":{"ModelParameterCount":14,"ModelParameters":[0.51296979188919067,0.51692354679107666,0.4927808940410614,0.49290284514427185,0.54793977737426758,-0.020971370860934258,-0.0026522316038608551,0.88967907428741455,0.086130209267139435,-0.013910384848713875,0,0,-7.2104157879948616E-5,2.0557534298859537E-5],"ModelType":"CALIBRATION_LensDistortionModelBrownConrady"},"Location":"CALIBRATION_CameraLocationD0","Purpose":"CALIBRATION_CameraPurposeDepth","MetricRadius":1.7399997711181641,"Rt":{"Rotation":[1,0,0,0,1,0,0,0,1],"Translation":[0,0,0]},"SensorHeight":1024,"SensorWidth":1024,"Shutter":"CALIBRATION_ShutterTypeUndefined","ThermalAdjustmentParams":{"Params":[0,0,0,0,0,0,0,0,0,0,0,0]}},{"Intrinsics":{"ModelParameterCount":14,"ModelParameters":[0.49984714388847351,0.50661760568618774,0.47796511650085449,0.63742393255233765,0.33012130856513977,-2.4893965721130371,1.453670859336853,0.20891207456588745,-2.302915096282959,1.3751275539398193,0,0,-0.00012944525224156678,-0.00016178244550246745],"ModelType":"CALIBRATION_LensDistortionModelBrownConrady"},"Location":"CALIBRATION_CameraLocationPV0","Purpose":"CALIBRATION_CameraPurposePhotoVideo","MetricRadius":0,"Rt":{"Rotation":[0.99999922513961792,0.0011817576596513391,-0.00044814043212682009,-0.0011387512786313891,0.99628061056137085,0.0861603245139122,0.0005482942215166986,-0.086159750819206238,0.9962812066078186],"Translation":[-0.032083429396152496,-0.0022053730208426714,0.0038836926687508821]},"SensorHeight":3072,"SensorWidth":4096,"Shutter":"CALIBRATION_ShutterTypeUndefined","ThermalAdjustmentParams":{"Params":[0,0,0,0,0,0,0,0,0,0,0,0]}}],"InertialSensors":[{"BiasTemperatureModel":[-0.00038840249180793762,0,0,0,0.0066333282738924026,0,0,0,0.0011737601598724723,0,0,0],"BiasUncertainty":[9.9999997473787516E-5,9.9999997473787516E-5,9.9999997473787516E-5],"Id":"CALIBRATION_InertialSensorId_LSM6DSM","MixingMatrixTemperatureModel":[0.99519604444503784,0,0,0,0.0019702909048646688,0,0,0,-0.00029000110225751996,0,0,0,0.0019510588608682156,0,0,0,1.00501549243927,0,0,0,0.0030893723014742136,0,0,0,-0.00028924690559506416,0,0,0,0.0031117112375795841,0,0,0,0.99779671430587769,0,0,0],"ModelTypeMask":16,"Noise":[0.00095000001601874828,0.00095000001601874828,0.00095000001601874828,0,0,0],"Rt":{"Rotation":[0.0018387800082564354,0.11238107085227966,-0.993663489818573,-0.99999493360519409,-0.002381625585258007,-0.0021198526956140995,-0.0026047658175230026,0.99366235733032227,0.11237611621618271],"Translation":[0,0,0]},"SecondOrderScaling":[0,0,0,0,0,0,0,0,0],"SensorType":"CALIBRATION_InertialSensorType_Gyro","TemperatureBounds":[5,60],"TemperatureC":0},{"BiasTemperatureModel":[0.1411108672618866,0,0,0,0.0053673363290727139,0,0,0,-0.42933177947998047,0,0,0],"BiasUncertainty":[0.0099999997764825821,0.0099999997764825821,0.0099999997764825821],"Id":"CALIBRATION_InertialSensorId_LSM6DSM","MixingMatrixTemperatureModel":[0.99946749210357666,0,0,0,3.0261040592449717E-5,0,0,0,-0.00310571794398129,0,0,0,3.0574858101317659E-5,0,0,0,0.98919963836669922,0,0,0,0.00054294260917231441,0,0,0,-0.0031195830088108778,0,0,0,0.00053976889466866851,0,0,0,0.995025634765625,0,0,0],"ModelTypeMask":56,"Noise":[0.010700000450015068,0.010700000450015068,0.010700000450015068,0,0,0],"Rt":{"Rotation":[0.0019369830843061209,0.1084766760468483,-0.994097113609314,-0.99999403953552246,-0.0026388836558908224,-0.0022364303003996611,-0.0028659072704613209,0.994095504283905,0.10847091674804688],"Translation":[-0.0509832426905632,0.0034723200369626284,0.0013277813559398055]},"SecondOrderScaling":[0,0,0,0,0,0,0,0,0],"SensorType":"CALIBRATION_InertialSensorType_Accelerometer","TemperatureBounds":[5,60],"TemperatureC":0}],"Metadata":{"SerialId":"001514394512","FactoryCalDate":"11/9/2019 10:32:12 AM GMT","Version":{"Major":1,"Minor":2},"DeviceName":"AzureKinect-PV","Notes":"PV0_max_radius_invalid"}}}',  # noqa: E501
        configuration=(
            ImageFormat.COLOR_MJPG.value,
            ColorResolution.RES_720P.value,
            DepthMode.NFOV_UNBINNED.value,
            FPS.FPS_5.value,
            1,
            1,
            1,
            1,
            1,
            WiredSyncMode.STANDALONE.value,
            0,
            336277,
        ),
    ),
    "file2_bad.mkv": PlaybackMeta(filename="file2_bad.mkv.mkv", length=0),
}


@pytest.fixture()
def patch_module_playback(monkeypatch):
    class PlaybackHandle:
        def __init__(self, filename: str):
            self._filename = filename
            self._meta: PlaybackMeta = PLAYBACK_METAS[filename]
            self._opened = True
            self._position: int = 0

        def close(self) -> int:
            assert self._opened
            self._opened = False
            return Result.Success.value

        def playback_get_recording_length_usec(self) -> int:
            assert self._opened
            return self._meta.length

        def playback_get_raw_calibration(self) -> Tuple[int, str]:
            assert self._opened
            if self._meta.calibration_json_str:
                return BufferResult.Success.value, self._meta.calibration_json_str
            return BufferResult.TooSmall.value, ""

        def playback_seek_timestamp(self, offset: int, origin: int) -> int:
            assert self._opened
            if self._meta.length == 0:
                return StreamResult.Failed.value
            typed_origin = SeekOrigin(origin)
            position = self._position
            if typed_origin == SeekOrigin.BEGIN:
                position = position + offset
                if position > self._meta.length:
                    return StreamResult.EOF.value
            elif typed_origin == SeekOrigin.END:
                position = position - offset
                if position < 0:
                    return StreamResult.EOF.value
            elif typed_origin == SeekOrigin.DEVICE_TIME:
                # not supported in mock
                pass
            self._position = position
            return StreamResult.Success.value

        def playback_get_calibration(self) -> Tuple[int, object]:
            assert self._opened
            return StreamResult.Success.value, CalibrationHandle()

        def playback_get_record_configuration(self) -> Tuple[int, Tuple[int, ...]]:
            assert self._opened
            if self._meta.configuration:
                return Result.Success.value, self._meta.configuration
            return Result.Failed.value, ()

    def _playback_open(filename: str, thread_safe: bool) -> Tuple[int, object]:
        if filename not in PLAYBACK_METAS:
            return Result.Failed.value, None
        capsule = PlaybackHandle(filename)
        return Result.Success.value, capsule

    def _playback_close(capsule: PlaybackHandle, thread_safe: bool) -> int:
        return capsule.close()

    def _playback_get_recording_length_usec(capsule: PlaybackHandle, thread_safe: bool) -> int:
        return capsule.playback_get_recording_length_usec()

    def _playback_get_raw_calibration(capsule: PlaybackHandle, thread_safe: bool) -> Tuple[int, str]:
        return capsule.playback_get_raw_calibration()

    def _playback_seek_timestamp(capsule: PlaybackHandle, thread_safe: bool, offset: int, origin: int) -> int:
        return capsule.playback_seek_timestamp(offset, origin)

    def _playback_get_calibration(capsule: PlaybackHandle, thread_safe: bool) -> Tuple[int, object]:
        return capsule.playback_get_calibration()

    def _playback_get_record_configuration(capsule: PlaybackHandle, thread_safe: bool) -> Tuple[int, Tuple[int, ...]]:
        return capsule.playback_get_record_configuration()

    monkeypatch.setattr("k4a_module.playback_open", _playback_open)
    monkeypatch.setattr("k4a_module.playback_close", _playback_close)
    monkeypatch.setattr("k4a_module.playback_get_recording_length_usec", _playback_get_recording_length_usec)
    monkeypatch.setattr("k4a_module.playback_get_raw_calibration", _playback_get_raw_calibration)
    monkeypatch.setattr("k4a_module.playback_seek_timestamp", _playback_seek_timestamp)
    monkeypatch.setattr("k4a_module.playback_get_calibration", _playback_get_calibration)
    monkeypatch.setattr("k4a_module.playback_get_record_configuration", _playback_get_record_configuration)


@pytest.fixture()
def recording_good_file(patch_module_playback: Any) -> str:
    return "file1.mkv"


@pytest.fixture()
def recording_bad_file(patch_module_playback: Any) -> str:
    return "file2_bad.mkv"


@pytest.fixture()
def recording_not_exists_file(patch_module_playback: Any) -> str:
    return "not_exists.mkv"
