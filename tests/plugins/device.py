import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import pytest

from pyk4a import (
    Calibration,
    ColorControlCapabilities,
    ColorControlCommand,
    ColorControlMode,
    ColorResolution,
    DepthMode,
)
from pyk4a.results import Result


@dataclass(frozen=True)
class DeviceMeta:
    id: int
    jack_in: bool = False
    jack_out: bool = False
    serial: str = "123456789"
    color_controls: Tuple[ColorControlCapabilities, ...] = (
        ColorControlCapabilities(
            color_control_command=ColorControlCommand.EXPOSURE_TIME_ABSOLUTE,
            supports_auto=True,
            min_value=500,
            max_value=133330,
            step_value=100,
            default_value=16670,
            default_mode=ColorControlMode.AUTO,
        ),
        ColorControlCapabilities(
            color_control_command=ColorControlCommand.AUTO_EXPOSURE_PRIORITY,
            supports_auto=False,
            min_value=0,
            max_value=0,
            step_value=0,
            default_value=0,
            default_mode=ColorControlMode.MANUAL,
        ),
        ColorControlCapabilities(
            color_control_command=ColorControlCommand.BRIGHTNESS,
            supports_auto=False,
            min_value=0,
            max_value=255,
            step_value=1,
            default_value=128,
            default_mode=ColorControlMode.MANUAL,
        ),
        ColorControlCapabilities(
            color_control_command=ColorControlCommand.CONTRAST,
            supports_auto=False,
            min_value=0,
            max_value=10,
            step_value=1,
            default_value=5,
            default_mode=ColorControlMode.MANUAL,
        ),
        ColorControlCapabilities(
            color_control_command=ColorControlCommand.SATURATION,
            supports_auto=False,
            min_value=0,
            max_value=63,
            step_value=1,
            default_value=32,
            default_mode=ColorControlMode.MANUAL,
        ),
        ColorControlCapabilities(
            color_control_command=ColorControlCommand.SHARPNESS,
            supports_auto=False,
            min_value=0,
            max_value=4,
            step_value=1,
            default_value=2,
            default_mode=ColorControlMode.MANUAL,
        ),
        ColorControlCapabilities(
            color_control_command=ColorControlCommand.WHITEBALANCE,
            supports_auto=True,
            min_value=2500,
            max_value=12500,
            step_value=10,
            default_value=4500,
            default_mode=ColorControlMode.AUTO,
        ),
        ColorControlCapabilities(
            color_control_command=ColorControlCommand.BACKLIGHT_COMPENSATION,
            supports_auto=False,
            min_value=0,
            max_value=1,
            step_value=1,
            default_value=0,
            default_mode=ColorControlMode.MANUAL,
        ),
        ColorControlCapabilities(
            color_control_command=ColorControlCommand.GAIN,
            supports_auto=False,
            min_value=0,
            max_value=255,
            step_value=1,
            default_value=128,
            default_mode=ColorControlMode.MANUAL,
        ),
        ColorControlCapabilities(
            color_control_command=ColorControlCommand.POWERLINE_FREQUENCY,
            supports_auto=False,
            min_value=1,
            max_value=2,
            step_value=1,
            default_value=2,
            default_mode=ColorControlMode.MANUAL,
        ),
    )


DEVICE_METAS: Mapping[int, DeviceMeta] = {0: DeviceMeta(id=0)}


@pytest.fixture()
def patch_module_device(monkeypatch, calibration_raw, capture_factory):
    @dataclass
    class ColorControl:
        mode: int
        value: int

    class DeviceHandle:
        def __init__(self, device_id: int):
            self._meta: DeviceMeta = DEVICE_METAS[device_id]
            self._opened = True
            self._cameras_started = False
            self._imu_started = False
            self._color_controls: Mapping[ColorControlCommand, ColorControl] = self._default_color_controls()

        def _default_color_controls(self) -> Mapping[ColorControlCommand, ColorControl]:
            ret: Mapping[ColorControlCommand, ColorControl] = {}
            for color_control in self._meta.color_controls:
                ret[color_control["color_control_command"]] = ColorControl(
                    mode=color_control["default_mode"], value=color_control["default_value"]
                )
            return ret

        def close(self) -> int:
            assert self._opened is True
            self._opened = False
            return Result.Success.value

        def get_sync_jack(self) -> Tuple[int, bool, bool]:
            assert self._opened is True
            return Result.Success.value, self._meta.jack_in, self._meta.jack_out

        def device_get_color_control(self, cmd: int) -> Tuple[int, int, int]:
            assert self._opened is True
            control = self._color_controls[ColorControlCommand(cmd)]
            return Result.Success.value, control.mode, control.value

        def device_set_color_control(self, cmd: int, mode: int, value: int):
            assert self._opened is True
            command = ColorControlCommand(cmd)
            control = self._color_controls[command]
            for color_control_meta in self._meta.color_controls:
                if color_control_meta["color_control_command"] == command:
                    control_meta: ColorControlCapabilities = color_control_meta
                    break
            else:
                # Non-reachable
                raise ValueError(f"Unknown cmd: {cmd}")

            if ColorControlMode(mode) == ColorControlMode.AUTO:
                if not control_meta["supports_auto"]:
                    return Result.Failed.value
                control.mode = mode
                control.value = control_meta["default_value"]
            else:
                if value < color_control_meta["min_value"] or value > color_control_meta["max_value"]:
                    return Result.Failed.value
                control.mode = mode
                control.value = value
            return Result.Success.value

        def device_get_color_control_capabilities(self, cmd: int) -> Tuple[int, ColorControlCapabilities]:
            assert self._opened is True
            command = ColorControlCommand(cmd)
            for color_control_meta in self._meta.color_controls:
                if color_control_meta["color_control_command"] == command:
                    control_meta: ColorControlCapabilities = color_control_meta
                    break
            else:
                # Non-reachable
                raise ValueError(f"Unknown cmd: {cmd}")
            return Result.Success.value, control_meta

        def device_start_cameras(self) -> int:
            assert self._opened is True
            if self._cameras_started:
                return Result.Failed.value
            self._cameras_started = True
            return Result.Success.value

        def device_stop_cameras(self) -> int:
            assert self._opened is True
            if not self._cameras_started:
                return Result.Failed.value
            self._cameras_started = False
            return Result.Success.value

        def device_start_imu(self) -> int:
            assert self._opened is True
            if not self._cameras_started:  # imu didnt work without color camera
                return Result.Failed.value
            if self._imu_started:
                return Result.Failed.value
            self._imu_started = True
            return Result.Success.value

        def device_stop_imu(self) -> int:
            assert self._opened is True
            if not self._imu_started:
                return Result.Failed.value
            self._imu_started = False
            return Result.Success.value

        def device_get_capture(self) -> Tuple[int, Optional[object]]:
            assert self._opened is True
            if not self._cameras_started:
                return Result.Failed.value, None
            return Result.Success.value, capture_factory()

        def device_get_imu_sample(
            self,
        ) -> Tuple[int, Optional[Tuple[float, Tuple[float, float, float], int, Tuple[float, float, float], int]]]:
            assert self._opened is True
            if not self._cameras_started:
                return Result.Failed.value, None
            return (
                Result.Success.value,
                (36.6, (0.1, 9.8, 0.005), int(time.time() * 1e6), (0.1, 0.2, 0.3), int(time.time() * 1e6)),
            )

        def device_get_calibration(self, depth_mode: int, color_resolution: int) -> Tuple[int, Optional[object]]:
            assert self._opened is True
            calibration = Calibration.from_raw(calibration_raw, DepthMode.NFOV_UNBINNED, ColorResolution.RES_720P)
            return Result.Success.value, calibration._calibration_handle

        def device_get_raw_calibration(self) -> Optional[str]:
            assert self._opened is True
            return "{}"

        def device_get_serialnum(self) -> str:
            assert self._opened is True
            return self._meta.serial

    def _device_open(device_id: int, thread_safe: bool) -> Tuple[int, object]:
        if device_id not in DEVICE_METAS:
            return Result.Failed.value, None
        capsule = DeviceHandle(device_id)
        return Result.Success.value, capsule

    def _device_close(capsule: DeviceHandle, thread_safe: bool) -> int:
        capsule.close()
        return Result.Success.value

    def _device_get_sync_jack(capsule: DeviceHandle, thread_safe: bool) -> Tuple[int, bool, bool]:
        return capsule.get_sync_jack()

    def _device_get_color_control(capsule: DeviceHandle, thread_safe: bool, cmd: int) -> Tuple[int, int, int]:
        return capsule.device_get_color_control(cmd)

    def _device_set_color_control(capsule: DeviceHandle, thread_safe: bool, cmd: int, mode: int, value: int) -> int:
        return capsule.device_set_color_control(cmd, mode, value)

    def _device_get_color_control_capabilities(
        capsule: DeviceHandle, thread_safe: bool, cmd: int
    ) -> Tuple[int, ColorControlCapabilities]:
        return capsule.device_get_color_control_capabilities(cmd)

    def _device_start_cameras(
        capsule: DeviceHandle,
        thread_safe: bool,
        color_format: int,
        color_resolution: int,
        dept_mode: int,
        camera_fps: int,
        synchronized_images_only: bool,
        depth_delay_off_color_usec: int,
        wired_sync_mode: int,
        subordinate_delay_off_master_usec: int,
        disable_streaming_indicator: bool,
    ) -> int:
        return capsule.device_start_cameras()

    def _device_stop_cameras(capsule: DeviceHandle, thread_safe: bool) -> int:
        return capsule.device_stop_cameras()

    def _device_start_imu(capsule: DeviceHandle, thread_safe: bool) -> int:
        return capsule.device_start_imu()

    def _device_stop_imu(capsule: DeviceHandle, thread_safe: bool) -> int:
        return capsule.device_stop_imu()

    def _device_get_capture(capsule: DeviceHandle, thread_safe: bool, timeout: int) -> Tuple[int, Optional[object]]:
        return capsule.device_get_capture()

    def _device_get_imu_sample(
        capsule: DeviceHandle, thread_safe: bool, timeout: int
    ) -> Tuple[int, Tuple[float, Tuple[float, float, float], int, Tuple[float, float, float], int]]:
        return capsule.device_get_imu_sample()

    def _device_get_calibration(
        capsule: DeviceHandle, thread_safe, depth_mode: int, color_resolution: int
    ) -> Tuple[int, Optional[object]]:
        return capsule.device_get_calibration(depth_mode, color_resolution)

    def _device_get_raw_calibration(capsule: DeviceHandle, thread_safe) -> Optional[str]:
        return capsule.device_get_raw_calibration()

    def _device_get_installed_count() -> int:
        return 1

    def _device_get_serialnum(capsule: DeviceHandle, thread_safe) -> Optional[str]:
        return capsule.device_get_serialnum()

    monkeypatch.setattr("k4a_module.device_open", _device_open)
    monkeypatch.setattr("k4a_module.device_close", _device_close)
    monkeypatch.setattr("k4a_module.device_get_sync_jack", _device_get_sync_jack)
    monkeypatch.setattr("k4a_module.device_get_color_control", _device_get_color_control)
    monkeypatch.setattr("k4a_module.device_set_color_control", _device_set_color_control)
    monkeypatch.setattr("k4a_module.device_get_color_control_capabilities", _device_get_color_control_capabilities)
    monkeypatch.setattr("k4a_module.device_start_cameras", _device_start_cameras)
    monkeypatch.setattr("k4a_module.device_stop_cameras", _device_stop_cameras)
    monkeypatch.setattr("k4a_module.device_start_imu", _device_start_imu)
    monkeypatch.setattr("k4a_module.device_stop_imu", _device_stop_imu)
    monkeypatch.setattr("k4a_module.device_get_capture", _device_get_capture)
    monkeypatch.setattr("k4a_module.device_get_imu_sample", _device_get_imu_sample)
    monkeypatch.setattr("k4a_module.device_get_calibration", _device_get_calibration)
    monkeypatch.setattr("k4a_module.device_get_raw_calibration", _device_get_raw_calibration)
    monkeypatch.setattr("k4a_module.device_get_installed_count", _device_get_installed_count)
    monkeypatch.setattr("k4a_module.device_get_serialnum", _device_get_serialnum)


@pytest.fixture()
def device_id_good(patch_module_device: Any) -> int:
    return 0


@pytest.fixture()
def device_id_not_exists(patch_module_device: Any) -> int:
    return 99
