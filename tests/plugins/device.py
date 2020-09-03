from dataclasses import dataclass
from typing import Any, Mapping, Tuple

import pytest

from pyk4a.pyk4a import ColorControlCapabilities, ColorControlCommand, ColorControlMode, Result


@dataclass(frozen=True)
class DeviceMeta:
    id: int
    jack_in: bool = False
    jack_out: bool = False
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
def patch_module_device(monkeypatch):
    @dataclass
    class ColorControl:
        mode: int
        value: int

    class DeviceHandle:
        def __init__(self, device_id: int):
            self._meta: DeviceMeta = DEVICE_METAS[device_id]
            self._opened = True
            self._color_controls: Mapping[ColorControlCommand, ColorControl] = self._default_color_controls()

        def _default_color_controls(self) -> Mapping[ColorControlCommand, ColorControl]:
            ret: Mapping[ColorControlCommand, ColorControl] = {}
            for color_control in self._meta.color_controls:
                ret[color_control["color_control_command"]] = ColorControl(
                    mode=color_control["default_mode"], value=color_control["default_value"]
                )
            return ret

        def close(self) -> int:
            assert self._opened
            self._opened = False
            return Result.Success.value

        def get_sync_jack(self) -> Tuple[int, bool, bool]:
            return Result.Success.value, self._meta.jack_in, self._meta.jack_out

        def device_get_color_control(self, cmd: int) -> Tuple[int, int, int]:
            control = self._color_controls[ColorControlCommand(cmd)]
            return Result.Success.value, control.mode, control.value

        def device_set_color_control(self, cmd: int, mode: int, value: int):
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
            command = ColorControlCommand(cmd)
            for color_control_meta in self._meta.color_controls:
                if color_control_meta["color_control_command"] == command:
                    control_meta: ColorControlCapabilities = color_control_meta
                    break
            else:
                # Non-reachable
                raise ValueError(f"Unknown cmd: {cmd}")
            return Result.Success.value, control_meta

    def _device_open(device_id: int, thread_safe: bool):
        if device_id not in DEVICE_METAS:
            return Result.Failed.value, None
        capsule = DeviceHandle(device_id)
        return Result.Success.value, capsule

    def _device_close(capsule: DeviceHandle, thread_safe: bool):
        capsule.close()
        return Result.Success

    def _device_get_sync_jack(capsule: DeviceHandle, thread_safe: bool):
        return capsule.get_sync_jack()

    def _device_get_color_control(capsule: DeviceHandle, thread_safe: bool, cmd: int):
        return capsule.device_get_color_control(cmd)

    def _device_set_color_control(capsule: DeviceHandle, thread_safe: bool, cmd: int, mode: int, value: int):
        return capsule.device_set_color_control(cmd, mode, value)

    def _device_get_color_control_capabilities(capsule: DeviceHandle, thread_safe: bool, cmd: int):
        return capsule.device_get_color_control_capabilities(cmd)

    monkeypatch.setattr("k4a_module.device_open", _device_open)
    monkeypatch.setattr("k4a_module.device_close", _device_close)
    monkeypatch.setattr("k4a_module.device_get_sync_jack", _device_get_sync_jack)
    monkeypatch.setattr("k4a_module.device_get_color_control", _device_get_color_control)
    monkeypatch.setattr("k4a_module.device_set_color_control", _device_set_color_control)
    monkeypatch.setattr("k4a_module.device_get_color_control_capabilities", _device_get_color_control_capabilities)


@pytest.fixture()
def device_id_good(patch_module_device: Any) -> int:
    return 0


@pytest.fixture()
def device_id_not_exists(patch_module_device: Any) -> int:
    return 99
