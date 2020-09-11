import sys
from enum import IntEnum
from pathlib import Path
from typing import Any, Optional, Tuple, Union


if sys.version_info < (3, 8):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

import k4a_module

from .calibration import Calibration
from .capture import PyK4ACapture
from .config import FPS, ColorResolution, DepthMode, ImageFormat, WiredSyncMode
from .errors import K4AException, _verify_error
from .results import Result, StreamResult


class SeekOrigin(IntEnum):
    BEGIN = 0
    END = 1
    DEVICE_TIME = 2


class Configuration(TypedDict):
    color_format: ImageFormat
    color_resolution: ColorResolution
    depth_mode: DepthMode
    camera_fps: FPS
    color_track_enabled: bool
    depth_track_enabled: bool
    ir_track_enabled: bool
    imu_track_enabled: bool
    depth_delay_off_color_usec: int
    wired_sync_mode: WiredSyncMode
    subordinate_delay_off_master_usec: int
    start_timestamp_offset_usec: int


class PyK4APlayback:
    def __init__(self, path: Union[str, Path], thread_safe: bool = True):
        self._path: Path = Path(path)
        self.thread_safe = thread_safe
        self._handle: Optional[object] = None
        self._length: Optional[int] = None
        self._calibration_json: Optional[str] = None
        self._calibration: Optional[Calibration] = None
        self._configuration: Optional[Configuration] = None

    def __del__(self):
        if self._handle:
            self.close()

    @property
    def path(self) -> Path:
        """
            Record file path
        """
        return self._path

    @property
    def configuration(self) -> Configuration:
        self._validate_is_open()
        if self._configuration is None:
            res, conf = k4a_module.playback_get_record_configuration(
                self._handle, self.thread_safe
            )  # type: int, Tuple[Any,...]
            _verify_error(res)
            self._configuration = Configuration(
                color_format=ImageFormat(conf[0]),
                color_resolution=ColorResolution(conf[1]),
                depth_mode=DepthMode(conf[2]),
                camera_fps=FPS(conf[3]),
                color_track_enabled=bool(conf[4]),
                depth_track_enabled=bool(conf[5]),
                ir_track_enabled=bool(conf[6]),
                imu_track_enabled=bool(conf[7]),
                depth_delay_off_color_usec=conf[8],
                wired_sync_mode=WiredSyncMode(conf[9]),
                subordinate_delay_off_master_usec=conf[10],
                start_timestamp_offset_usec=conf[11],
            )
        return self._configuration

    @property
    def length(self) -> int:
        """
            Record length in usec
        """
        if self._length is None:
            self._validate_is_open()
            self._length = k4a_module.playback_get_recording_length_usec(self._handle, self.thread_safe)
        return self._length

    @property
    def calibration_raw(self) -> str:
        self._validate_is_open()
        res, raw = k4a_module.playback_get_raw_calibration(self._handle, self.thread_safe)
        _verify_error(res)
        return raw

    @calibration_raw.setter
    def calibration_raw(self, value: str):
        self._validate_is_open()
        self._calibration = Calibration.from_raw(
            value, self.calibration.depth_mode, self.calibration.color_resolution, self.thread_safe
        )

    @property
    def calibration(self) -> Calibration:
        self._validate_is_open()
        if self._calibration is None:
            res, handle = k4a_module.playback_get_calibration(self._handle, self.thread_safe)
            _verify_error(res)
            self._calibration = Calibration(
                handle=handle,
                depth_mode=self.configuration["depth_mode"],
                color_resolution=self.configuration["color_resolution"],
                thread_safe=self.thread_safe,
            )
        return self._calibration

    def open(self) -> None:
        """
            Open record file
        """
        if self._handle:
            raise K4AException("Playback already opened")
        result, handle = k4a_module.playback_open(str(self._path), self.thread_safe)
        if Result(result) != Result.Success:
            raise K4AException(f"Cannot open file {self._path}")

        self._handle = handle

    def close(self):
        """
            Close record file
        """
        self._validate_is_open()
        k4a_module.playback_close(self._handle, self.thread_safe)
        self._handle = None

    def seek(self, offset: int, origin: SeekOrigin = SeekOrigin.BEGIN) -> None:
        """
            Seek playback pointer to specified offset
        """
        self._validate_is_open()
        result = k4a_module.playback_seek_timestamp(self._handle, self.thread_safe, offset, int(origin))
        self._verify_stream_error(result)

    def get_next_capture(self):
        self._validate_is_open()
        result, capture_handle = k4a_module.playback_get_next_capture(self._handle, self.thread_safe)
        self._verify_stream_error(result)
        return PyK4ACapture(
            calibration=self._calibration,
            capture_handle=capture_handle,
            color_format=self.configuration["color_format"],
            thread_safe=self.thread_safe,
        )

    def get_previouse_capture(self):
        self._validate_is_open()
        result, capture_handle = k4a_module.playback_get_previous_capture(self._handle, self.thread_safe)
        self._verify_stream_error(result)
        return PyK4ACapture(
            calibration=self._calibration,
            capture_handle=capture_handle,
            color_format=self.configuration["color_format"],
            thread_safe=self.thread_safe,
        )

    def _validate_is_open(self):
        if not self._handle:
            raise K4AException("Playback not opened.")

    @staticmethod
    def _verify_stream_error(res: int):
        """
        Validate k4a_module result(k4a_stream_result_t)
        """
        result = StreamResult(res)
        if result == StreamResult.Failed:
            raise K4AException()
        elif result == StreamResult.EOF:
            raise EOFError()
