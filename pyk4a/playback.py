from enum import IntEnum
from pathlib import Path
from typing import Optional, Union

import k4a_module

from .pyk4a import K4AException, Result


# k4a_buffer_result_t
class BufferResult(IntEnum):
    Success = 0
    Failed = 1
    TooSmall = 2


# k4a_stream_result_t
class StreamResult(IntEnum):
    Success = 0
    Failed = 1
    EOF = 2


class SeekOrigin(IntEnum):
    BEGIN = 0
    END = 1
    DEVICE_TIME = 2


class PyK4APlayback:
    def __init__(self, path: Union[str, Path], thread_safe: bool = True):
        self._path: Path = Path(path)
        self._thread_safe = thread_safe
        self._handle: Optional[object] = None
        self._length: Optional[int] = None
        self._calibration_json: Optional[str] = None

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
    def length(self) -> int:
        """
            Record length in usec
        """
        if self._length is None:
            self._validate_is_open()
            self._length = k4a_module.playback_get_recording_length_usec(
                self._handle, self._thread_safe
            )
        return self._length

    @property
    def calibration_json(self) -> str:
        """
            Calibration parameters as JSON string
        """
        if self._calibration_json is None:
            self._validate_is_open()
            result, self._calibration_json = k4a_module.playback_get_calibration(
                self._handle, self._thread_safe
            )
            typed_result = BufferResult(result)
            if typed_result != BufferResult.Success:  # pragma: no cover
                raise K4AException(f"Cannot read calibration from file: {typed_result}")
        return self._calibration_json

    def open(self) -> None:
        """
            Open record file
        """
        result, handle = k4a_module.playback_open(str(self._path), self._thread_safe)
        if Result(result) != Result.Success:
            raise K4AException(f"Cannot open file {self._path}")

        self._handle = handle

    def close(self):
        """
            Close record file
        """
        self._validate_is_open()
        k4a_module.playback_close(self._handle, self._thread_safe)
        self._handle = None

    def seek(self, offset: int, origin: SeekOrigin = SeekOrigin.BEGIN) -> None:
        """
            Seek playback pointer to specified offset
        """
        self._validate_is_open()
        result = k4a_module.playback_seek_timestamp(
            self._handle, self._thread_safe, offset, int(origin)
        )
        typed_result = StreamResult(result)
        if typed_result != StreamResult.Success:
            raise K4AException(f"Cannot seek to specified position: {typed_result}")

    def _validate_is_open(self):
        if not self._handle:
            raise K4AException("Playback not opened.")
