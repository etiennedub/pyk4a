from pathlib import Path
from typing import Optional, Union

from .capture import PyK4ACapture
from .config import Config
from .errors import K4AException
from .module import k4a_module
from .pyk4a import PyK4A
from .results import Result


class PyK4ARecord:
    def __init__(
        self, path: Union[str, Path], config: Config, device: Optional[PyK4A] = None, thread_safe: bool = True
    ):
        self._path: Path = Path(path)
        self.thread_safe = thread_safe
        self._device: Optional[PyK4A] = device
        self._config: Config = config
        self._handle: Optional[object] = None
        self._header_written: bool = False
        self._captures_count: int = 0

    def __del__(self):
        if self.created:
            self.close()

    def create(self) -> None:
        """Create record file"""
        if self.created:
            raise K4AException(f"Record already created {self._path}")
        device_handle = self._device._device_handle if self._device else None
        result, handle = k4a_module.record_create(
            device_handle, str(self._path), self.thread_safe, *self._config.unpack()
        )
        if result != Result.Success:
            raise K4AException(f"Cannot create record {self._path}")
        self._handle = handle

    def close(self):
        """Close record"""
        self._validate_is_created()
        k4a_module.record_close(self._handle, self.thread_safe)
        self._handle = None

    def write_header(self):
        """Write MKV header"""
        self._validate_is_created()
        if self.header_written:
            raise K4AException(f"Header already written {self._path}")
        result: Result = k4a_module.record_write_header(self._handle, self.thread_safe)
        if result != Result.Success:
            raise K4AException(f"Cannot write record header {self._path}")
        self._header_written = True

    def write_capture(self, capture: PyK4ACapture):
        """Write capture to file (send to queue)"""
        self._validate_is_created()
        if not self.header_written:
            self.write_header()
        result: Result = k4a_module.record_write_capture(self._handle, capture._capture_handle, self.thread_safe)
        if result != Result.Success:
            raise K4AException(f"Cannot write capture {self._path}")
        self._captures_count += 1

    def flush(self):
        """Flush queue"""
        self._validate_is_created()
        result: Result = k4a_module.record_flush(self._handle, self.thread_safe)
        if result != Result.Success:
            raise K4AException(f"Cannot flush data {self._path}")

    @property
    def created(self) -> bool:
        return self._handle is not None

    @property
    def header_written(self) -> bool:
        return self._header_written

    @property
    def captures_count(self) -> int:
        return self._captures_count

    @property
    def path(self) -> Path:
        return self._path

    def _validate_is_created(self):
        if not self.created:
            raise K4AException("Record not created.")
