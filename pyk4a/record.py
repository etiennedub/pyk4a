from pathlib import Path
from typing import Optional, Union

import k4a_module

from .capture import PyK4ACapture
from .config import Config
from .errors import K4AException
from .pyk4a import PyK4A
from .results import Result


class PyK4ARecord:
    def __init__(self, device: Optional[PyK4A], path: Union[str, Path], config: Config, thread_safe: bool = True):
        self._path: Path = Path(path)
        self.thread_safe = thread_safe
        self._device: Optional[PyK4A] = device
        self._config: Config = config
        self._handle: Optional[object] = None

    def __del__(self):
        if self._handle:
            self.close()

    def create(self) -> None:
        """ Create recording file """
        result, handle = k4a_module.record_create(
            self._device._device_handle, str(self._path), self.thread_safe, *self._config.unpack()
        )
        if result != Result.Success:
            raise K4AException(f"Cannot create record {self._path}")
        self._handle = handle

    def close(self):
        """ Close recording """
        self._validate_is_open()
        k4a_module.record_close(self._handle, self.thread_safe)
        self._handle = None

    def write_header(self):
        self._validate_is_open()
        result: Result = k4a_module.record_write_header(self._handle, self.thread_safe)
        if result != Result.Success:
            raise K4AException(f"Cannot write record header {self._path}")

    def write_capture(self, capture: PyK4ACapture):
        result: Result = k4a_module.record_write_capture(self._handle, capture._capture_handle, self.thread_safe)
        if result != Result.Success:
            raise K4AException(f"Cannot write capture {self._path}")

    def flush(self):
        self._validate_is_open()
        result: Result = k4a_module.record_flush(self._handle, self.thread_safe)
        if result != Result.Success:
            raise K4AException(f"Cannot flush data {self._path}")

    def _validate_is_open(self):
        if not self._handle:
            raise K4AException("Record not opened.")
