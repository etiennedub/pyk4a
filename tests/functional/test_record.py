from pathlib import Path

import pytest

from pyk4a import Config, ImageFormat, PyK4ACapture, PyK4APlayback, PyK4ARecord


@pytest.fixture()
def record_path(tmp_path: Path) -> Path:
    return tmp_path / "record.mkv"


@pytest.fixture()
def record(record_path: Path) -> PyK4ARecord:
    return PyK4ARecord(config=Config(color_format=ImageFormat.COLOR_MJPG), path=record_path)


@pytest.fixture()
def created_record(record: PyK4ARecord) -> PyK4ARecord:
    record.create()
    return record


@pytest.fixture()
def capture(playback: PyK4APlayback) -> PyK4ACapture:
    playback.open()
    return playback.get_next_capture()


class TestCreate:
    @staticmethod
    def test_create(record: PyK4ARecord):
        record.create()
        assert record.path.exists()


class TestClose:
    @staticmethod
    def test_closing(created_record: PyK4ARecord):
        created_record.close()
        assert created_record.path.exists()


class TestFlush:
    @staticmethod
    def test_file_size_increased(created_record: PyK4ARecord, capture: PyK4ACapture):
        created_record.write_header()
        size_before = created_record.path.stat().st_size
        created_record.write_capture(capture)
        created_record.flush()
        size_after = created_record.path.stat().st_size
        assert size_after > size_before
