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


class TestAddTag:
    @staticmethod
    def test_adding_tag(created_record: PyK4ARecord, capture: PyK4ACapture):
        created_record.add_tag("MYTAG", "Some tag value")
        created_record.write_header()
        created_record.write_capture(capture)
        created_record.close()
        bytes = created_record.path.read_bytes()
        # very simple
        assert b"MYTAG" in bytes, f"Tag MYTAG not found in {created_record.path}"
        assert b"Some tag value" in bytes, f"Tag value not found in {created_record.path}"


class TestAddAttachment:
    @staticmethod
    def test_adding_tag(created_record: PyK4ARecord, capture: PyK4ACapture):
        created_record.add_attachment("file.txt", b"Some text value")
        created_record.write_header()
        created_record.write_capture(capture)
        created_record.close()
        bytes = created_record.path.read_bytes()
        # very simple
        assert b"file.txt" in bytes, f"Attached file.txt not found in {created_record.path}"
        assert b"Some text value" in bytes, f"Attached file content not found in {created_record.path}"
