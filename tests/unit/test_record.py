from pathlib import Path

import pytest

from pyk4a import Config, ImageFormat, K4AException, PyK4ACapture, PyK4APlayback, PyK4ARecord


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
    def test_bad_path(tmp_path: Path):
        path = tmp_path / "not-exists" / "file.mkv"
        record = PyK4ARecord(config=Config(), path=path)
        with pytest.raises(K4AException, match=r"Cannot create"):
            record.create()
        assert not record.created

    @staticmethod
    def test_create(record: PyK4ARecord):
        record.create()
        assert record.created

    @staticmethod
    def test_recreate(created_record: PyK4ARecord):
        with pytest.raises(K4AException, match=r"already"):
            created_record.create()


class TestClose:
    @staticmethod
    def test_not_created_record(record: PyK4ARecord):
        with pytest.raises(K4AException, match=r"not created"):
            record.close()

    @staticmethod
    def test_closing(created_record: PyK4ARecord):
        created_record.close()
        assert not created_record.created


class TestWriteHeader:
    @staticmethod
    def test_not_created_record(record: PyK4ARecord):
        with pytest.raises(K4AException, match=r"not created"):
            record.write_header()

    @staticmethod
    def test_double_writing(created_record: PyK4ARecord):
        created_record.write_header()
        with pytest.raises(K4AException, match=r"already written"):
            created_record.write_header()


class TestWriteCapture:
    @staticmethod
    def test_not_created_record(record: PyK4ARecord, capture: PyK4ACapture):
        with pytest.raises(K4AException, match=r"not created"):
            record.write_capture(capture)

    @staticmethod
    def test_header_created(created_record: PyK4ARecord, capture: PyK4ACapture):
        created_record.write_capture(capture)
        assert created_record.header_written

    @staticmethod
    def test_captures_count_increased(created_record: PyK4ARecord, capture: PyK4ACapture):
        created_record.write_capture(capture)
        assert created_record.captures_count == 1


class TestFlush:
    @staticmethod
    def test_not_created_record(record: PyK4ARecord):
        with pytest.raises(K4AException, match=r"not created"):
            record.flush()
