from pathlib import Path
from typing import Iterator

import pytest

from pyk4a import K4AException, PyK4APlayback


@pytest.fixture()
def playback(recording_good_file: str) -> Iterator[PyK4APlayback]:
    playback = PyK4APlayback(path=recording_good_file)
    yield playback
    # autoclose
    try:
        playback.close()
    except K4AException:
        pass


@pytest.fixture()
def playback_bad(recording_bad_file: str) -> Iterator[PyK4APlayback]:
    playback = PyK4APlayback(path=recording_bad_file)
    yield playback
    # autoclose
    try:
        playback.close()
    except K4AException:
        pass


class TestInit:
    @staticmethod
    def test_path_argument():
        playback = PyK4APlayback(path=Path("some.mkv"))
        assert playback.path == Path("some.mkv")

        playback = PyK4APlayback(path="some.mkv")
        assert playback.path == Path("some.mkv")

    @staticmethod
    def test_not_existing_path(recording_not_exists_file: str):
        playback = PyK4APlayback(path=recording_not_exists_file)
        with pytest.raises(K4AException):
            playback.open()


class TestPropertyLength:
    @staticmethod
    def test_validate_if_record_opened(playback: PyK4APlayback):
        with pytest.raises(K4AException, match="Playback not opened."):
            playback.length

    @staticmethod
    def test_good_file(playback: PyK4APlayback):
        playback.open()
        assert playback.length == 1234


class TestPropertyCalibrationJson:
    @staticmethod
    def test_validate_if_record_opened(playback: PyK4APlayback):
        with pytest.raises(K4AException, match="Playback not opened."):
            playback.calibration_json

    @staticmethod
    def test_good_file(playback: PyK4APlayback):
        playback.open()
        assert playback.calibration_json

    @staticmethod
    def test_bad_file(playback_bad: PyK4APlayback):
        playback_bad.open()
        with pytest.raises(K4AException, match=r"Cannot read calibration from file"):
            playback_bad.calibration_json


class TestSeek:
    @staticmethod
    def test_validate_if_record_opened(playback: PyK4APlayback):
        with pytest.raises(K4AException, match="Playback not opened."):
            playback.seek(1)

    @staticmethod
    def test_bad_file(playback_bad: PyK4APlayback):
        playback_bad.open()
        with pytest.raises(K4AException, match=r"Cannot seek to specified position"):
            playback_bad.seek(10)

    @staticmethod
    def test_good_file(playback: PyK4APlayback):
        playback.open()
        playback.seek(10)

    @staticmethod
    def test_seek_eof(playback: PyK4APlayback):
        playback.open()
        with pytest.raises(K4AException, match=r"Cannot seek to specified position"):
            playback.seek(9999)