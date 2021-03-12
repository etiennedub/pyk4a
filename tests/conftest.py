from pathlib import Path

import pytest

from pyk4a import PyK4APlayback


pytest_plugins = [
    "tests.plugins.calibration",
    "tests.plugins.capture",
    "tests.plugins.device",
    "tests.plugins.playback",
]


@pytest.fixture()
def recording_path() -> Path:
    return Path(__file__).parent / "assets" / "recording.mkv"


@pytest.fixture()
def playback(recording_path: Path) -> PyK4APlayback:
    return PyK4APlayback(path=recording_path)
