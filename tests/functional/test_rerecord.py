from pathlib import Path

import pytest

from pyk4a import Config, PyK4APlayback, PyK4ARecord


@pytest.fixture()
def record_path(tmp_path: Path) -> Path:
    return "/tmp/1.mkv"  # tmp_path / "record.mkv"


def test_transfering_calibration(playback: PyK4APlayback, record_path: Path):
    playback.open()
    calibration_raw = playback.calibration_raw
    config = Config(
        color_format=playback.configuration["color_format"],
        color_resolution=playback.configuration["color_resolution"],
        depth_mode=playback.configuration["depth_mode"],
        camera_fps=playback.configuration["camera_fps"],
        depth_delay_off_color_usec=playback.configuration["depth_delay_off_color_usec"],
        wired_sync_mode=playback.configuration["wired_sync_mode"],
    )

    record = PyK4ARecord(record_path, config=config)
    record.create()
    record.add_tag("K4A_CALIBRATION_FILED", "calibration.json")
    record.add_attachment("calibration.json", calibration_raw.encode())
    while True:
        try:
            capture = playback.get_next_capture()
            record.write_capture(capture)
        except EOFError:
            break
    playback.close()
    record.close()
    replay = PyK4APlayback(record_path)
    replay.open()
    assert replay.calibration_raw == calibration_raw, "Calibrations doesn't transferred"
