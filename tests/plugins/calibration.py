from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass(frozen=True)
class CalibrationHandle:
    pass


@pytest.fixture()
def calibration_raw() -> str:
    json_file = Path(__file__).parent.parent / "assets" / "calibration.json"
    return json_file.read_text()
