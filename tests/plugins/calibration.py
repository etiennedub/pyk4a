from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationHandle:
    depth_mode: int
    color_resolution: int
