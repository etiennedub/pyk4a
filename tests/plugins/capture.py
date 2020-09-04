from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Capture:
    color: Optional[np.ndarray]
    depth: Optional[np.ndarray]
    ir: Optional[np.ndarray]


def random_capture(color: bool = True, depth: bool = True, ir: bool = True) -> Capture:
    color_image: Optional[np.ndarray] = None
    depth_image: Optional[np.ndarray] = None
    ir_image: Optional[np.ndarray] = None
    if color:
        color_image = np.random.randint(low=0, high=255, size=(720, 1280, 4), dtype=np.uint8)
    if depth:
        depth_image = np.random.randint(low=0, high=15000, size=(576, 640), dtype=np.uint16)
    if ir:
        color_image = np.random.randint(low=0, high=3000, size=(576, 640), dtype=np.uint16)

    return Capture(color=color_image, depth=depth_image, ir=ir_image)
