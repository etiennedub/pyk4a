import pyk4a
from pyk4a import Config, PyK4A, ColorResolution

import cv2
import numpy as np


def get_color_image_size(config):
    k4a = PyK4A(config)
    k4a.connect()
    count = 0
    while count < 5:
        img_color = k4a.get_capture(color_only=True)
        if np.any(img_color):
            count += 1
    return img_color.nbytes


if __name__ == "__main__":
    config_BGRA32 = Config(color_format=pyk4a.ColorFormat.BGRA32)
    config_MJPG = Config(color_format=pyk4a.ColorFormat.MJPG)

    nbytes_BGRA32 = get_color_image_size(config_BGRA32)
    nbytes_MJPG = get_color_image_size(config_MJPG)

    print(f"{nbytes_BGRA32=} {nbytes_MJPG=}")
    print(f"BGRA32 is {nbytes_BGRA32/nbytes_MJPG} larger")

    # output:
    # nbytes_BGRA32=3686400 nbytes_MJPG=229693
    # BGRA32 is 16.04924834452944 larger