import cv2
import numpy as np

import pyk4a
from helpers import convert_to_bgra_if_required
from pyk4a import Config, PyK4A


def get_color_image_size(config, imshow=True):
    if imshow:
        cv2.namedWindow("k4a")
    k4a = PyK4A(config)
    k4a.start()
    count = 0
    while count < 60:
        capture = k4a.get_capture()
        if np.any(capture.color):
            count += 1
            if imshow:
                cv2.imshow("k4a", convert_to_bgra_if_required(config.color_format, capture.color))
                cv2.waitKey(10)
    cv2.destroyAllWindows()
    k4a.stop()
    return capture.color.nbytes


if __name__ == "__main__":
    imshow = True
    config_BGRA32 = Config(color_format=pyk4a.ImageFormat.COLOR_BGRA32)
    config_MJPG = Config(color_format=pyk4a.ImageFormat.COLOR_MJPG)
    config_NV12 = Config(color_format=pyk4a.ImageFormat.COLOR_NV12)
    config_YUY2 = Config(color_format=pyk4a.ImageFormat.COLOR_YUY2)

    nbytes_BGRA32 = get_color_image_size(config_BGRA32, imshow=imshow)
    nbytes_MJPG = get_color_image_size(config_MJPG, imshow=imshow)
    nbytes_NV12 = get_color_image_size(config_NV12, imshow=imshow)
    nbytes_YUY2 = get_color_image_size(config_YUY2, imshow=imshow)

    print(f"BGRA32: {nbytes_BGRA32},  MJPG: {nbytes_MJPG}, NV12: {nbytes_NV12}, YUY2: {nbytes_YUY2}")
    print(f"BGRA32 is {nbytes_BGRA32/nbytes_MJPG:0.2f} larger than MJPG")

    # output:
    # nbytes_BGRA32=3686400 nbytes_MJPG=229693
    # COLOR_BGRA32 is 16.04924834452944 larger
