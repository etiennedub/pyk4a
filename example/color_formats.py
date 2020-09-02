import cv2
import numpy as np

import pyk4a
from pyk4a import Config, PyK4A


def get_color_image_size(config, imshow=True):
    if imshow:
        cv2.namedWindow("k4a")
    k4a = PyK4A(config)
    k4a.connect()
    count = 0
    while count < 60:
        capture = k4a.get_capture()
        if np.any(capture.color):
            count += 1
            if imshow:
                cv2.imshow("k4a", convert_to_bgra_if_required(k4a, capture.color))
                cv2.waitKey(10)
    cv2.destroyAllWindows()
    k4a.disconnect()
    return capture.color.nbytes


def convert_to_bgra_if_required(k4a, img_color):
    # examples for all possible pyk4a.ColorFormats
    if k4a._config.color_format == pyk4a.ColorFormat.MJPG:
        img_color = cv2.imdecode(img_color, cv2.IMREAD_COLOR)
    elif k4a._config.color_format == pyk4a.ColorFormat.NV12:
        img_color = cv2.cvtColor(img_color, cv2.COLOR_YUV2BGRA_NV12)
        # this also works and it explains how the NV12 color format is stored in memory
        # h, w = img_color.shape[0:2]
        # h = h // 3 * 2
        # luminance = img_color[:h]
        # chroma = img_color[h:, :w//2]
        # img_color = cv2.cvtColorTwoPlane(luminance, chroma, cv2.COLOR_YUV2BGRA_NV12)
    elif k4a._config.color_format == pyk4a.ColorFormat.YUY2:
        img_color = cv2.cvtColor(img_color, cv2.COLOR_YUV2BGRA_YUY2)
    return img_color


if __name__ == "__main__":
    imshow = True
    config_BGRA32 = Config(color_format=pyk4a.ColorFormat.BGRA32)
    config_MJPG = Config(color_format=pyk4a.ColorFormat.MJPG)
    config_NV12 = Config(color_format=pyk4a.ColorFormat.NV12)
    config_YUY2 = Config(color_format=pyk4a.ColorFormat.YUY2)

    nbytes_BGRA32 = get_color_image_size(config_BGRA32, imshow=imshow)
    nbytes_MJPG = get_color_image_size(config_MJPG, imshow=imshow)
    nbytes_NV12 = get_color_image_size(config_NV12, imshow=imshow)
    nbytes_YUY2 = get_color_image_size(config_YUY2, imshow=imshow)

    print(f"{nbytes_BGRA32} {nbytes_MJPG}")
    print(f"BGRA32 is {nbytes_BGRA32/nbytes_MJPG} larger")

    # output:
    # nbytes_BGRA32=3686400 nbytes_MJPG=229693
    # BGRA32 is 16.04924834452944 larger
