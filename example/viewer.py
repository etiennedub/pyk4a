import pyk4a
from pyk4a import Config, PyK4A

import cv2
import numpy as np


def main():
    k4a = PyK4A(Config(color_resolution=pyk4a.ColorResolution.RES_720P,
                       depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
                       synchronized_images_only=True, ))
    k4a.connect()

    # getters and setters directly get and set on device
    k4a.whitebalance = 4500
    assert k4a.whitebalance == 4500
    k4a.whitebalance = 4510
    assert k4a.whitebalance == 4510

    while 1:
        capture = k4a.get_capture()
        if np.any(capture.color):
            cv2.imshow('k4a', capture.color[:, :, :3])
            key = cv2.waitKey(10)
            if key != -1:
                cv2.destroyAllWindows()
                break
    k4a.disconnect()


if __name__ == "__main__":
    main()