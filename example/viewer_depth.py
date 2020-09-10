import cv2
import numpy as np

import pyk4a
from pyk4a import Config, PyK4A


def main():
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.OFF,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=False,
        )
    )
    k4a.connect()

    # getters and setters directly get and set on device
    k4a.whitebalance = 4500
    assert k4a.whitebalance == 4500
    k4a.whitebalance = 4510
    assert k4a.whitebalance == 4510

    while True:
        capture = k4a.get_capture()
        if np.any(capture.depth):
            # Clip 5000 mm(5 meters)
            clipped_depth = capture.depth.clip(None, 5000)
            # normalize and convert to 8bit
            normalized_depth = cv2.normalize(clipped_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # coloring image by choosed color map
            colored_depth = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_HSV)
            cv2.imshow("k4a", colored_depth)
            key = cv2.waitKey(10)
            if key != -1:
                cv2.destroyAllWindows()
                break
    k4a.disconnect()


if __name__ == "__main__":
    main()
