import cv2

import pyk4a
from helpers import colorize
from pyk4a import Config, PyK4A


def main():
    k4a = PyK4A(Config(color_resolution=pyk4a.ColorResolution.RES_720P, depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,))
    k4a.start()

    while True:
        capture = k4a.get_capture()
        if capture.depth is not None:
            cv2.imshow("Depth", colorize(capture.depth, (None, 5000)))
        if capture.ir is not None:
            cv2.imshow("IR", colorize(capture.ir, (None, 500), colormap=cv2.COLORMAP_JET))
        if capture.color is not None:
            cv2.imshow("Color", capture.color)
        if capture.transformed_depth is not None:
            cv2.imshow("Transformed Depth", colorize(capture.transformed_depth, (None, 5000)))
        if capture.transformed_color is not None:
            cv2.imshow("Transformed Color", capture.transformed_color)
        if capture.transformed_ir is not None:
            cv2.imshow("Transformed IR", colorize(capture.transformed_ir, (None, 500), colormap=cv2.COLORMAP_JET))

        key = cv2.waitKey(10)
        if key != -1:
            cv2.destroyAllWindows()
            break

    k4a.stop()


if __name__ == "__main__":
    main()
