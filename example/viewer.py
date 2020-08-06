import pyk4a
from pyk4a import Config, PyK4A, ColorResolution

import cv2
import numpy as np

k4a = PyK4A(Config(color_resolution=ColorResolution.RES_720P,
                   # color_format=pyk4a.ColorFormat.MJPG,
                   depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
                   synchronized_images_only=True, ))
k4a.connect()

# getters and setters directly get and set on device
k4a.whitebalance = 4500
assert k4a.whitebalance == 4500
k4a.whitebalance = 4510
assert k4a.whitebalance == 4510

while 1:
    img_color = k4a.get_capture(color_only=True)
    if k4a._config.color_format == pyk4a.ColorFormat.MJPG:
        img_color = cv2.imdecode(img_color, cv2.IMREAD_COLOR)
    # img_color, img_depth = k4a.get_capture()  # Would also fetch the depth image
    if np.any(img_color):
        cv2.imshow('k4a', img_color[:, :, :3])
        key = cv2.waitKey(10)
        if key != -1:
            cv2.destroyAllWindows()
            break
k4a.disconnect()
