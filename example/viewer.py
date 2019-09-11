from pyk4a.config import Config
from pyk4a.pyk4a import PyK4A

import cv2
import numpy as np

k4a = PyK4A(Config())
k4a.connect()
while 1:
    img_color = k4a.device_get_capture(color_only=True)
    if np.any(img_color):
        cv2.imshow('k4a', img_color[:,:,:3])
        key = cv2.waitKey(10)
        if key != -1:
            cv2.destroyAllWindows()
            break
k4a.disconnect()
