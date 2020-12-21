import cv2

import pyk4a
from helpers import colorize
from pyk4a import ColorResolution, Config, PyK4A


k4a = PyK4A(
    Config(
        color_resolution=ColorResolution.RES_720P, depth_mode=pyk4a.DepthMode.NFOV_UNBINNED, camera_fps=pyk4a.FPS.FPS_5,
    )
)
k4a.start()

while True:
    capture = k4a.get_capture()
    body_skeleton = capture.body_skeleton

    if body_skeleton is None:
        continue

    frame = colorize(capture.depth, (None, 5000))

    if body_skeleton is not None and body_skeleton.shape[0] > 0:
        pts = body_skeleton[0, :, :]
        for i in range(1):
            try:
                x, y = int(pts[i, -2]), int(pts[i, -1])
                print(x, y)
                assert x > 0 and y > 0
                cv2.circle(frame, (x, y), 12, (0, 0, 0), thickness=-1, lineType=cv2.FILLED)
            except Exception:
                pass

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

k4a.stop()
cv2.destroyAllWindows()
