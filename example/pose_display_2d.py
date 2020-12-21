import cv2

import pyk4a
from helpers import colorize
from pyk4a import ColorResolution, Config, PyK4A


k4a = PyK4A(
    Config(
        color_resolution=ColorResolution.RES_720P,
        depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
        camera_fps=pyk4a.FPS.FPS_30,
    )
)
k4a.start()

while True:
    capture = k4a.get_capture()
    body_skeleton = capture.body_skeleton

    if body_skeleton is None:
        continue

    frame = colorize(capture.transformed_depth, (None, 5000))

    if body_skeleton is None:
        continue
    for body_index in range(body_skeleton.shape[0]):
        skeleton = body_skeleton[body_index, :, :]
        for joint_index in range(skeleton.shape[0]):
            try:
                valid = int(skeleton[joint_index, -1])
                if valid != 1:
                    continue
                x, y = skeleton[joint_index, (-3, -2)].astype(int)
                cv2.circle(frame, (x, y), 12, (50, 50, 50), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frame, str(joint_index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            except Exception as e:
                print(e)
                pass

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

k4a.stop()
cv2.destroyAllWindows()
