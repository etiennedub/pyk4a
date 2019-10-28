import numpy as np
import cv2
import pyk4a
from pyk4a import Config, PyK4A, ColorResolution

#mapping from kinect's pose to openpose's coco pose
kinect2coco = [27, 3, 12, 13, 14, 5, 6, 7, 22, 23, 24, 18, 19, 20, 30, 28, 31, 29]
num_joints = len(kinect2coco)

k4a = PyK4A(Config(color_resolution=ColorResolution.RES_720P,
                   depth_mode=pyk4a.DepthMode.NFOV_UNBINNED))
k4a.connect()

while True:
    frame, _, pose = k4a.get_capture()
    source_H, source_W, _ = frame.shape
    frame = cv2.resize(frame, (256 * source_W // source_H, 256))
    target_H, target_W, _ = frame.shape

    if pose is not None and pose.shape[0] > 0:
        pts = pose[0, :, :2].reshape(-1, 2)[kinect2coco]
        pts[:, 0] = pts[:, 0]*(target_H/source_H)
        pts[:, 1] = pts[:, 1]*(target_W/source_W)

        for i in range(pts.shape[0]):
            cv2.circle(frame, (int(pts[i, 0]), int(pts[i, 1])), 15, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

k4a.disconnect()
cv2.destroyAllWindows()
