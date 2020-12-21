import sys

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui

import pyk4a
from pyk4a import ColorResolution, Config, PyK4A


# mapping from kinect's pose to openpose's coco pose
kinect2coco = [27, 3, 12, 13, 14, 5, 6, 7, 22, 23, 24, 18, 19, 20, 30, 28, 31, 29]
num_joints = len(kinect2coco)


# https://gist.github.com/markjay4k/da2f55e28514be7160a7c5fbf95bd243
class Visualizer(object):
    def __init__(self):
        self.k4a = PyK4A(Config(color_resolution=ColorResolution.RES_720P, depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,))
        self.k4a.start()

        self.traces = dict()
        self.app = QtGui.QApplication(sys.argv)
        self.w = gl.GLViewWidget()
        self.w.opts["distance"] = 15000
        self.w.setWindowTitle("Pose")
        self.w.setGeometry(0, 110, 1920, 1080)
        self.w.show()

        # create the background grids
        gy = gl.GLGridItem()
        gy.setSize(4000, 4000, 4000)
        gy.setSpacing(500, 500, 500)
        gy.rotate(90, 1, 0, 0)
        gy.translate(0, 1000, 0)
        self.w.addItem(gy)

        self.n = 10
        for i in range(self.n):
            pts = np.zeros((num_joints, 3))
            self.traces[i] = gl.GLScatterPlotItem(pos=pts, color=pg.glColor((i, self.n * 1.3)))
            self.w.addItem(self.traces[i])

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
            QtGui.QApplication.instance().exec_()

    def set_plotdata(self, name, points, color):
        self.traces[name].setData(pos=points, color=color)

    def update(self):
        capture = self.k4a.get_capture()
        pose = capture.body_skeleton
        if pose is not None:
            for i in range(pose.shape[0]):
                pts = pose[i, :, :3].reshape(-1, 3)[kinect2coco]
                self.set_plotdata(name=i, points=pts, color=pg.glColor((i, self.n * 0.3)))
            for i in range(pose.shape[0], self.n):
                self.set_plotdata(name=i, points=np.zeros((1, 3)), color=pg.glColor((i, self.n * 0.3)))

    def animation(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(20)
        self.start()


# Start Qt event loop unless running in interactive mode.
if __name__ == "__main__":
    v = Visualizer()
    v.animation()
