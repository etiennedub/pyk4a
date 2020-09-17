import matplotlib.pyplot as plt
import numpy as np

import pyk4a
from pyk4a import Config, PyK4A


MAX_SAMPLES = 1000


def set_default_data(data):
    data["acc_x"] = MAX_SAMPLES * [data["acc_x"][-1]]
    data["acc_y"] = MAX_SAMPLES * [data["acc_y"][-1]]
    data["acc_z"] = MAX_SAMPLES * [data["acc_z"][-1]]
    data["gyro_x"] = MAX_SAMPLES * [data["acc_x"][-1]]
    data["gyro_y"] = MAX_SAMPLES * [data["acc_y"][-1]]
    data["gyro_z"] = MAX_SAMPLES * [data["acc_z"][-1]]


def main():
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True,
        )
    )
    k4a.start()

    plt.ion()
    fig, axes = plt.subplots(3, sharex=False)

    data = {
        "temperature": [0] * MAX_SAMPLES,
        "acc_x": [0] * MAX_SAMPLES,
        "acc_y": [0] * MAX_SAMPLES,
        "acc_z": [0] * MAX_SAMPLES,
        "acc_timestamp": [0] * MAX_SAMPLES,
        "gyro_x": [0] * MAX_SAMPLES,
        "gyro_y": [0] * MAX_SAMPLES,
        "gyro_z": [0] * MAX_SAMPLES,
        "gyro_timestamp": [0] * MAX_SAMPLES,
    }
    y = np.zeros(MAX_SAMPLES)
    lines = {
        "temperature": axes[0].plot(y, label="temperature")[0],
        "acc_x": axes[1].plot(y, label="acc_x")[0],
        "acc_y": axes[1].plot(y, label="acc_y")[0],
        "acc_z": axes[1].plot(y, label="acc_z")[0],
        "gyro_x": axes[2].plot(y, label="gyro_x")[0],
        "gyro_y": axes[2].plot(y, label="gyro_y")[0],
        "gyro_z": axes[2].plot(y, label="gyro_z")[0],
    }

    for i in range(MAX_SAMPLES):
        sample = k4a.get_imu_sample()
        sample["acc_x"], sample["acc_y"], sample["acc_z"] = sample.pop("acc_sample")
        sample["gyro_x"], sample["gyro_y"], sample["gyro_z"] = sample.pop("gyro_sample")
        for k, v in sample.items():
            data[k][i] = v
        if i == 0:
            set_default_data(data)

        for k in ("temperature", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"):
            lines[k].set_data(range(MAX_SAMPLES), data[k])

        acc_y = data["acc_x"] + data["acc_y"] + data["acc_z"]
        gyro_y = data["gyro_x"] + data["gyro_y"] + data["gyro_z"]
        lines["acc_x"].axes.set_ylim(min(acc_y), max(acc_y))
        lines["gyro_x"].axes.set_ylim(min(gyro_y), max(gyro_y))
        lines["temperature"].axes.set_ylim(min(data["temperature"][0 : i + 1]), max(data["temperature"][0 : i + 1]))

        fig.canvas.draw()
        fig.canvas.flush_events()

    k4a._stop_imu()
    k4a.stop()


if __name__ == "__main__":
    main()
