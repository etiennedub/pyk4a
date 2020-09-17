import threading
from math import sin
from time import sleep
from typing import Dict, List

from pyk4a import PyK4A


class Worker(threading.Thread):
    def __init__(self):
        self._halt = False
        self._count = 0
        super().__init__()

    def halt(self):
        self._halt = True

    @property
    def count(self):
        return self._count


class CpuWorker(Worker):
    def run(self) -> None:
        while not self._halt:
            sin(self._count)
            self._count += 1


class CameraWorker(Worker):
    def __init__(self, device_id=0, thread_safe: bool = True):
        self._device_id = device_id
        self.thread_safe = thread_safe
        super().__init__()

    def run(self) -> None:
        print("Start run")
        camera = PyK4A(device_id=self._device_id, thread_safe=self.thread_safe)
        camera.start()
        while not self._halt:
            capture = camera.get_capture()
            assert capture.depth is not None
            self._count += 1
        sleep(0.1)
        camera.stop()
        del camera
        print("Stop run")


def bench(camera_workers: List[CameraWorker], cpu_workers: List[CpuWorker], duration: float) -> int:
    # start cameras
    for camera_worker in camera_workers:
        camera_worker.start()
        while not camera_worker.count:
            sleep(0.1)
            if not camera_worker.is_alive():
                print("Cannot start camera")
                exit(1)
    # start cpu-bound workers
    for cpu_worker in cpu_workers:
        cpu_worker.start()

    sleep(duration)

    for cpu_worker in cpu_workers:
        cpu_worker.halt()
    for camera_worker in camera_workers:
        camera_worker.halt()

    # wait while all workers stop
    workers: List[Worker] = [*camera_workers, *cpu_workers]
    while True:
        for worker in workers:
            if worker.is_alive():
                sleep(0.05)
                break
        else:
            break
    total = 0
    for cpu_worker in cpu_workers:
        total += cpu_worker.count
    return total


def draw(results: Dict[int, Dict[bool, int]]):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title("Threading performance")
    plt.ylabel("Operations Count")
    plt.xlabel("CPU Workers count")
    plt.plot(
        results.keys(), [result[True] for result in results.values()], "r", label="Thread safe",
    )
    plt.plot(
        results.keys(), [result[False] for result in results.values()], "g", label="Non thread safe",
    )
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title("Difference")
    plt.ylabel("Difference, %")
    plt.xlabel("CPU Workers count")
    plt.plot(
        results.keys(), [float(result[False] - result[True]) / result[True] * 100 for result in results.values()],
    )
    xmin, xmax, ymin, ymax = plt.axis()
    if ymin > 0:
        ymin = 0
        plt.axis([xmin, xmax, ymin, ymax])

    plt.show()


MAX_CPU_WORKERS_COUNT = 5
DURATION = 10
results: Dict[int, Dict[bool, int]] = {}
for cpu_workers_count in range(1, MAX_CPU_WORKERS_COUNT + 1):
    result: Dict[bool, int] = {}
    for thread_safe in (True, False):
        camera_workers = [CameraWorker(thread_safe=thread_safe)]
        cpu_workers = [CpuWorker() for i in range(cpu_workers_count)]
        operations = bench(camera_workers=camera_workers, cpu_workers=cpu_workers, duration=DURATION)
        print(f"Bench result: cpu_workers={cpu_workers_count}, " f"thread_safe={thread_safe}, operations={operations}")
        result[thread_safe] = operations

    percent = float(result[False] - result[True]) / result[True] * 100
    print(f"Difference: {percent: 0.2f} %")
    results[cpu_workers_count] = result

draw(results)
