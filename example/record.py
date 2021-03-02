from pyk4a.config import Config, ImageFormat
from pyk4a.pyk4a import PyK4A
from pyk4a.record import PyK4ARecord


config = Config(color_format=ImageFormat.COLOR_MJPG)
device = PyK4A(config=config)
device.start()

record = PyK4ARecord(device=device, config=config, path="/tmp/1.mkv")
record.create()
record.add_imu_track()
record.write_header()

captures = []
for _ in range(30 * 10):
    print(".", end="", flush=True)
    captures.append(device.get_capture())
print("Writing")
for i, capture in enumerate(captures):
    print(".", end="", flush=True)
    record.write_capture(capture)
    if i % 10 == 0:
        record.flush()
print("End")
record.flush()
record.close()
