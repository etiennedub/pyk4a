from argparse import ArgumentParser
from json import dumps, loads

from pyk4a import PyK4APlayback


def info(playback: PyK4APlayback):
    print(f"Record length: {playback.length / 1000000: 0.2f} sec")

    calibration_str = playback.calibration_json
    calibration_formatted = dumps(loads(calibration_str), indent=2)
    print("=== Calibration ===")
    print(calibration_formatted)


def main() -> None:
    parser = ArgumentParser(description="pyk4a player")
    parser.add_argument(
        "FILE", type=str, help="Path to MKV file written by k4arecorder"
    )

    args = parser.parse_args()
    filename: str = args.FILE

    playback = PyK4APlayback(filename)
    playback.open()

    info(playback)

    playback.close()


if __name__ == "__main__":
    main()
