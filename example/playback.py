from argparse import ArgumentParser

import cv2

from helpers import colorize, convert_to_bgra_if_required
from pyk4a import PyK4APlayback


def info(playback: PyK4APlayback):
    print(f"Record length: {playback.length / 1000000: 0.2f} sec")


def play(playback: PyK4APlayback):
    while True:
        try:
            capture = playback.get_next_capture()
            if capture.color is not None:
                cv2.imshow("Color", convert_to_bgra_if_required(playback.configuration["color_format"], capture.color))
            if capture.depth is not None:
                cv2.imshow("Depth", colorize(capture.depth, (None, 5000)))
            key = cv2.waitKey(10)
            if key != -1:
                break
        except EOFError:
            break
    cv2.destroyAllWindows()


def main() -> None:
    parser = ArgumentParser(description="pyk4a player")
    parser.add_argument("--seek", type=float, help="Seek file to specified offset in seconds", default=0.0)
    parser.add_argument("FILE", type=str, help="Path to MKV file written by k4arecorder")

    args = parser.parse_args()
    filename: str = args.FILE
    offset: float = args.seek

    playback = PyK4APlayback(filename)
    playback.open()

    info(playback)

    if offset != 0.0:
        playback.seek(int(offset * 1000000))
    play(playback)

    playback.close()


if __name__ == "__main__":
    main()
