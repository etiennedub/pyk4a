from argparse import Action, ArgumentParser, Namespace
from enum import Enum
from time import monotonic

from pyk4a import FPS, ColorResolution, Config, DepthMode, ImageFormat, PyK4A, WiredSyncMode


class EnumAction(Action):
    """
    Argparse action for handling Enums
    """

    def __init__(self, **kwargs):
        # Pop off the type value
        enum = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum, Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        kwargs.setdefault("choices", tuple(e.name for e in enum))

        super(EnumAction, self).__init__(**kwargs)

        self._enum = enum

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        setattr(namespace, self.dest, self._enum(values))


class EnumActionTuned(Action):
    """
    Argparse action for handling Enums
    """

    def __init__(self, **kwargs):
        # Pop off the type value
        enum = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum, Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        kwargs.setdefault("choices", tuple(e.name.split("_")[-1] for e in enum))

        super(EnumActionTuned, self).__init__(**kwargs)

        self._enum = enum

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        items = {item.name.split("_")[-1]: item.value for item in self._enum}
        setattr(namespace, self.dest, self._enum(items[values]))


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Camera captures transfer speed benchmark. \n"
        "You can check if you USB controller/cable has enough performance."
    )
    parser.add_argument("--device-id", type=int, default=0, help="Device ID, from zero. Default: 0")
    parser.add_argument(
        "--color-resolution",
        type=ColorResolution,
        action=EnumActionTuned,
        default=ColorResolution.RES_720P,
        help="Color sensor resoultion. Default: 720P",
    )
    parser.add_argument(
        "--color-format",
        type=ImageFormat,
        action=EnumActionTuned,
        default=ImageFormat.COLOR_BGRA32,
        help="Color color_image color_format. Default: BGRA32",
    )
    parser.add_argument(
        "--depth-mode",
        type=DepthMode,
        action=EnumAction,
        default=DepthMode.NFOV_UNBINNED,
        help="Depth sensor mode. Default: NFOV_UNBINNED",
    )
    parser.add_argument(
        "--camera-fps", type=FPS, action=EnumActionTuned, default=FPS.FPS_30, help="Camera FPS. Default: 30"
    )
    parser.add_argument(
        "--synchronized-images-only",
        action="store_true",
        dest="synchronized_images_only",
        help="Only synchronized color and depth images, default",
    )
    parser.add_argument(
        "--no-synchronized-images",
        action="store_false",
        dest="synchronized_images_only",
        help="Color and Depth images can be non synced.",
    )
    parser.set_defaults(synchronized_images_only=True)
    parser.add_argument(
        "--wired-sync-mode",
        type=WiredSyncMode,
        action=EnumActionTuned,
        default=WiredSyncMode.STANDALONE,
        help="Wired sync mode. Default: STANDALONE",
    )
    return parser.parse_args()


def bench(config: Config, device_id: int):
    device = PyK4A(config=config, device_id=device_id)
    device.start()
    depth = color = depth_period = color_period = 0
    print("Press CTRL-C top stop benchmark")
    started_at = started_at_period = monotonic()
    while True:
        try:
            capture = device.get_capture()
            if capture.color is not None:
                color += 1
                color_period += 1
            if capture.depth is not None:
                depth += 1
                depth_period += 1
            elapsed_period = monotonic() - started_at_period
            if elapsed_period >= 2:
                print(
                    f"Color: {color_period / elapsed_period:0.2f} FPS, Depth: {depth_period / elapsed_period: 0.2f} FPS"
                )
                color_period = depth_period = 0
                started_at_period = monotonic()
        except KeyboardInterrupt:
            break
    elapsed = monotonic() - started_at
    device.stop()
    print()
    print(f"Result: Color: {color / elapsed:0.2f} FPS, Depth: {depth / elapsed: 0.2f} FPS")


def main():
    args = parse_args()
    config = Config(
        color_resolution=args.color_resolution,
        color_format=args.color_format,
        depth_mode=args.depth_mode,
        synchronized_images_only=args.synchronized_images_only,
        wired_sync_mode=args.wired_sync_mode,
    )
    bench(config, args.device_id)


if __name__ == "__main__":
    main()
