from enum import IntEnum
from typing import Optional, Tuple


# k4a_fps_t
class FPS(IntEnum):
    FPS_5 = 0
    FPS_15 = 1
    FPS_30 = 2


# k4a_image_format_t
class ImageFormat(IntEnum):
    COLOR_MJPG = 0
    COLOR_NV12 = 1
    COLOR_YUY2 = 2
    COLOR_BGRA32 = 3
    DEPTH16 = 4
    IR16 = 5
    CUSTOM8 = 6
    CUSTOM16 = 7
    CUSTOM = 8


# k4a_depth_mode_t
class DepthMode(IntEnum):
    OFF = 0
    NFOV_2X2BINNED = 1
    NFOV_UNBINNED = 2
    WFOV_2X2BINNED = 3
    WFOV_UNBINNED = 4
    PASSIVE_IR = 5


# k4a_color_resolution_t
class ColorResolution(IntEnum):
    OFF = 0
    RES_720P = 1
    RES_1080P = 2
    RES_1440P = 3
    RES_1536P = 4
    RES_2160P = 5
    RES_3072P = 6


# k4a_wired_sync_mode_t
class WiredSyncMode(IntEnum):
    STANDALONE = 0
    MASTER = 1
    SUBORDINATE = 2


# k4a_color_control_command_t
class ColorControlCommand(IntEnum):
    EXPOSURE_TIME_ABSOLUTE = 0
    AUTO_EXPOSURE_PRIORITY = 1  # deprecated
    BRIGHTNESS = 2
    CONTRAST = 3
    SATURATION = 4
    SHARPNESS = 5
    WHITEBALANCE = 6
    BACKLIGHT_COMPENSATION = 7
    GAIN = 8
    POWERLINE_FREQUENCY = 9


# k4a_color_control_mode_t
class ColorControlMode(IntEnum):
    AUTO = 0
    MANUAL = 1


class Config:
    def __init__(
        self,
        color_resolution: ColorResolution = ColorResolution.RES_720P,
        color_format: ImageFormat = ImageFormat.COLOR_BGRA32,
        depth_mode: DepthMode = DepthMode.NFOV_UNBINNED,
        camera_fps: FPS = FPS.FPS_30,
        synchronized_images_only: bool = True,
        depth_delay_off_color_usec: int = 0,
        wired_sync_mode: WiredSyncMode = WiredSyncMode.STANDALONE,
        subordinate_delay_off_master_usec: int = 0,
        disable_streaming_indicator: bool = False,
    ):
        self.color_resolution = color_resolution
        self.color_format = color_format
        self.depth_mode = depth_mode
        self.camera_fps = camera_fps
        self.synchronized_images_only = synchronized_images_only
        self.depth_delay_off_color_usec = depth_delay_off_color_usec
        self.wired_sync_mode = wired_sync_mode
        self.subordinate_delay_off_master_usec = subordinate_delay_off_master_usec
        self.disable_streaming_indicator = disable_streaming_indicator
        assert self.subordinate_delay_off_master_usec >= 0

    def unpack(self) -> Tuple[ImageFormat, ColorResolution, DepthMode, FPS, bool, int, WiredSyncMode, int, bool]:
        return (
            self.color_format,
            self.color_resolution,
            self.depth_mode,
            self.camera_fps,
            self.synchronized_images_only,
            self.depth_delay_off_color_usec,
            self.wired_sync_mode,
            self.subordinate_delay_off_master_usec,
            self.disable_streaming_indicator,
        )


class BTSensorOrientation(IntEnum):
    DEFAULT = 0
    CLOCKWISE90 = 1
    COUNTERCLOCKWISE90 = 2
    FLIP180 = 3


class BTConfig:
    def __init__(
        self,
        sensor_orientation: BTSensorOrientation = BTSensorOrientation.DEFAULT,
        gpu_device_id: Optional[int] = None,
    ):
        self.sensor_orientation = sensor_orientation
        self.gpu_device_id = gpu_device_id
        self.tracker_processing_mode = self.gpu_device_id is None
