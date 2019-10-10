from enum import Enum, IntEnum


# k4a_fps_t
class FPS(IntEnum):
    FPS_5 = 0
    FPS_15 = 1
    FPS_30 = 2


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
    def __init__(self,
                 color_resolution=ColorResolution.RES_720P,
                 depth_mode=DepthMode.NFOV_UNBINNED,
                 camera_fps=FPS.FPS_30,
                 synchronized_images_only=False,
                 depth_delay_off_color_usec=0,
                 wired_sync_mode=WiredSyncMode.STANDALONE,
                 subordinate_delay_off_master_usec=0,
                 disable_streaming_indicator=False):
        self._color_resolution = color_resolution
        self._depth_mode = depth_mode
        self._camera_fps = camera_fps
        self._synchronized_images_only = synchronized_images_only
        self._depth_delay_off_color_usec = depth_delay_off_color_usec
        self._wired_sync_mode = wired_sync_mode
        self._subordinate_delay_off_master_usec = subordinate_delay_off_master_usec
        self._disable_streaming_indicator = disable_streaming_indicator
        self._color_format = 3  # BGRA32

    def unpack(self):
        return (self._color_format,
                self._color_resolution,
                self._depth_mode,
                self._camera_fps,
                self._synchronized_images_only,
                self._depth_delay_off_color_usec,
                self._wired_sync_mode,
                self._subordinate_delay_off_master_usec,
                self._disable_streaming_indicator)
