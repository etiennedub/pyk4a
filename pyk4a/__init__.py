from .calibration import Calibration, CalibrationType
from .capture import PyK4ACapture
from .config import (
    FPS,
    ColorControlCommand,
    ColorControlMode,
    ColorResolution,
    Config,
    DepthMode,
    ImageFormat,
    WiredSyncMode,
)
from .errors import K4AException, K4ATimeoutException
from .module import k4a_module
from .playback import PyK4APlayback, SeekOrigin
from .pyk4a import ColorControlCapabilities, PyK4A
from .record import PyK4ARecord
from .transformation import (
    color_image_to_depth_camera,
    depth_image_to_color_camera,
    depth_image_to_color_camera_custom,
    depth_image_to_point_cloud,
)


def connected_device_count() -> int:
    return k4a_module.device_get_installed_count()


__all__ = (
    "Calibration",
    "CalibrationType",
    "FPS",
    "ColorControlCommand",
    "ColorControlMode",
    "ImageFormat",
    "ColorResolution",
    "Config",
    "DepthMode",
    "WiredSyncMode",
    "K4AException",
    "K4ATimeoutException",
    "PyK4A",
    "PyK4ACapture",
    "PyK4APlayback",
    "SeekOrigin",
    "ColorControlCapabilities",
    "color_image_to_depth_camera",
    "depth_image_to_point_cloud",
    "depth_image_to_color_camera",
    "depth_image_to_color_camera_custom",
    "PyK4ARecord",
    "connected_device_count",
)
