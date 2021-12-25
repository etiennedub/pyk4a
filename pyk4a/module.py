import os
import sys
from pathlib import Path

import pyk4a

try:
    import k4a_module  # noqa: F401
except BaseException as e:
    native_kinect_lib_dir = Path(os.path.abspath(os.path.dirname(pyk4a.__file__)))

    if sys.platform == "win32":
        try:
            from .win32_utils import add_dll_directory
            add_dll_directory(native_kinect_lib_dir)
            import k4a_module  # noqa: F401
        except BaseException:
            raise ImportError(
                (
                    "Cannot import k4a_module. "
                    f"DLL directory added was {native_kinect_lib_dir}. "
                    "You can provide a different path containing `k4a.dll`"
                    "using the environment variable `K4A_DLL_DIR`. "
                    "Also make sure pyk4a was properly built."
                )
            ) from e
    else:
        try:
            from ctypes import *
            import ctypes

            # todo: remove version numbers
            CDLL(str(native_kinect_lib_dir / "libk4a.so.1.4"), mode=ctypes.RTLD_GLOBAL)
            CDLL(str(native_kinect_lib_dir / "libdepthengine.so.2.0"), mode=ctypes.RTLD_GLOBAL)
            CDLL(str(native_kinect_lib_dir / "libk4arecord.so.1.4"), mode=ctypes.RTLD_GLOBAL)

            import k4a_module  # noqa: F401
        except BaseException:
            raise ImportError(
                (
                    "Cannot import k4a_module. "
                    "Make sure `libk4a.so` can be found. "
                    "Add the directory to your `LD_LIBRARY_PATH` if required. "
                    "Also make sure pyk4a is properly built."
                )
            ) from e
