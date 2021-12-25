import os
import sys
from pathlib import Path

import pyk4a

try:
    import k4a_module  # noqa: F401
except BaseException as e:

    added_dll_dir = Path(os.path.abspath(os.path.dirname(pyk4a.__file__)))

    try:
        from .win32_utils import add_dll_directory
        add_dll_directory(added_dll_dir)
        import k4a_module  # noqa: F401
    except BaseException:
        if sys.platform == "win32":
            raise ImportError(
                (
                    "Cannot import k4a_module. "
                    f"DLL directory added was {added_dll_dir}. "
                    "You can provide a different path containing `k4a.dll`"
                    "using the environment variable `K4A_DLL_DIR`. "
                    "Also make sure pyk4a was properly built."
                )
            ) from e
        else:
            raise ImportError(
                (
                    "Cannot import k4a_module. "
                    "Make sure `libk4a.so` can be found. "
                    "Add the directory to your `LD_LIBRARY_PATH` if required. "
                    "Also make sure pyk4a is properly built."
                )
            ) from e
