import sys


try:
    import k4a_module  # noqa: F401
except BaseException as e:
    if sys.platform == "win32":
        from .win32_utils import prepare_import_k4a_module

        added_dll_dir = prepare_import_k4a_module()
        try:
            import k4a_module  # noqa: F401
        except BaseException:
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
