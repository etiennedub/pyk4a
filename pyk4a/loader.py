import sys


if sys.platform == "win32":
    import os
    from pathlib import Path

    def _add_dll_directory(path: Path):
        from ctypes import c_wchar_p, windll  # type: ignore
        from ctypes.wintypes import DWORD

        AddDllDirectory = windll.kernel32.AddDllDirectory
        AddDllDirectory.restype = DWORD
        AddDllDirectory.argtypes = [c_wchar_p]
        AddDllDirectory(str(path))

    program_files = Path(os.getenv("ProgramFiles", "C:\\Program Files\\"))
    for dir in sorted(program_files.glob("Azure Kinect SDK v*"), reverse=True):
        candidate = dir / "sdk" / "windows-desktop" / "amd64" / "release" / "bin"
        dll = candidate / "k4a.dll"
        if dll.exists():
            _add_dll_directory(candidate)
            break

try:
    import k4a_module  # noqa: F401
except BaseException as e:
    raise RuntimeError(f"Cannot load k4a_module, maybe kinect SDK is not available: {e}")
