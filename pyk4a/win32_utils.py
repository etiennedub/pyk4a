import os
import sys
from pathlib import Path
from typing import Optional


if sys.platform != "win32":
    raise ImportError(f"This file should only be used on a win32 OS: {__file__}")


def add_dll_directory(path: Path):
    if hasattr(os, "add_dll_directory"):
        # only available for python 3.8+ on win32
        os.add_dll_directory(str(path))  # type: ignore
    else:
        from ctypes import c_wchar_p, windll  # type: ignore
        from ctypes.wintypes import DWORD

        AddDllDirectory = windll.kernel32.AddDllDirectory
        AddDllDirectory.restype = DWORD
        AddDllDirectory.argtypes = [c_wchar_p]
        AddDllDirectory(str(path))


def find_k4a_dll_dir() -> Optional[Path]:
    # get program_files
    for k in ("ProgramFiles", "PROGRAMFILES"):
        if k in os.environ:
            program_files = Path(os.environ[k])
            break
    else:
        program_files = Path("C:\\Program Files\\")
    # search through program_files
    arch = os.getenv("PROCESSOR_ARCHITECTURE", "amd64")
    for dir in sorted(program_files.glob("Azure Kinect SDK v*"), reverse=True):
        candidate = dir / "sdk" / "windows-desktop" / arch / "release" / "bin"
        dll = candidate / "k4a.dll"
        if dll.exists():
            dll_dir = candidate
    return dll_dir


def prepare_import_k4a_module() -> Optional[Path]:
    dll_dir: Optional[Path]
    if "K4A_DLL_DIR" in os.environ:
        dll_dir = Path(os.environ["K4A_DLL_DIR"])
    else:
        dll_dir = find_k4a_dll_dir()

    if dll_dir is not None:
        add_dll_directory(dll_dir)
    return dll_dir
