import glob
import os
import shutil

from setuptools import setup, Extension
from pathlib import Path
import sys
import platform
from setuptools.command.build_ext import build_ext
from typing import Tuple, Optional, Dict

if sys.version_info[0] == 2:
    sys.exit("Python 2 is not supported.")

# Enables --editable install with --user
# https://github.com/pypa/pip/issues/7953
import site

site.ENABLE_USER_SITE = "--user" in sys.argv[1:]


# Bypass import numpy before running install_requires
# https://stackoverflow.com/questions/54117786/add-numpy-get-include-argument-to-setuptools-without-preinstalled-numpy
class get_numpy_include:
    def __str__(self):
        import numpy
        return numpy.get_include()


def detect_win32_sdk_include_and_library_dirs() -> Optional[Tuple[str, str]]:
    # get program_files path
    for k in ("ProgramFiles", "PROGRAMFILES"):
        if k in os.environ:
            program_files = Path(os.environ[k])
            break
    else:
        program_files = Path("C:\\Program Files\\")
    # search through program_files
    arch = os.getenv("PROCESSOR_ARCHITECTURE", "amd64")
    for dir in sorted(program_files.glob("Azure Kinect SDK v*"), reverse=True):
        include = dir / "sdk" / "include"
        lib = dir / "sdk" / "windows-desktop" / arch / "release" / "lib"
        if include.exists() and lib.exists():
            return str(include), str(lib)
    return None


def detect_and_insert_sdk_include_and_library_dirs(include_dirs, library_dirs) -> None:
    if sys.platform == "win32":
        r = detect_win32_sdk_include_and_library_dirs()
    else:
        # Only implemented for windows
        r = None

    if r is None:
        print("Automatic kinect SDK detection did not yield any results.")
    else:
        include_dir, library_dir = r
        print(f"Automatically detected kinect SDK. Adding include dir: {include_dir} and library dir {library_dir}.")
        include_dirs.insert(0, include_dir)
        library_dirs.insert(0, library_dir)


def bundle_release_libraries(package_data: Dict):
    system_name = platform.system()
    is_64_bit = platform.architecture()[0] == "32bit"
    arch = "x86" if is_64_bit else "amd64"

    # check if is arm processor
    if is_64_bit and platform.machine().startswith("arm"):
        arch = "arm64"

    # detect release folder by os
    if system_name == "Windows":
        # todo: define path directly because on
        #  build server we don't have to search the dir
        include_dir, library_dir = detect_win32_sdk_include_and_library_dirs()
        binary_ext = "*.dll"
        binary_dir = Path(library_dir).parent / "bin"

        # add libraries to package
        for file in glob.glob(str(Path(binary_dir) / binary_ext)):
            shutil.copy(file, package_name)

    elif system_name == "Linux":
        binary_ext = "*.so*"

        if arch == "arm64":
            binary_dir = Path("/usr/lib/aarch64-linux-gnu/")
        else:
            binary_dir = Path("/usr/lib/x86_64-linux-gnu/")

        # add linux specific libraries
        # version number is needed
        shutil.copy(binary_dir / "libk4a.so.1.4", package_name, follow_symlinks=True)
        shutil.copy(binary_dir / "libk4arecord.so.1.4", package_name, follow_symlinks=True)
        shutil.copy(binary_dir / "libk4a1.4" / "libdepthengine.so.2.0", package_name, follow_symlinks=True)
    else:
        raise Exception(f"OS {system_name} not supported.")

    package_data[package_name] = [binary_ext]


# include native libraries
package_name = "pyk4a"
package_data = {}

if "bdist_wheel" in sys.argv:
    print("adding native files to package")
    bundle_release_libraries(package_data)

include_dirs = [get_numpy_include()]
library_dirs = []
detect_and_insert_sdk_include_and_library_dirs(include_dirs, library_dirs)
module = Extension('k4a_module',
                   sources=['pyk4a/pyk4a.cpp'],
                   libraries=['k4a', 'k4arecord'],
                   include_dirs=include_dirs,
                   library_dirs=library_dirs
                   )

setup(
    ext_modules=[module],
    include_package_data=True,
    package_data=package_data
)
