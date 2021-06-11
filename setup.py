import os

from setuptools import setup, Extension
from pathlib import Path
import sys
from setuptools.command.build_ext import build_ext
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


def _detect_kinect_sdk():
    if sys.platform == "win32":
        program_files = Path(os.getenv("ProgramFiles", "C:\\Program Files\\"))
        for dir in sorted(program_files.glob("Azure Kinect SDK v*"), reverse=True):
            include = dir / "sdk" / "include"
            arch = os.getenv("PROCESSOR_ARCHITECTURE", "amd64")
            lib = dir / "sdk" / "windows-desktop" / arch / "release"
            if include.exists() and lib.exists():
                return str(include), str(lib)
    return None, None


include_dirs = [get_numpy_include()]
library_dirs = []

kinect_include_dir, kinect_library_dir = _detect_kinect_sdk()
if kinect_include_dir:
    include_dirs.insert(0, kinect_include_dir)
if kinect_library_dir:
    library_dirs.insert(0, kinect_library_dir)

module = Extension('k4a_module',
                   sources=['pyk4a/pyk4a.cpp'],
                   libraries=['k4a', 'k4arecord'],
                   include_dirs=include_dirs,
                   library_dirs=library_dirs
                   )

setup(
    ext_modules=[module],
)
