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
            if include.exists():
                return dir.resolve()
    return None

class BuildExt(build_ext):
    user_options = build_ext.user_options + [
        ('kinect-sdk=', None ,"Kinect SDK dir for windows, leave empty for autodetect")]
    kinect_sdk = None



    def __init__(self, *args, **kwargs):
        print(111)
        super().__init__(*args, **kwargs)

    def run(self):
        if sys.platform == "win32":
            kinect_sdk = Path(self.kinect_sdk) if self.kinect_sdk else _detect_kinect_sdk()
            if kinect_sdk:
                arch = os.getenv("PROCESSOR_ARCHITECTURE", "amd64")
                include_path = kinect_sdk / "sdk" / "include"
                lib_path = kinect_sdk / "windows-desktop" / arch / "release" / "lib"
                self.include_dirs.append(include_path)
                self.library_dirs.append(lib_path)

        super().run()

module = Extension('k4a_module',
                   sources=['pyk4a/pyk4a.cpp'],
                   include_dirs=[get_numpy_include()],
                   libraries=['k4a', 'k4arecord'])

setup(
    ext_modules=[module],
    cmdclass={
        'build_ext': BuildExt
    }
)
