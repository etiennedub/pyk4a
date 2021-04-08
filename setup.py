from setuptools import setup, Extension

import sys
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


module = Extension('k4a_module',
                   sources=['pyk4a/pyk4a.cpp'],
                   include_dirs=[get_numpy_include()],
                   libraries=['k4a', 'k4arecord'])

setup(ext_modules=[module])
