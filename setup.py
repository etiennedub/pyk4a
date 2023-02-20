from setuptools import setup, Extension
from distutils.command.build_ext import build_ext

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


k4a_module = Extension('k4a_module',
                       sources=['pyk4a/pyk4a.cpp'],
                       include_dirs=[get_numpy_include()],
                       libraries=['k4a', 'k4arecord'])


class pyk4a_build_ext(build_ext):
    user_options = build_ext.user_options + [('enable-body-tracking', None, 'Compile with body-tracking support'), ]
    boolean_options = build_ext.boolean_options + ['enable-body-tracking', ]

    def initialize_options(self):
        self.enable_body_tracking = True
        build_ext.initialize_options(self)

    def finalize_options(self):
        build_ext.finalize_options(self)

    def build_extensions(self):
        # modify k4a_module extension depending on arguments like "--enable-body-tracking"
        assert k4a_module in self.extensions
        if self.enable_body_tracking:
            k4a_module.libraries.append('k4abt')
            k4a_module.define_macros.append(('ENABLE_BODY_TRACKING', '1'))

        build_ext.build_extensions(self)


setup(name='pyk4a',
      version='1.1.0',
      description='Python wrapper over Azure Kinect SDK',
      license='GPL-3.0',
      author='Etienne Dubeau',
      install_requires=['numpy'],
      python_requires='>=3.4',
      author_email='etienne.dubeau.1@ulaval.ca',
      url='https://github.com/etiennedub/pyk4a/',
      download_url='https://github.com/etiennedub/pyk4a/archive/1.0.1.tar.gz',
      packages=['pyk4a'],
      package_data={'pyk4a': ["py.typed"]},
      ext_modules=[k4a_module],
      cmdclass={'build_ext': pyk4a_build_ext},
      )
