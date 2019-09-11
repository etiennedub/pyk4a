from distutils.core import Extension
from setuptools import setup
import numpy

module = Extension('k4a_module',
                    sources=['pyk4a/pyk4a.cpp'],
                    include_dirs=[numpy.get_include()],
                    library_dirs=['/usr/local/lib64/'],
                    libraries=['k4a'])

setup(name='pyk4a',
      version='0.1',
      description='Python wrapper over Azure Kinect SDK',
      license='GPL-3.0',
      author='Etienne Dubeau',
      author_email='etienne.dubeau.1@ulaval.ca',
      packages=['pyk4a'],
      ext_modules=[module])
