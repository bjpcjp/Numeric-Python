# usage: $python setup.py build_ext --inplace 
# instructs distutils to build extension module in same directory as source.

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(ext_modules=cythonize('cy_sum.pyx'),
    include_dirs=[np.get_include()],
    requires=['Cython','numpy'])