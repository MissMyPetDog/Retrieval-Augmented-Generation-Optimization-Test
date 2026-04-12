"""
Build Cython extensions:
    python setup_cython.py build_ext --inplace
"""
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(
        "optimized/similarity_cython.pyx",
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
    include_dirs=[np.get_include()],
)
