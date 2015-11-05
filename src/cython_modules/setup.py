from distutils.core import setup

from Cython.Build import cythonize

setup(
    name='Minimum Distance Enforcement',
    ext_modules=cythonize(["mde.pyx", "dynamic_planner.pyx"]),
)
