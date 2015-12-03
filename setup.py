from distutils.core import setup

from Cython.Build import cythonize
cython_path = 'src/cython_modules'
module_list = ["grid_computer.pyx", "mde.pyx", "dynamic_planner.pyx"]
setup(
    name='Minimum Distance Enforcement',
    ext_modules=cythonize(["%s/%s"%(cython_path,module) for module in module_list]),
)
