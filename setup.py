from distutils.core import setup

from Cython.Build import cythonize
cython_path = 'src/cython_modules'
module_list = ["grid_computer_cy.pyx", "mde_cy.pyx", "dynamic_planner_cy.pyx"]
setup(
    name='Minimum Distance Enforcement',
    ext_modules=cythonize(["%s/%s"%(cython_path,module) for module in module_list]),
)
