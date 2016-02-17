from distutils.core import setup

from numpy.distutils.core import Extension
from numpy.distutils.core import setup as npsetup
from Cython.Build import cythonize

mde = Extension(name='mde', sources=['src/fortran/mde.f90'])
micro_macro = Extension(name='micro_macro', sources=['src/fortran/micro_macro.f90'])
pgs_solver = Extension(name='pgs_solver', sources=['src/fortran/projected_gs.f90'])

if __name__ == "__main__":
    npsetup(name='FORTRAN modules',
            description="FORTRAN modules for Mercurial",
            author="Omar Richardson",
            ext_modules=[mde, micro_macro, pgs_solver])

    # Cython will be removed in the next version
    cython_path = 'src/cython_modules'
    module_list = ["grid_computer_cy.pyx", "mde_cy.pyx", "dynamic_planner_cy.pyx"]
    setup(name='Cython modules',
          description="Cython modules for Mercurial (deprecated)",
          author="Omar Richardson",
          ext_modules=cythonize(["%s/%s" % (cython_path, module) for module in module_list]))
