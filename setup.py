from distutils.core import setup
import os
from numpy.distutils.core import Extension
from numpy.distutils.core import setup as npsetup
from Cython.Build import cythonize

mde = Extension(name='mde', sources=['src/fortran/mde.f90'])
micro_macro = Extension(name='micro_macro', sources=['src/fortran/micro_macro.f90'])
potential_computer = Extension(name='micro_macro', sources=['src/fortran/compute_potential_field.f90'])
pressure_computer = Extension(name='pressure_computer', sources=['src/fortran/pressure_modules.f90',
                                                                 'src/fortran/sparse_modules.f90'])

if __name__ == "__main__":
    if not os.path.exists('images'):
        os.makedirs('images')
    npsetup(name='FORTRAN modules',
            description="FORTRAN modules for Mercurial",
            author="Omar Richardson",
            ext_modules=[mde, micro_macro,potential_computer, pressure_computer])