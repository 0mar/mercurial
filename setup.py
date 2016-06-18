import os
from numpy.distutils.core import Extension
from numpy.distutils.core import setup as npsetup

mde = Extension(name='mde', sources=['src/fortran/mde.f90'])
micro_macro = Extension(name='micro_macro', sources=['src/fortran/micro_macro.f90'])
pressure_computer = Extension(name='pressure_computer', sources=['src/fortran/pressure_modules.f90',
                                                                 'src/fortran/sparse_modules.f90'])

if __name__ == "__main__":
    # Create a folder for images
    if not os.path.exists('images'):
        os.makedirs('images')
    # Build the Fortran modules
    npsetup(name='FORTRAN modules',
            description="FORTRAN modules for Mercurial",
            author="Omar Richardson",
            ext_modules=[mde, micro_macro, pressure_computer])