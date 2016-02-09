# About the fortran modules #

This directory contains FORTRAN 90 code used in *Mercurial*. 
Compiling happens using the command `python3 setup.py install`.
Alternatively, for debugging or extending the modules, any FORTRAN compiler should work.
However, these modules have been tested using `f2py` (included in `numpy`) and `gfortran`).

## Syntax ##

The code in these modules follows the free-form syntax. All indexing is zero-based, unless indicated otherwise.

## Floating point precision ##

Precision is indicated with the `(kind=8)` option, ensuring double precision (on my machine). 
Portability could be improved (see [Fortran standards][fs]), and might be in the future. However, I guess the most used compilers correctly deduce the precision.


[fs]: (http://www.fortran90.org/src/best-practices.html#floating-point-numbers)
