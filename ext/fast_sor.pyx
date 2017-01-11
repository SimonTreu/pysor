#   PySOR - solve Poisson's equation with successive over-relaxation.
#   Copyright (C) 2017  Christoph Wehmeyer
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
cimport numpy as np

cdef extern from "src_fast_sor.h":
    double _sor_step_1d(double *phi, double *rho, int n, double w, double he)
    double _sor_step_2d(double *phi, double *rho, int n, double w, double he)
    double _sor_step_3d(double *phi, double *rho, int n, double w, double he)

def sor_1d(
    np.ndarray[double, ndim=1, mode='c'] phi not None,
    np.ndarray[double, ndim=1, mode='c'] rho not None,
    double w, double he, int maxiter, double maxerr):
    cdef:
        int i
        double error
    for i in range(maxiter):
        error = _sor_step_1d(
            <double*> np.PyArray_DATA(phi),
            <double*> np.PyArray_DATA(rho),
            phi.shape[0], w, he)
        if error < maxerr:
            break
    return phi

def sor_2d(
    np.ndarray[double, ndim=2, mode='c'] phi not None,
    np.ndarray[double, ndim=2, mode='c'] rho not None,
    double w, double he, int maxiter, double maxerr):
    cdef:
        int i
        double error
    for i in range(maxiter):
        error = _sor_step_2d(
            <double*> np.PyArray_DATA(phi),
            <double*> np.PyArray_DATA(rho),
            phi.shape[0], w, he)
        if error < maxerr:
            break
    return phi

def sor_3d(
    np.ndarray[double, ndim=3, mode='c'] phi not None,
    np.ndarray[double, ndim=3, mode='c'] rho not None,
    double w, double he, int maxiter, double maxerr):
    cdef:
        int i
        double error
    for i in range(maxiter):
        error = _sor_step_3d(
            <double*> np.PyArray_DATA(phi),
            <double*> np.PyArray_DATA(rho),
            phi.shape[0], w, he)
        if error < maxerr:
            break
    return phi
