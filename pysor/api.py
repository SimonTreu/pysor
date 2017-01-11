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
import fast_sor as fs
import naive_sor as ns
import laplacian as lp

def sor(rho, h, epsilon=1.0, maxiter=1000, maxerr=1.0E-7, w=None, fast=True):
    r"""Solve the dim-D Poisson equation using the successive overrelaxation method.

    Parameters
    ----------
    rho : numpy.ndarray() or arraylike of float
        The charge density grid; allowed shapes are (n,), (n, n), and (n, n, n).
    h : float
        The grid spacing along each axis.
    epsilon : float, optional, default=1.0
        The vacuum permittivity.
    maxerr : float, optional, default=1.eE-8
        The convergence criterion.
    maxiter : int, optional, default=1000
        The number of iterations.
    w : float, optional, default=None
        Overwrite the automatically computed SOR parameter.
    fast : boolean, optional, default=True
        Use a fast version of the SOR code instead of a slow but simple
        reference implementation.

    Returns
    -------
    numpy.ndarray(shape=rho.shape, dtype=rho.dtype)
        The potential grid.

    """
    rho = np.asarray(rho, dtype=np.float64)
    dim = rho.ndim
    if fast:
        phi = np.zeros(shape=rho.shape, dtype=rho.dtype)
        if w is None:
            w = 2.0 / (1.0 + np.pi / float(rho.shape[0]))
        if dim == 1:
            return fs.sor_1d(phi, rho, w, h / epsilon, maxiter, maxerr)
        elif dim == 2:
            return fs.sor_2d(phi, rho, w, h * h / epsilon, maxiter, maxerr)
        elif dim == 3:
            return fs.sor_3d(phi, rho, w, h * h * h / epsilon, maxiter, maxerr)
        else:
            raise ValueError("dimensionality must be 1, 2, 3; got %d" % dim)
    else:
        if dim == 1:
            return ns.sor_1d(rho, h, epsilon=epsilon, maxiter=maxiter, maxerr=maxerr, w=w)
        elif dim == 2:
            return ns.sor_2d(rho, h, epsilon=epsilon, maxiter=maxiter, maxerr=maxerr, w=w)
        elif dim == 3:
            return ns.sor_3d(rho, h, epsilon=epsilon, maxiter=maxiter, maxerr=maxerr, w=w)
        else:
            raise ValueError("dimensionality must be 1, 2, 3; got %d" % dim)

def laplacian(n, dim):
    r"""The dim-D Laplace operator independent of the grid spacing.
    
    Parameters
    ----------
    n : int
        The number of grid points along each axis.
    dim : int
        The number of axes; allowed are the values 1, 2, and 3.
    
    Returns
    -------
    numpy.ndarray(shape=(n^dim, n^dim), dtype=numpy.float64)
        The Laplace operator matrix.
    
    """
    if dim == 1:
        return lp.laplacian_1d(n)
    elif dim == 2:
        return lp.laplacian_2d(n)
    elif dim == 3:
        return lp.laplacian_3d(n)
    else:
        raise ValueError("dim must be 1, 2, 3; got %d" % dim)
