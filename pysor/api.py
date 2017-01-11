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
import naive_sor as ns
import laplacian as lp

def sor(rho, h, epsilon=1.0, maxiter=1000, maxerr=1.0E-7, w=None):
    rho = np.asarray(rho, dtype=np.float64)
    dim = rho.ndim
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
