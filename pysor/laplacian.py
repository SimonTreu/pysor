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
from numpy.testing import assert_array_equal

def laplacian_1d(n):
    r"""The 1D Laplace operator independent of the grid spacing.
    
    Parameters
    ----------
    n : int
        The number of grid points along the discretized axis.
    
    Returns
    -------
    numpy.ndarray(shape=(n, n), dtype=numpy.float64)
        The Laplace operator matrix.
    
    """
    laplacian = np.zeros(shape=(n, n), dtype=np.float64)
    for i in range(n):
        laplacian[i, i] = -2.0
        laplacian[i, (i + 1) % n] += 1.0
        laplacian[i, (i - 1) % n] += 1.0
    return laplacian

def laplacian_2d(n):
    r"""The 2D Laplace operator independent of the grid spacing.
    
    Parameters
    ----------
    n : int
        The number of grid points along each axis.
    
    Returns
    -------
    numpy.ndarray(shape=(n^2, n^2), dtype=numpy.float64)
        The Laplace operator matrix.
    
    """
    laplacian = np.zeros(shape=(n, n, n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            laplacian[i, j, i, j] = -4.0
            laplacian[i, j, (i + 1) % n, j] += 1.0
            laplacian[i, j, (i - 1) % n, j] += 1.0
            laplacian[i, j, i, (j + 1) % n] += 1.0
            laplacian[i, j, i, (j - 1) % n] += 1.0
    return laplacian.reshape((n * n, -1))

def laplacian_3d(n):
    r"""The 3D Laplace operator independent of the grid spacing.
    
    Parameters
    ----------
    n : int
        The number of grid points along each axis.
    
    Returns
    -------
    numpy.ndarray(shape=(n^3, n^3), dtype=numpy.float64)
        The Laplace operator matrix.
    
    """
    laplacian = np.zeros(shape=(n, n, n, n, n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                laplacian[i, j, k, i, j, k] = -6.0
                laplacian[i, j, k, (i + 1) % n, j, k] += 1.0
                laplacian[i, j, k, (i - 1) % n, j, k] += 1.0
                laplacian[i, j, k, i, (j + 1) % n, k] += 1.0
                laplacian[i, j, k, i, (j - 1) % n, k] += 1.0
                laplacian[i, j, k, i, j, (k + 1) % n] += 1.0
                laplacian[i, j, k, i, j, (k - 1) % n] += 1.0
    return laplacian.reshape((n * n * n, -1))
