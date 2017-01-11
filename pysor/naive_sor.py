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

def sor_1d(rho, h, epsilon=1.0, maxiter=1000, maxerr=1.0E-7, w=None):
    r"""Solve the 1D Poisson equation using the successive overrelaxation method.

    Parameters
    ----------
    rho : numpy.ndarray(shape=(n,))
        The charge density grid.
    h : float
        The grid spacing.
    epsilon : float, optional, default=1.0
        The vacuum permittivity.
    maxerr : float, optional, default=1.eE-8
        The convergence criterion.
    maxiter : int, optional, default=1000
        The number of iterations.
    w : float, optional, default=None
        Overwrite the automatically computed SOR parameter.

    Returns
    -------
    numpy.ndarray(shape=rho.shape, dtype=rho.dtype)
        The potential grid.

    """
    if rho.ndim != 1:
        raise ValueError("rho must be of shape=(n,)")
    phi = np.zeros(shape=rho.shape, dtype=rho.dtype)
    n = rho.shape[0]
    if w is None:
        w = 2.0 / (1.0 + np.pi / float(n))
    for iteration in range(maxiter):
        error = 0.0
        for x in range(n):
            if x % 2 == 0: continue
            phi_x = (
                phi[(x - 1) % n] + \
                phi[(x + 1) % n] + \
                rho[x] * h / epsilon) / 2.0
            phi[x] = (1.0 - w) * phi[x] + w * phi_x
            error += (phi[x] - phi_x)**2
        for x in range(n):
            if x % 2 != 0: continue
            phi_x = (
                phi[(x - 1) % n] + \
                phi[(x + 1) % n] + \
                rho[x] * h / epsilon) / 2.0
            phi[x] = (1.0 - w) * phi[x] + w * phi_x
            error += (phi[x] - phi_x)**2
        if error < maxerr:
            break
    return phi

def sor_2d(rho, h, epsilon=1.0, maxiter=1000, maxerr=1.0E-7, w=None):
    r"""Solve the 2D Poisson equation using the successive overrelaxation method.

    Parameters
    ----------
    rho : numpy.ndarray(shape=(n, n))
        The charge density grid.
    h : float
        The grid spacing.
    epsilon : float, optional, default=1.0
        The vacuum permittivity.
    maxerr : float, optional, default=1.eE-8
        The convergence criterion.
    maxiter : int, optional, default=1000
        The number of iterations.
    w : float, optional, default=None
        Overwrite the automatically computed SOR parameter.

    Returns
    -------
    numpy.ndarray(shape=rho.shape, dtype=rho.dtype)
        The potential grid.

    """
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be of shape=(n, n)")
    phi = np.zeros(shape=rho.shape, dtype=rho.dtype)
    n = rho.shape[0]
    if w is None:
        w = 2.0 / (1.0 + np.pi / float(n))
    for iteration in range(maxiter):
        error = 0.0
        for x in range(n):
            for y in range(n):
                if (x + y) % 2 == 0: continue
                phi_xy = (
                    phi[(x - 1) % n, y] + \
                    phi[(x + 1) % n, y] + \
                    phi[x, (y - 1) % n] + \
                    phi[x, (y + 1) % n] + \
                    rho[x, y] * h * h / epsilon) / 4.0
                phi[x, y] = (1.0 - w) * phi[x, y] + w * phi_xy
                error += (phi[x, y] - phi_xy)**2
        for x in range(n):
            for y in range(n):
                if (x + y) % 2 != 0: continue
                phi_xy = (
                    phi[(x - 1) % n, y] + \
                    phi[(x + 1) % n, y] + \
                    phi[x, (y - 1) % n] + \
                    phi[x, (y + 1) % n] + \
                    rho[x, y] * h * h / epsilon) / 4.0
                phi[x, y] = (1.0 - w) * phi[x, y] + w * phi_xy
                error += (phi[x, y] - phi_xy)**2
        if error < maxerr:
            break
    return phi

def sor_3d(rho, h, epsilon=1.0, maxiter=1000, maxerr=1.0E-7, w=None):
    r"""Solve the 3D Poisson equation using the successive overrelaxation method.
    
    Parameters
    ----------
    rho : numpy.ndarray(shape=(n, n, n))
        The charge density grid.
    h : float
        The grid spacing.
    maxerr : float, optional, default=1.eE-8
        The convergence criterion.
    maxiter : int, optional, default=1000
        The number of iterations.
    w : float, optional, default=None
        Overwrite the automatically computed SOR parameter.
    
    Returns
    -------
    numpy.ndarray(shape=rh,shape, dtype=rho.dtype)
        The potential grid.
    
    """
    if rho.ndim != 3 or rho.shape[0] != rho.shape[1] != rho.shape[2]:
        raise ValueError("rho must be of shape=(n, n, n)")
    phi = np.zeros(shape=rho.shape, dtype=rho.dtype)
    n = rho.shape[0]
    if w is None:
        w = 2.0 / (1.0 + np.pi / float(n))
    errors = []
    for iteration in range(maxiter):
        error = 0.0
        for x in range(n):
            for y in range(n):
                for z in range(n):
                    if (x + y + z) % 2 == 0: continue
                    phi_xyz = (
                        phi[(x - 1) % n, y, z] + \
                        phi[(x + 1) % n, y, z] + \
                        phi[x, (y - 1) % n, z] + \
                        phi[x, (y + 1) % n, z] + \
                        phi[x, y, (z - 1) % n] + \
                        phi[x, y, (z + 1) % n] + \
                        rho[x, y, z] * h * h * h / epsilon) / 6.0
                    phi[x, y, z] = (1.0 - w) * phi[x, y, z] + w * phi_xyz
                    error += (phi[x, y, z] - phi_xyz)**2
        for x in range(n):
            for y in range(n):
                for z in range(n):
                    if (x + y + z) % 2 != 0: continue
                    phi_xyz = (
                        phi[(x - 1) % n, y, z] + \
                        phi[(x + 1) % n, y, z] + \
                        phi[x, (y - 1) % n, z] + \
                        phi[x, (y + 1) % n, z] + \
                        phi[x, y, (z - 1) % n] + \
                        phi[x, y, (z + 1) % n] + \
                        rho[x, y, z] * h * h * h / epsilon) / 6.0
                    phi[x, y, z] = (1.0 - w) * phi[x, y, z] + w * phi_xyz
                    error += (phi[x, y, z] - phi_xyz)**2
        if error < maxerr:
            break
    return phi
