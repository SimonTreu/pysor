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
from .api import sor
from .api import laplacian
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

def test_laplacian_1d():
    from .laplacian import laplacian_1d
    n = np.random.randint(2, 10)
    assert_array_equal(
        laplacian(n, 1),
        laplacian_1d(n))

def test_laplacian_2d():
    from .laplacian import laplacian_2d
    n = np.random.randint(2, 10)
    assert_array_equal(
        laplacian(n, 2),
        laplacian_2d(n))

def test_laplacian_3d():
    from .laplacian import laplacian_3d
    n = np.random.randint(2, 10)
    assert_array_equal(
        laplacian(n, 3),
        laplacian_3d(n))

#   These tests compute phi vor SOR and compute rho back via a matrix vector
#   product with the corresponding laplacian. If the computed rho is close
#   enough to the original rho, the test ist passed.

def check_poisson_consistency(rho, h, maxiter, maxerr):
    assert_array_almost_equal(
        np.dot(
            laplacian(rho.shape[0], rho.ndim),
            sor(rho, h, maxiter=maxiter, maxerr=maxerr, fast=False).reshape((-1,))).reshape(rho.shape),
        h**rho.ndim * (-rho),
        decimal=4)

def test_sor_1d_random_naive():
    n = np.random.randint(100, 200)
    g = np.linspace(0, 1, n, endpoint=False)
    rho = np.exp((-1000.0) * (g - 0.3)**2) - np.exp((-1000.0) * (g - 0.7)**2)
    rho -= rho.mean()
    check_poisson_consistency(rho, g[1] - g[0], 100000, 1.0E-10)

def test_sor_2d_random_naive():
    n = np.random.randint(30, 50)
    g = np.linspace(0, 1, n, endpoint=False)
    x, y = np.meshgrid(g, g)
    rho = np.exp((-1000.0) * ((x - 0.3)**2 + (y - 0.3)**2)) -\
        np.exp((-1000.0) * ((x - 0.7)**2 + (y - 0.7)**2))
    rho -= rho.mean()
    check_poisson_consistency(rho, g[1] - g[0], 100000, 1.0E-10)

def test_sor_3d_random_naive():
    n = np.random.randint(10, 15)
    g = np.linspace(0, 1, n, endpoint=False)
    x, y, z = np.meshgrid(g, g, g)
    rho = np.exp((-1000.0) * ((x - 0.3)**2 + (y - 0.3)**2 + (z - 0.3)**2)) \
        - np.exp((-1000.0) * ((x - 0.7)**2 + (y - 0.7)**2 + (z - 0.7)**2))
    rho -= rho.mean()
    check_poisson_consistency(rho, g[1] - g[0], 100000, 1.0E-10)

#   These tests compare the fast version with the tested naive implementation

def test_sor_1d_random_fast():
    n = np.random.randint(30, 50)
    g = np.linspace(0, 1, n, endpoint=False)
    rho = np.exp((-1000.0) * (g - 0.3)**2) - np.exp((-1000.0) * (g - 0.7)**2)
    rho -= rho.mean()
    assert_array_almost_equal(
        sor(rho, g[1] - g[0], maxiter=100000, maxerr=1.0E-10, fast=True),
        sor(rho, g[1] - g[0], maxiter=100000, maxerr=1.0E-10, fast=False),
        decimal=12)

def test_sor_2d_random_fast():
    n = np.random.randint(50, 100)
    g = np.linspace(0, 1, n, endpoint=False)
    x, y = np.meshgrid(g, g)
    rho = np.exp((-1000.0) * ((x - 0.3)**2 + (y - 0.3)**2)) -\
        np.exp((-1000.0) * ((x - 0.7)**2 + (y - 0.7)**2))
    rho -= rho.mean()
    assert_array_almost_equal(
        sor(rho, g[1] - g[0], maxiter=100000, maxerr=1.0E-10, fast=True),
        sor(rho, g[1] - g[0], maxiter=100000, maxerr=1.0E-10, fast=False),
        decimal=12)

def test_sor_3d_random_fast():
    n = np.random.randint(10, 15)
    g = np.linspace(0, 1, n, endpoint=False)
    x, y, z = np.meshgrid(g, g, g)
    rho = np.exp((-1000.0) * ((x - 0.3)**2 + (y - 0.3)**2 + (z - 0.3)**2)) \
        - np.exp((-1000.0) * ((x - 0.7)**2 + (y - 0.7)**2 + (z - 0.7)**2))
    rho -= rho.mean()
    assert_array_almost_equal(
        sor(rho, g[1] - g[0], maxiter=100000, maxerr=1.0E-10, fast=True),
        sor(rho, g[1] - g[0], maxiter=100000, maxerr=1.0E-10, fast=False),
        decimal=12)
