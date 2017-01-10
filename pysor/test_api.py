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

def test_sor_callable():
    assert False
