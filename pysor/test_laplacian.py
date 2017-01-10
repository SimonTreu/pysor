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
from .laplacian import laplacian_1d
from .laplacian import laplacian_2d
from .laplacian import laplacian_3d

def test_laplacian_1d_2():
    assert_array_equal(
        laplacian_1d(2), 
        np.asarray([
            [-2, 2],
            [2, -2]]))

def test_laplacian_1d_3():
    assert_array_equal(
        laplacian_1d(3),
        np.asarray([
            [-2, 1, 1],
            [1, -2, 1],
            [1, 1, -2]]))

def test_laplacian_1d_4():
    assert_array_equal(
        laplacian_1d(4),
        np.asarray([
            [-2, 1, 0, 1],
            [1, -2, 1, 0],
            [0, 1, -2, 1],
            [1, 0, 1, -2]]))

def test_laplacian_2d_2():
    assert_array_equal(
        laplacian_2d(2),
        np.asarray([
            [-4, 2, 2, 0],
            [2, -4, 0, 2],
            [2, 0, -4, 2],
            [0, 2, 2, -4]]))

def test_laplacian_3d_2():
    assert_array_equal(
        laplacian_3d(2),
        np.asarray([
            [-6, 2, 2, 0, 2, 0, 0, 0],
            [2, -6, 0, 2, 0, 2, 0, 0],
            [2, 0, -6, 2, 0, 0, 2, 0],
            [0, 2, 2, -6, 0, 0, 0, 2],
            [2, 0, 0, 0, -6, 2, 2, 0],
            [0, 2, 0, 0, 2, -6, 0, 2],
            [0, 0, 2, 0, 2, 0, -6, 2],
            [0, 0, 0, 2, 0, 2, 2, -6]]))
