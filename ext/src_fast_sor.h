/*  PySOR - solve Poisson's equation with successive over-relaxation.
*   Copyright (C) 2017  Christoph Wehmeyer
*
*   This program is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*   the Free Software Foundation, either version 3 of the License, or
*   (at your option) any later version.
*
*   This program is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License
*   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef PYSOR
#define PYSOR

double _sor_step_1d(double *phi, double *rho, int n, double w, double he);
double _sor_step_2d(double *phi, double *rho, int n, double w, double he);
double _sor_step_3d(double *phi, double *rho, int n, double w, double he);

#endif
