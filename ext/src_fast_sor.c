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

inline int even(int value) { return (value % 2 == 0) ? 1 : 0; }

inline int pbc(int i, int n) {
    while(i >= n) i -= n;
    while(i < 0) i += n;
    return i;
}

inline double sqr(double value) { return (value == 0.0) ? 0.0 : value * value; }

double _sor_step_1d(double *phi, double *rho, int n, double w, double he) {
    int i;
    double phi_i, error = 0.0;
    for(i=0; i<n; ++i) {
        if(even(i) == 1) continue;
        phi_i = 0.5 * (phi[pbc(i - 1, n)] + phi[pbc(i + 1, n)] + rho[i] * he);
        phi[i] = (1.0 - w) * phi[i] + w * phi_i;
        error += sqr(phi[i] - phi_i);
    }
    for(i=0; i<n; ++i) {
        if(even(i) == 0) continue;
        phi_i = 0.5 * (phi[pbc(i - 1, n)] + phi[pbc(i + 1, n)] + rho[i] * he);
        phi[i] = (1.0 - w) * phi[i] + w * phi_i;
        error += sqr(phi[i] - phi_i);
    }
    return error;
}

inline int map2d(int i, int j, int n) { return i * n + j; }

double _sor_step_2d(double *phi, double *rho, int n, double w, double he) {
    int i, j;
    double phi_ij, error = 0.0;
    for(i=0; i<n; ++i) {
        for(j=0; j<n; ++j) {
            if(even(i + j) == 1) continue;
            phi_ij = 0.25 * (
                phi[map2d(pbc(i - 1, n), j, n)] +
                phi[map2d(pbc(i + 1, n), j, n)] +
                phi[map2d(i, pbc(j - 1, n), n)] +
                phi[map2d(i, pbc(j + 1, n), n)] +
                rho[map2d(i, j, n)] * he);
            phi[map2d(i, j, n)] = (1.0 - w) * phi[map2d(i, j, n)] + w * phi_ij;
            error += sqr(phi[map2d(i, j, n)] - phi_ij);          
        }
    }
    for(i=0; i<n; ++i) {
        for(j=0; j<n; ++j) {
            if(even(i + j) == 0) continue;
            phi_ij = 0.25 * (
                phi[map2d(pbc(i - 1, n), j, n)] +
                phi[map2d(pbc(i + 1, n), j, n)] +
                phi[map2d(i, pbc(j - 1, n), n)] +
                phi[map2d(i, pbc(j + 1, n), n)] +
                rho[map2d(i, j, n)] * he);
            phi[map2d(i, j, n)] = (1.0 - w) * phi[map2d(i, j, n)] + w * phi_ij;
            error += sqr(phi[map2d(i, j, n)] - phi_ij);          
        }
    }
    return error;
}

inline int map3d(int i, int j, int k, int n) { return n * map2d(i, j, n) + k; } 

double _sor_step_3d(double *phi, double *rho, int n, double w, double he) {
    int i, j, k;
    double phi_ijk, error = 0.0;
    for(i=0; i<n; ++i) {
        for(j=0; j<n; ++j) {
            for(k=0; k<n; ++k) {
                if(even(i + j + k) == 1) continue;
                phi_ijk = 0.166666666666666657 * (
                    phi[map3d(pbc(i - 1, n), j, k, n)] +
                    phi[map3d(pbc(i + 1, n), j, k, n)] +
                    phi[map3d(i, pbc(j - 1, n), k, n)] +
                    phi[map3d(i, pbc(j + 1, n), k, n)] +
                    phi[map3d(i, j, pbc(k - 1, n), n)] +
                    phi[map3d(i, j, pbc(k + 1, n), n)] +
                    rho[map3d(i, j, k, n)] * he);
                phi[map3d(i, j, k, n)] = (1.0 - w) * phi[map3d(i, j, k, n)] + w * phi_ijk;
                error += sqr(phi[map3d(i, j, k, n)] - phi_ijk);                 
            }
        }
    }
    for(i=0; i<n; ++i) {
        for(j=0; j<n; ++j) {
            for(k=0; k<n; ++k) {
                if(even(i + j + k) == 0) continue;
                phi_ijk = 0.166666666666666657 * (
                    phi[map3d(pbc(i - 1, n), j, k, n)] +
                    phi[map3d(pbc(i + 1, n), j, k, n)] +
                    phi[map3d(i, pbc(j - 1, n), k, n)] +
                    phi[map3d(i, pbc(j + 1, n), k, n)] +
                    phi[map3d(i, j, pbc(k - 1, n), n)] +
                    phi[map3d(i, j, pbc(k + 1, n), n)] +
                    rho[map3d(i, j, k, n)] * he);
                phi[map3d(i, j, k, n)] = (1.0 - w) * phi[map3d(i, j, k, n)] + w * phi_ijk;
                error += sqr(phi[map3d(i, j, k, n)] - phi_ijk);                 
            }
        }
    }
    return error;
}
