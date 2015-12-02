__author__ = 'omar'
import sys
import numpy as np
import math
sys.path.insert(1, '..')
import functions as ft
cimport numpy as np

float_type = np.float64

def compute_density_and_velocity_field(cell_dim, size_array,
                                       np.ndarray[np.float64_t, ndim=2] position_array,
                                       np.ndarray[np.float64_t, ndim=2] velocity_array,
                                       np.ndarray[np.float64_t, ndim=1] active_entries, eps = 0.01):
    """
    Compute the density and velocity field in cell centers as done in Treuille et al. (2004).
    Density and velocity are splatted to only surrounding cell centers.
    This implementation loops over all cells.  Suggestions for improvement are welcome.
    :return: (density, velocity_x, velocity_y) as 2D arrays
    """
    cdef Py_ssize_t cell_dim_x = cell_dim[0]
    cdef Py_ssize_t cell_dim_y = cell_dim[1]
    cdef Py_ssize_t cell_x, cell_y

    cdef np.ndarray[np.float64_t, ndim=2] v_x = np.zeros([cell_dim_x, cell_dim_y])
    cdef np.ndarray[np.float64_t, ndim=2] v_y = np.zeros([cell_dim_x, cell_dim_y])
    cdef np.ndarray[np.float64_t, ndim=2] density_field = np.zeros([cell_dim_x, cell_dim_y]) + eps
    cdef np.ndarray[np.float64_t, ndim=1] cell_size = size_array / [cell_dim_x, cell_dim_y]
    cdef float cutoff = math.sqrt(cell_size[0] ** 2 + cell_size[1] ** 2)
    cdef float smoothing_length = cutoff/2
    # cell_locations = np.floor(position_array / cell_size)

    cdef np.ndarray[np.float64_t, ndim=2] differences
    cdef np.ndarray[np.float64_t, ndim=1] distances, close_distances, weights
    #Todo: Move active entries to here, saves us x400 times or such

    for (cell_x, cell_y) in np.ndindex((cell_dim_x, cell_dim_y)):
        differences = position_array - cell_size * (cell_x + 0.5, cell_y + 0.5)
        distances = np.linalg.norm(differences, axis=1)
        close_indices = np.where(np.logical_and(distances < cutoff, active_entries))[0]
        if len(close_indices):
            close_distances = distances[close_indices]
            weights = weight_function(close_distances,smoothing_length)
            total_weight = np.sum(weights)
            density_field[cell_x, cell_y] += total_weight
            v_x[cell_x, cell_y] = np.sum(velocity_array[:, 0][close_indices] * weights)
            v_y[cell_x, cell_y] = np.sum(velocity_array[:, 1][close_indices] * weights)
    return (density_field - eps), v_x / density_field, v_y / density_field

def weight_function(np.ndarray[np.float64_t, ndim=1] array, float smoothing_length=1.):
    """
    Using the Wendland kernel to determine the interpolation weight
    Calculation is performed in two steps to take advantage of numpy's speed
    :param array: Array of distances to apply the kernel on.
    :param smoothing_length: Steepness factor (standard deviation) of kernel
    :return: Weights of interpolation
    """
    array /= smoothing_length
    return 7. / (4 * np.pi * smoothing_length * smoothing_length) * np.maximum(1 - array / 2, 0) ** 4 * (1 + 2 * array)

def solve_LCP_with_pgs(np.ndarray[np.float64_t, ndim=2] M, np.ndarray[np.float64_t, ndim=1] q_vec, init_guess = None):
    """
    Solves the linear complementarity problem w = Mz + q using a Projected Gauss Seidel solver.
    Possible improvements:
        -Sparse matrix use
    :param M: nxn non-singular positive definite matrix
    :param q: length n vector
    :return: length n vector z such that z>=0, w>=0, (w,z)\approx 0 if optimum is found, else zeros vector.
    """
    cdef float eps = 1e-02
    cdef int max_it = 10000
    n = len(q_vec)
    cdef np.ndarray[np.float64_t, ndim=2] q = q_vec[:, None]
    O = np.zeros([n, 1])
    cdef np.ndarray[np.float64_t, ndim=2] z
    if init_guess is not None:
        z = init_guess
    else:
        z = np.ones([n, 1])
    cdef np.ndarray[np.float64_t, ndim=2] w = np.dot(M, z) + q
    cdef int it = 0
    cdef Py_ssize_t i
    cdef float r
    while (np.any(w < -eps) or np.abs(np.dot(w.T, z)) > eps or np.any(z < -eps)) and it < max_it:
        it += 1
        for i in range(n):
            r = -q[i] - np.dot(M[i, :], z) + M[i, i] * z[i]
            z[i] = max(0, r / M[i, i])
        w = np.dot(M, z) + q
    ft.debug("Iterations: %d" % it)
    if it == max_it:
        ft.warn("Max iterations reached, no optimal result found")
        return O
    else:
        return z
