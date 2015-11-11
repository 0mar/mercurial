__author__ = 'omar'
import sys
import numpy as np

sys.path.insert(1, '..')
import functions as ft
cimport numpy as np

float_type = np.float64

def compute_density_and_velocity_field(size_array, np.ndarray[np.float64_t, ndim=2] position_array,
                                       np.ndarray[np.float64_t, ndim=2] velocity_array, eps = 0.01):
    """
    Compute the density and velocity field in cell centers as done in Treuille et al. (2004).
    Density and velocity are splatted to only surrounding cell centers.
    This is a naive implementation, looping over all pedestrians
    :return: (density, velocity_x, velocity_y) as 2D arrays
    """
    cdef Py_ssize_t cell_dim_x = 20
    cdef Py_ssize_t cell_dim_y = 20  # Semifixed
    cdef Py_ssize_t cell_x, cell_y

    cdef np.ndarray[np.float64_t, ndim=2] v_x = np.zeros([cell_dim_x, cell_dim_y])
    cdef np.ndarray[np.float64_t, ndim=2] v_y = np.zeros([cell_dim_x, cell_dim_y])
    cdef np.ndarray[np.float64_t, ndim=2] density_field = np.zeros([cell_dim_x, cell_dim_y]) + eps
    cdef np.ndarray[np.float64_t, ndim=1] cell_size = size_array / [cell_dim_x, cell_dim_y]
    cdef float cutoff = ft.norm(cell_size[0], cell_size[1])
    cdef float smoothing_length = 1
    # cell_locations = np.floor(position_array / cell_size)

    cdef np.ndarray[np.float64_t, ndim=2] differences
    cdef np.ndarray[np.float64_t, ndim=1] distances, close_distances, weights

    for (cell_x, cell_y) in np.ndindex((cell_dim_x, cell_dim_y)):
        differences = position_array - cell_size * (cell_x + 0.5, cell_y + 0.5)
        distances = np.linalg.norm(differences, axis=1)
        close_indices = np.where(distances < cutoff)[0]
        if len(close_indices):
            close_distances = distances[close_indices]
            weights = weight_function(close_distances / cutoff)
            total_weight = np.sum(weights)
            density_field[cell_x, cell_y] = total_weight
            v_x[cell_x, cell_y] = np.sum(velocity_array[close_indices] * weights[:, None])
            v_y[cell_x, cell_y] = np.sum(velocity_array[close_indices] * weights[:, None])
    return (density_field - eps), v_x / density_field, v_y / density_field

cdef inline np.ndarray[np.float64_t, ndim=1] weight_function(np.ndarray[np.float64_t, ndim=1] array,
                                                             float smoothing_length=1.):
    """
    Using the Wendland kernel to determine the interpolation weight
    Calculation is performed in two steps to take advantage of numpy's speed
    :param array: Array of distances to apply the kernel on.
    :param smoothing_length: Steepness factor (standard deviation) of kernel
    :return: Weights of interpolation
    """
    array /= smoothing_length
    return 7. / (4 * np.pi * smoothing_length * smoothing_length) * np.maximum(1 - array / 2, 0) ** 4 ** (1 + 2 * array)
