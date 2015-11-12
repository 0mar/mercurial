__author__ = 'omar'
import sys
import numpy as np
sys.path.insert(1, '..')
cimport numpy as np
import functions as ft

float_type = np.float64

def minimum_distance_enforcement(size_array, np.ndarray[np.float64_t, ndim=2] position_array,
                                 np.ndarray[np.float64_t, ndim=1] active_entries, float min_distance):
    cdef int cell_dim_x = 12
    cdef int cell_dim_y = 12
    cdef Py_ssize_t cell_x, cell_y
    cell_size = size_array / [cell_dim_x, cell_dim_y]
    cdef np.ndarray[np.float64_t, ndim=2] all_corrections = np.zeros((position_array.shape[0], position_array.shape[1]))
    cell_locations = np.floor(position_array / cell_size).astype(int)
    cdef np.ndarray position, replica, differences, distance, corrections, total_corrections
    cdef int n
    for (cell_x, cell_y) in np.ndindex((cell_dim_x, cell_dim_y)):
        same_cells = np.logical_and(cell_locations[:, 0] == cell_x, cell_locations[:, 1] == cell_y)
        index_list = np.where(np.logical_and(same_cells, active_entries))[0]
        if len(index_list) > 1:
            position = position_array[index_list]
            n = len(index_list)
            replica = np.tile(position, (n, 1, 1))
            differences = np.transpose(replica, axes=[1, 0, 2]) - replica
            distance = np.linalg.norm(differences, axis=2)
            distance[distance > min_distance] = np.nan
            corrections = -differences / (distance[:, :, None] + ft.EPS) * (min_distance - distance[:, :, None]) / 2
            corrections[np.isnan(corrections)] = 0
            total_corrections = np.sum(corrections, axis=0)
            all_corrections[index_list] += total_corrections
    return all_corrections
