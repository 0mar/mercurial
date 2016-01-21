__author__ = 'omar'
import sys
import numpy as np
sys.path.insert(1, '..')
import functions as ft
cimport numpy as np
import math
float_type = np.float64

def compute_density_and_velocity_field(cell_dim, size_array,
                                       np.ndarray[np.float64_t, ndim=2] position_array,
                                       np.ndarray[np.float64_t, ndim=2] velocity_array,
                                       pedestrian_list, eps = 0.01):
    """
    Compute the density and velocity field in cell centers as done in Treuille et al. (2004).
    Density and velocity are splatted to only surrounding cell centers.
    This is a semi-naive implementation, looping over all pedestrians
    :return: (density, velocity_x, velocity_y) as 2D arrays
    """
    cdef np.ndarray[np.float64_t, ndim=2]density_field = np.zeros(cell_dim) + eps
    # Initialize density with an epsilon to facilitate division
    cdef np.ndarray[np.float64_t, ndim=2] v_x = np.zeros(cell_dim)
    cdef np.ndarray[np.float64_t, ndim=2] v_y = np.zeros(cell_dim)
    cell_center_indices = (np.around(position_array / size_array) - np.array([1, 1])).astype(int)
    # These are the closest cell center indices whose coordinates both less than the corresponding pedestrians.
    cell_centers = (cell_center_indices + [0.5, 0.5]) * size_array
    # These are the coordinates of those centers
    differences = position_array - cell_centers
    # These are the differences between the cell centers and the pedestrian positions.
    # They should all be positive and smaller than (dx,dy)
    rel_differences = differences / size_array
    density_contributions = [[None, None], [None, None]]
    cdef Py_ssize_t x_coord, y_coord, x, y, ped_index, list_index
    cdef int max_cell_x = cell_dim[0]
    cdef int max_cell_y = cell_dim[1]
    cdef float vel_x, vel_y, dens_cont
    for x in range(2):
        for y in range(2):
            density_contributions[x][y] = np.minimum(
                1 - rel_differences[:, 0] + (2 * rel_differences[:, 0] - 1) * x,
                1 - rel_differences[:, 1] + (2 * rel_differences[:, 1] - 1) * y) ** 2  # density exponent
    # For each cell center surrounding the pedestrian, add (in both dimensions) 1-\delta . or \delta .
    for list_index, ped in enumerate(pedestrian_list):
        ped_index = ped.index
        vel_x = velocity_array[ped_index, 0]
        vel_y = velocity_array[ped_index, 0]
        dens_cont = density_contributions[x][y][ped_index]
        for x in range(2):
            x_coord = cell_center_indices[ped_index, 0] + x
            if 0 <= x_coord < max_cell_x:
                for y in range(2):
                    y_coord = cell_center_indices[ped_index, 1] + y
                    if 0 <= y_coord < max_cell_y:
                        density_field[x_coord, y_coord] += dens_cont
                        v_x[x_coord, y_coord] += dens_cont * vel_x
                        v_y[x_coord, y_coord] += dens_cont * vel_y
    # For each pedestrian, and for each surrounding cell center, add the corresponding density distribution.
    return density_field, v_x / density_field, v_y / density_field

def compute_potential_cy(cell, np.ndarray[np.float64_t, ndim=2] potential_field, unit_field_dict, opposites):
    """
    Computes the potential in one cell, using potential in neighbouring cells.
    """
    # Find the minimal directions along a grid cell.
    # Assume left and below are best, then overwrite with right and up if they are better
    cdef float inf = np.inf
    neighbour_pots = {direction: inf for direction in ft.DIRECTIONS}

    cdef float hor_potential, ver_potential, hor_cost, ver_cost, cost, pot
    hor_cost = inf
    ver_cost = inf

    cdef Py_ssize_t cell_x, cell_y, nb_cell_x, nb_cell_y, normal_x, normal_y, face_index_x, face_index_y
    cell_x = cell[0]
    cell_y = cell[1]
    for direction in ft.DIRECTIONS:
        normal_x = ft.DIRECTIONS[direction][0]
        normal_y = ft.DIRECTIONS[direction][1]
        # numerical direction
        nb_cell_x = cell_x + normal_x
        nb_cell_y = cell_y + normal_y
        if not exists(nb_cell_x, nb_cell_y, potential_field.shape[0], potential_field.shape[1]):
            continue
        pot = potential_field[nb_cell_x, nb_cell_y]
        # potential in that neighbour field
        if direction == 'right':
            face_index_x = nb_cell_x - 1
            face_index_y = nb_cell_y
        elif direction == 'up':
            face_index_x = nb_cell_x
            face_index_y = nb_cell_y - 1
            # Unit cost values are defined w.r.t faces, not cells! So the indexing is different with right and up.
        else:
            face_index_x = nb_cell_x
            face_index_y = nb_cell_y
        cost = unit_field_dict[opposites[direction]].array[face_index_x, face_index_y]
        # Cost to go from there to here
        neighbour_pots[direction] = pot + cost
        # total potential
        if neighbour_pots[direction] < neighbour_pots[opposites[direction]]:
            if direction in ft.HORIZONTAL_DIRECTIONS:
                hor_potential = pot
                hor_cost = cost
                # lowest in horizontal direction
            else:
                ver_potential = pot
                ver_cost = cost
                # lowest in vertical direction
    # Coefficients of quadratic equation
    cdef float a, b, c, D, x_high
    if hor_cost == inf:
        a = 1. / (ver_cost * ver_cost)
        b = -2. * ver_potential / (ver_cost * ver_cost)
        c = (ver_potential / ver_cost) * (ver_potential / ver_cost) - 1
    elif ver_cost == inf:
        a = 1. / (hor_cost * hor_cost)
        b = -2. * hor_potential / (hor_cost * hor_cost)
        c = (hor_potential / hor_cost) * (hor_potential / hor_cost) - 1
    else:
        a = 1. / (hor_cost * hor_cost) + 1. / (ver_cost * ver_cost)
        b = -2. * (hor_potential / (hor_cost * hor_cost) + ver_potential / (ver_cost * ver_cost))
        c = (hor_potential / hor_cost) * (hor_potential / hor_cost) + (ver_potential / ver_cost) * (
    ver_potential / ver_cost) - 1

    D = b * b - 4. * a * c
    if D < 0:
        ft.warn("Negative D: %.2e" % D)
        D = 0
    # x_high = (2.*c)/(-b - math.sqrt(D))
    x_high = (-b + math.sqrt(D)) / (2. * a)
    # Might not be obvious, but why we take the largest root is found in report.
    return x_high

cdef inline int exists(int cell_x, int cell_y, int cell_max_x, int cell_max_y):
    if cell_x < 0 or cell_y < 0 or cell_x >= cell_max_x or cell_y >= cell_max_y:
        return 0
    else:
        return 1

def weight_function(np.ndarray array, float smoothing_length=1.):
    """
    Using the Wendland kernel to determine the interpolation weight
    Calculation is performed in two steps to take advantage of numpy's speed
    :param array: Array of distances to apply the kernel on.
    :param smoothing_length: Steepness factor (standard deviation) of kernel
    :return: Weights of interpolation
    """
    array /= smoothing_length
    return 7. / (4 * np.pi * smoothing_length * smoothing_length) * np.maximum(1 - array / 2, 0) ** 4 * (1 + 2 * array)
