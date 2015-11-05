__author__ = 'omar'
import sys

sys.path.insert(1, '..')
cimport numpy as np
import itertools
import functions as ft

float_type = np.float64

def minimum_distance_enforcement(cells, np.ndarray[np.float64_t, ndim=2] position_array, min_distance):
    """
    Finds the pedestrian pairs that are closer than the specified distance.
    Does so by comparing the distances of all pedestrians a,b in a cell.
    Note that intercellullar pedestrian pairs are ignored,
    we might fix this later.

    :param min_distance: minimum distance between pedestrians, including their size.
    :return: list of pedestrian index pairs with distances lower than min_distance.
    """
    list_a = []
    list_b = []
    index_list = []
    for cell in cells:
        for ped_combination in itertools.combinations(cell.pedestrian_set, 2):
            list_a.append(position_array[ped_combination[0].index])
            list_b.append(position_array[ped_combination[1].index])
            index_list.append([ped_combination[0].index, ped_combination[1].index])
    array_a = np.array(list_a)
    array_b = np.array(list_b)
    array_index = np.array(index_list)
    differences = array_a - array_b
    if len(differences) == 0:
        return
    distances = np.linalg.norm(differences, axis=1)
    indices = np.where(distances < min_distance)[0]

    mde_index_pairs = array_index[indices]
    mde_corrections = (min_distance / (distances[indices][:, None] + ft.EPS) - 1) * differences[indices] / 2
    ordered_corrections = np.zeros([position_array.shape[0], position_array.shape[1]])
    for it in range(len(mde_index_pairs)):
        pair = mde_index_pairs[it]
        ordered_corrections[pair[0]] += mde_corrections[it]
        ordered_corrections[pair[1]] -= mde_corrections[it]
    return ordered_corrections
