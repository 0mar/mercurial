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

def mde2(cells, position_array, min_distance):
    all_corrections = np.zeros((position_array.shape[0], position_array.shape[1]))
    index_list_list = []
    for cell in cells:
        index_list_list.append([ped.index for ped in cell.pedestrian_set])

    for index_list in index_list_list:
        index_list = sorted(index_list)
        if len(index_list) > 1:
            position = position_array[index_list]
            n = len(index_list)
            replica = np.tile(position, (n, 1, 1))
            differences = np.transpose(replica, axes=[1, 0, 2]) - replica
            distance = np.linalg.norm(differences, axis=2)
            distance[distance > min_distance] = np.nan
            corrections = -differences / (distance[:, :, None] + ft.EPS) * (min_distance - distance[:, :, None]) / 2
            corrections = np.nan_to_num(corrections)
            total_corrections = np.sum(corrections, axis=0)
            all_corrections[index_list] += total_corrections
    return all_corrections
