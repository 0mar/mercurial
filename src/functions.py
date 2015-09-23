#!/usr/bin/env python

import numpy as np

EPS = 1e-6
VERBOSE = False


def error(msg: str):
    raise RuntimeError("\033[91mError: %s \033[0m" % msg)


def warn(msg: str):
    print("\033[93mWarning: %s \033[0m" % msg)


def log(msg: str):
    print("\033[92m%s \033[0m" % msg)


def debug(msg: str):
    if VERBOSE:
        print(msg)


def empty_method():
    pass


def rot_mat(angle: float):
    """
    Computes the rotation matrix for the given angle
    :param angle: rotation angle in radians
    :return: 2d rotation matrix
    """
    return np.array([[np.sin(angle), np.cos(angle)], [-np.cos(angle), np.sin(angle)]])


def rectangles_intersect(start_1, end_1, start_2, end_2, open_sets=False):
    """
    Expexct ordered coordinates
    :param start_1:
    :param end_1:
    :param start_2:
    :param end_2:
    :param open_sets:
    :return:
    """
    cmp = np.greater_equal
    if not open_sets:
        cmp = np.greater
    if cmp(start_1[0], end_2[0]) or cmp(start_2[0], end_1[0]) \
            or cmp(start_1[1], end_2[1]) or cmp(start_2[1], end_1[1]):
        return False
    else:
        return True


def get_hyperplane_function(p, q):
    """

    :param a:
    :param b:
    :return:
    """
    a1 = -(p[1] - q[1])
    a2 = p[0] - q[0]
    b = p[1] * (p[0] - q[0]) - p[0] * (p[1] - q[1])
    return lambda x1, x2: a1 * x1 + a2 * x2 - b
