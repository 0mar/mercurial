#!/usr/bin/env python

import numpy as np
import scipy.optimize as opt

EPS = 1e-9
VERBOSE = False


def error(msg: str):
    raise RuntimeError("\033[91mError: %s \033[0m" % msg)


def warn(msg: str):
    print("\033[93mWarning: %s \033[0m" % msg)


def fyi(msg: str):
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


def solve_convex_piece_lin_ineq(func, lower_bound: float, interval, max_iter=10, strict=False):
    """
    Solves a convex inequation. Stops when lower bound is reached and return argument.
    Inequation is solved by a simple iterative binary solver.
    :param func: function to be solved. In general, only piecewise linear functions are used
    :param lower_bound: lower bound for the function
    :param interval: interval to evaluate the function on.
    :param max_iter: maximum number of iterations to make
    :param strict: whether to evaluate a strict inequation
    :return: value for with the inequality holds. This should be checked though, because of finite precision
    """
    (left, right) = (interval.begin, interval.end)
    dx = 0.001

    cmp = np.less_equal
    if strict:
        cmp = np.less
    if cmp(func(left), lower_bound):
        return interval.begin
    elif cmp(func(right), lower_bound):
        return interval.end
    else:
        for counter in range(max_iter):
            mid_point = (left + right) / 2.
            mid_value = func(mid_point)
            if cmp(mid_value, lower_bound):
                return mid_point
            dmid = func(mid_point + dx) - mid_value
            if dmid > 0:
                right = mid_point
            else:
                left = mid_point
        else:
            return mid_point


def solve_convex_piece_lin_ineq2(func, lower_bound: float, interval, max_iter=10, strict=False):
    res = opt.minimize(func, x0=(interval.begin + interval.end) / 2)
    print(res)
    return res.x[0]


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
