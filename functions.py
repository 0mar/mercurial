#!/usr/bin/env python
import random
import math

import numpy as np


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
    return np.array([[np.sin(angle), np.cos(angle)], [-np.cos(angle), np.sin(angle)]])


# Return minimum of convex piecewise linear function. Stops when reached lower bound. High acc.
def solve_convex_piece_lin_ineq(func, lower_bound: float, interval: Interval, max_iter=10, strict=False):
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
