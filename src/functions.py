import math

import numpy as np

# Numerical tolerance
EPS = 1e-6
# Verbosity
VERBOSE = False
HORIZONTAL_DIRECTIONS = ['left', 'right']
VERTICAL_DIRECTIONS = ['up', 'down']
DIRECTIONS = {'left': [-1, 0], 'right': [1, 0], 'up': [0, 1], 'down': [0, -1]}


def error(msg):
    """
    Report error to stdout and quit application
    :param msg: error report
    :raises: Runtime error with error report
    """
    raise RuntimeError("\033[91mError: %s \033[0m" % str(msg))


def warn(msg):
    """
    Print warning message to stdout
    :param msg: warning
    :return: None
    """
    print("\033[93mWarning: %s \033[0m" % str(msg))


def log(msg):
    """
    Print logging message to stdout
    :param msg: log string
    :return: None
    """
    print("\033[92m%s \033[0m" % str(msg))


def debug(msg):
    """
    Print debug information to stdout
    :param msg:
    :return:
    """
    if VERBOSE:
        print(msg)


def empty_method():
    """
    Placeholder for some function insert methods.
    :return: None
    """
    pass


def norm(a, b):
    """
    Computes a norm of two numbers. For single numbers, faster than np.linalg.norm
    :param a: float a
    :param b: float b
    :return: norm of (a,b)
    """
    return math.sqrt(a * a + b * b)


def normalize(array, safe=False):
    """
    Normalized the array, i.e. Makes sure all rows have norm 1.
    Does not check for zero rows, in that case numpy returns NaNs.
    :param array: Array to be normalized
    :param safe: account for zeros in array
    :return: new array with rows normalized one.
    """
    if not safe:
        return array / np.linalg.norm(array, axis=1)[:, None]
    else:
        return array / (np.linalg.norm(array, axis=1)[:, None] + EPS)



def is_close(a, b):
    """
    Checks if two numbers are sufficiently close together. Absolute tolerance only
    :param a: first number
    :param b: second number
    :return: True if difference is smaller than general epsilon
    """
    return math.fabs(a - b) < EPS


def rot_mat(angle):
    """
    Computes a 2D rotation matrix
    :param angle: rotation angle in radians
    :return: 2x2 rotation matrix
    """
    return np.array([[np.sin(angle), np.cos(angle)], [-np.cos(angle), np.sin(angle)]])


def rectangles_intersect(start_1, end_1, start_2, end_2, open_sets=False):
    """
    Expect ordered coordinates.
    Checks whether two rectangles have a nonempty intersection.
    :param start_1: Tuple with bottom coordinates of first rectangle
    :param end_1: Tuple with top coordinates of first rectangle
    :param start_2: Tuple with bottom coordinates of second rectangle
    :param end_2: Tuple with top coordinates of second rectangle
    :param open_sets: Whether we check for strict inequality or equality also suffices
    :return: True if rectangles intersect, false otherwise
    """
    cmp = np.greater_equal
    if not open_sets:
        cmp = np.greater
    if cmp(start_1[0], end_2[0]) or cmp(start_2[0], end_1[0]) \
            or cmp(start_1[1], end_2[1]) or cmp(start_2[1], end_1[1]):
        return False
    else:
        return True


def get_hyperplane_functional(p, q):
    """
    Returns a functional that computes whether a vector is on either side (or on) a hyperplane.
    The hyperplane (in 2D) is determined by the input vectors.
    :param p: one of the 2D vectors determining the hyperplane
    :param q:one of the 2D vectors (!=p) determining the hyperplane
    :return: 2D functional that is positive for vectors on one side of the hyperplane, negative
    for vectors on the other side, and zero for vectors on the plane
    """
    a1 = -(p[1] - q[1])
    a2 = p[0] - q[0]
    b = p[1] * (p[0] - q[0]) - p[0] * (p[1] - q[1])
    return lambda x1, x2: a1 * x1 + a2 * x2 - b


def increase_array_size(array, increment_factor=2):
    """
    Increases the size (first dimension) of a numpy array with the given factor
    :param array: Original array
    :param increment_factor: Multiplication factor for the size
    :return: array with same values for each entry of the old array, and 0 further
    """
    inc = int(increment_factor)
    new_shape = array.shape
    new_shape[0] = 2 * array.shape[0]
    new_array = np.zeros(new_shape)
    new_array[0:array.shape[0], :] = array
    return new_array
