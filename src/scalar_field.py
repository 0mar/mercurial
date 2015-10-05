__author__ = 'omar'
from enum import Enum

import numpy as np


class ScalarField(np.ndarray):
    """
    Wrapper class around 2D numpy arrays.

    Can be used for logging the values.
    """

    class Orientation(Enum):
        horizontal_face = (0, -1)
        vertical_face = (-1, 0)
        center = (0, 0)

    def __new__(cls, grid_shape, orientation, name, time_step=0):
        try:
            shape = (grid_shape[0] + orientation.value[0], grid_shape[1] + orientation.value[1])
        except TypeError:
            raise AttributeError("%s is not iterable" % grid_shape)
        if not isinstance(orientation, ScalarField.Orientation):
            raise AttributeError("Keyword orientation expects ScalarField.Orientation enum")
        obj = np.ndarray.__new__(cls, shape=shape, dtype=float, buffer=None, order='C')
        obj.name = name
        obj.orientation = orientation
        obj.time_step = time_step
        return obj

    def update(self, new_field):
        self[:, :] = new_field
        self.time_step += 1

    def domain(self):
        return None

    def __repr__(self):
        return "%s(%d,%d)#%d" % (self.name, self.shape[0], self.shape[1], self.time_step)

    def __str__(self):
        field_repr = ""
        for row in self:
            field_repr += " [%s]\n" % "\t".join(["%4.2f" % val for val in row])
        return repr(self) + "\n[%s]" % field_repr[1:-1]

    def __array_finalize__(self, obj):
        if obj is None:
            print("Making new object")
            return
        print("Starting from older object")
        if not hasattr(obj, 'orientation'):
            raise AttributeError('Numpy array is being cast to scalar field without orientation')
        self.name = getattr(obj, 'name', '')
        self.orientation = obj.orientation
        self.time_step = getattr(obj, 'time_step', '0')
