from enum import Enum

import numpy as np
from scipy.interpolate import RectBivariateSpline as Rbs

from math_objects import functions as ft


class ScalarField:
    """
    Wrapper class around 2D numpy arrays used as discrete scalar fields.
    Implements several convenience functions, like interpolation,
    obtaining the underlying mesh, computing a gradient approximation
    and normalizing. Also prints in the right orientation.
    """

    class Orientation(Enum):
        horizontal_face = (0, -1)
        vertical_face = (-1, 0)
        center = (0, 0)

    def __init__(self, grid_shape, orientation, name, cell_size=(1, 1), time_step=0):
        try:
            shape = (grid_shape[0] + orientation.value[0], grid_shape[1] + orientation.value[1])
        except TypeError:
            raise AttributeError("%s is not iterable" % grid_shape)
        if not isinstance(orientation, ScalarField.Orientation):
            raise AttributeError("Keyword orientation expects ScalarField.Orientation enum")
        self.array = np.zeros(shape, float)
        self.name = name
        self.orientation = orientation
        self.time_step = time_step
        self.dx, self.dy = cell_size

        self.width = self.dx * grid_shape[0]
        self.height = self.dy * grid_shape[1]
        if self.orientation.value[0] == 0:
            # x-coordinates lie on cell centers
            self.x_range = np.linspace(self.dx / 2, self.width - self.dx / 2, shape[0])
        else:
            # x-coordinates lie on cell faces
            self.x_range = np.linspace(self.dx, self.width - self.dx, shape[0])
        if self.orientation.value[1] == 0:
            # y-coordinates lie on cell centers
            self.y_range = np.linspace(self.dy / 2, self.height - self.dy / 2, shape[1])
        else:
            # y-coordinates lie on cell faces
            self.y_range = np.linspace(self.dy, self.height - self.dy, shape[1])
        x_grid, y_grid = np.meshgrid(self.x_range, self.y_range)
        self.mesh_grid = x_grid.T, y_grid.T

    def update(self, new_field):
        """
        Preferred way of updating the array from this field.
        :param new_field: np.array (must be same size) to update the Scalar field with
        :return:
        """
        if not self.array.shape == new_field.shape:
            ft.debug((self.name, self.array.shape, new_field.shape))
            assert self.array.shape == new_field.shape

        self.array = new_field.copy()
        # TODO: How about deleting this copy step?
        self.time_step += 1

    def __repr__(self):
        """
        repr override
        return: identifying string representing the scalar field
        """
        return "%s%s#%d" % (self.name, self.array.shape, self.time_step)

    def __str__(self):
        """
        str override
        return: Contents of the scalar field in the current time step.
        """
        field_repr = ""
        for row in np.rot90(self.array):
            field_repr += " [%s]\n" % "\t".join(["%4.2e" % val for val in row])
        return repr(self) + "\n[%s]" % field_repr[1:-1]

    def with_offset(self, direction, cutoff=1):
        """
        Returns a slice of the center field with all {direction} neighbour values of the cells.
        So, offset 'top' returns a center field slice omitting the bottom row.
        Example present in one of the ipython notebooks.
        :param direction: any of the 4 directions
        :return: slice of center field in indicated direction
        """
        if self.orientation != ScalarField.Orientation.center:
            raise NotImplementedError("We only take offset fields of center fields.")
        return ScalarField.get_with_offset(self.array, direction, cutoff)

    @staticmethod
    def get_with_offset(array, direction, cutoff=1):
        """
        Creates a subarray cutting values in a certain direction.
        See: ScalarField.with_offset
        return: slice of array in indicated direction
        """
        size = array.shape
        normal = ft.DIRECTIONS[direction]
        return array[max(0, normal[0] * cutoff):size[0] + normal[0] * cutoff,
               max(0, normal[1] * cutoff):size[1] + normal[1] * cutoff]

    def without_boundary(self, cutoff=1):
        """
        Slice array in every direction with {cutoff}
        return: slice of array
        """
        return self.array[cutoff:-cutoff, cutoff:-cutoff]

    def gradient(self, axis):
        """
        Second order central difference approximation of center fields
        :param axis: 'x' or 'y'
        :return:gradient of same shape except -2 in the axis direction
        """

        if axis == 'x':
            left_field = self.with_offset('left', 2)
            right_field = self.with_offset('right', 2)
            return (right_field - left_field) / (2 * self.dx)
        elif axis == 'y':
            up_field = self.with_offset('up', 2)
            down_field = self.with_offset('down', 2)
            return (up_field - down_field) / (2 * self.dy)
        else:
            raise AttributeError("Axis %s not known" % axis)


    def normalized(self, min_value=0, max_value=1):
        """
        Normalizes field by bringing all values of field into [0,1]:
        for all x in field:     x < min_value => norm_x = 0
                    min_value < x < max_value => norm_x = (x - min_value)/(max_value - min_value)
                    max_value < x             -> norm_x = 1
        :param min_value: lower value -> 0
        :param max_value: upper value ->1
        :return: Array with all values between 0 and 1
        """
        rel_field = (self.array - min_value) / (max_value - min_value)
        return np.clip(rel_field, 0, 1)

    def get_interpolation_function(self):
        return Rbs(self.x_range, self.y_range, self.array)
