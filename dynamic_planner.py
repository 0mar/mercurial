__author__ = 'omar'

import numpy as np


class DynamicPlanner:
    HORIZONTAL_DIRECTIONS = ['left', 'right']
    VERTICAL_DIRECTIONS = ['up', 'down']
    DIRECTIONS = {'left': [-1, 0], 'right': [1, 0], 'up': [1, 0], 'down': [-1, 0]}

    def __init__(self, grid_computer):
        self.grid_computer = grid_computer
        self.scene = self.grid_computer.scene
        self.grid_dimension = (20, 20)
        self.cell_center_dims = self.grid_dimension
        self.horizontal_faces_dims = (self.grid_dimension[0], self.grid_dimension[1] + 1)
        self.vertical_faces_dims = (self.grid_dimension[0] + 1, self.grid_dimension[1])
        self.max_velocity = 2

        self.speed_field_weight = 1
        self.discomfort_field_weight = 1
        self.path_length_weight = 1

    def get_speed_field(self, direction):
        """
        Obtain maximum speed field f, direction dependent
        :param direction direction to be accounted for
        :return: Discretized field as a nested numpy array
        """
        """
        Outline:
        Obtain density field relative to rho_min and rho_max and between 0 and 1
        Use these to compute f_max + rel_dens * (f_flow - f_max) for each direction

        """
        if direction in DynamicPlanner.HORIZONTAL_DIRECTIONS:
            return self.max_velocity * np.ones(self.horizontal_faces_dims)
        elif direction in DynamicPlanner.VERTICAL_DIRECTIONS:
            return self.max_velocity * np.ones(self.vertical_faces_dims)
        else:
            raise ValueError('%s not a valid direction' % direction)

    def get_discomfort_field(self):
        """
        Obtain discomfort field G.
        Not prescribed in paper how to choose this.
        I'd go with something density dependent.
        :return: Discretized field as a nested numpy array
        """
        return np.random.random(self.grid_dimension)

    def get_unit_cost_field(self, direction):
        """
        Compute the total unit cost vector field in two directions
        :return: discretized field as a nested numpy array
        """
        if direction in DynamicPlanner.HORIZONTAL_DIRECTIONS:
            return self.max_velocity * np.ones(self.horizontal_faces_dims)
        elif direction in DynamicPlanner.VERTICAL_DIRECTIONS:
            return self.max_velocity * np.ones(self.vertical_faces_dims)
        else:
            raise ValueError('%s not a valid direction' % direction)

    def get_potential_field(self):
        """
        Compute the potential field as a function of the unit cost.
        Implemented using the fast marching method(?)

        :return:
        """

    def compute_gradient(self, field, axis):
        """
        Compute a gradient component approximation of the provided field.
        Considering 2th order CD as an approximation.
        :param field: discrete field to compute the gradient of
        :param axis: 'x' or 'y'
        :return: gradient component
        """
