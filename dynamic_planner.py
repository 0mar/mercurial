__author__ = 'omar'

import numpy as np


class DynamicPlanner:
    """
    This class implements staggered grids.
    Density and measured velocity are defined on centers of cells.
    These fields are called center fields.

    Maximum speed and unit costs are defined on faces of cells
    These fields are called face fields.
    """
    HORIZONTAL_DIRECTIONS = ['left', 'right']
    VERTICAL_DIRECTIONS = ['up', 'down']
    DIRECTIONS = {'left': [-1, 0], 'right': [1, 0], 'up': [0, 1], 'down': [0, -1]}

    def __init__(self, scene):
        """
        Initializes a dynamic planner object. Takes a scene as argument.
        Parameters are initialized in this constructor, still need to be validated.
        :param scene: scene object to impose planner on
        :return: dynamic planner object
        """
        # Initialize depending on scene or on grid_computer?
        self.scene = scene
        self.grid_dimension = (20, 20)
        self.dx, self.dy = self.scene.size.array / self.grid_dimension

        # Todo: Replace with general eps
        self.density_epsilon = 0.001
        self.cell_center_dims = self.grid_dimension

        # Todo: Not class members?
        self.horizontal_faces_dims = (self.grid_dimension[0], self.grid_dimension[1] + 1)
        self.vertical_faces_dims = (self.grid_dimension[0] + 1, self.grid_dimension[1])

        self.max_speed = 2

        self.speed_field_weight = 1
        self.discomfort_field_weight = 1
        self.path_length_weight = 1

        self.min_density = 0
        # This is likely on another scale
        self.max_density = 0.7

        self.density_exponent = 2
        self.density_threshold = (1 / 2) ** self.density_exponent

        self.density = np.array(self.grid_dimension)
        self.v_x = np.array(self.grid_dimension)
        self.v_y = np.array(self.grid_dimension)
        self.speed_field_dict = {direction: None for direction in DynamicPlanner.DIRECTIONS}

    def _new_face_array(self, direction):
        if direction in DynamicPlanner.HORIZONTAL_DIRECTIONS:
            return np.zeros(self.horizontal_faces_dims)
        elif direction in DynamicPlanner.VERTICAL_DIRECTIONS:
            return np.zeros(self.vertical_faces_dims)
        else:
            raise ValueError("Direction %s not a direction" % direction)
    @staticmethod
    def _get_center_field_with_offset(center_field, direction):
        """
        Returns a slice of the center field with all {direction} neighbour values of the cells.
        So, offset 'top' returns a center field slice omitting the bottom row.
        Example present in one of the ipython notebooks.
        :param center_field: center field under consideration
        :param direction: any of the 4 directions
        :return: slice of center field with almost the same dimensions
        """
        size = center_field.shape
        normal = DynamicPlanner.DIRECTIONS[direction]
        return center_field[max(0, normal[0]):size[0] + normal[0], max(0, normal[1]):size[1] + normal[1]]

    def compute_density_and_velocity_field(self):
        """
        Compute the density and velocity field in cell centers as done in Treuille et al. (2004).
        Density and velocity are splatted to only surrounding cell centers.
        This is a naive implementation, looping over all pedestrians
        :return: (density, velocity_x, velocity_y) as 2D arrays
        """
        # Todo: Integrate with grid_computer upon remerging project
        self.density = np.zeros(self.grid_dimension) + self.density_epsilon
        # Initialize density with an epsilon to facilitate division
        self.v_x = np.zeros(self.grid_dimension)
        self.v_y = np.zeros(self.grid_dimension)
        cell_size = np.array([self.dx, self.dy])
        cell_center_indices = (np.around(self.scene.position_array / cell_size) - np.array([1, 1])).astype(int)
        # These are the closest cell center indices whose coordinates both less than the corresponding pedestrians.
        cell_centers = (cell_center_indices + [0.5, 0.5]) * cell_size
        # These are the coordinates of those centers
        differences = self.scene.position_array - cell_centers
        # These are the differences between the cell centers and the pedestrian positions.
        assert np.all(differences <= cell_size) and np.all(differences >= 0)
        # They should all be positive and smaller than (dx,dy)
        rel_differences = differences / cell_size
        density_contributions = [[None, None], [None, None]]
        for x in range(2):
            for y in range(2):
                density_contributions[x][y] = np.minimum(
                    1 - rel_differences[:, 0] + (2 * rel_differences[:, 0] - 1) * x,
                    1 - rel_differences[:, 1] + (2 * rel_differences[:, 1] - 1) * y) ** self.density_exponent
        # For each cell center surrounding the pedestrian, add (in either dimension) 1-\delta . or \delta .

        for pedestrian in self.scene.pedestrian_list:
            for x in range(2):
                x_coord = cell_center_indices[pedestrian.counter][0] + x
                if 0 <= x_coord < self.grid_dimension[0]:
                    for y in range(2):
                        y_coord = cell_center_indices[pedestrian.counter][1] + y
                        if 0 <= y_coord < self.grid_dimension[1]:
                            self.density[x_coord, y_coord] += density_contributions[x][y][pedestrian.counter]
                            self.v_x[x_coord, y_coord] += \
                                density_contributions[x][y][pedestrian.counter] * pedestrian.velocity.x
                            self.v_y[x_coord, y_coord] += \
                                density_contributions[x][y][pedestrian.counter] * pedestrian.velocity.y
        # For each pedestrian, and for each surrounding cell center, add the corresponding density distribution.
        self.v_x /= self.density
        self.v_y /= self.density
        self.density -= self.density_epsilon

    def compute_speed_field(self, direction):
        """
        Obtain maximum speed field f, direction dependent.
        We choose our radius to be dx/2
        :param direction: which face is evaluated.
        :return: Discretized field as a nested numpy array
        """
        """
        Outline:
        Obtain density field relative to rho_min and rho_max and between 0 and 1
        Use these to compute f_max + rel_dens * (f_flow - f_max) for each direction

        """
        normal = DynamicPlanner.DIRECTIONS[direction]
        rel_density = self.get_normalized_field(self.density, self.min_density, self.max_density)
        measured_rel_dens = DynamicPlanner._get_center_field_with_offset(rel_density, direction)
        if direction in DynamicPlanner.HORIZONTAL_DIRECTIONS:
            measured_speed = np.maximum(DynamicPlanner._get_center_field_with_offset(self.v_x, direction) * normal[0],
                                        0)
        elif direction in DynamicPlanner.VERTICAL_DIRECTIONS:
            measured_speed = np.maximum(DynamicPlanner._get_center_field_with_offset(self.v_y, direction) * normal[1],
                                        0)
        else:
            raise ValueError("Direction %s not recognized" % direction)

        speed_field = self.max_speed + measured_rel_dens * (measured_speed - self.max_speed)
        self.speed_field_dict[direction] = speed_field

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
            return self.max_speed * np.ones(self.horizontal_faces_dims)
        elif direction in DynamicPlanner.VERTICAL_DIRECTIONS:
            return self.max_speed * np.ones(self.vertical_faces_dims)
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

    @staticmethod
    def get_normalized_field(field, min_value, max_value):
        """
        Normalizes field by bringing all values of field into [0,1]:
        for all x in field:     x < min_value => norm_x = 0
                    min_value < x < max_value => norm_x = (x - min_value)/(max_value - min_value)
                    max_value < x             -> norm_x = 1
        :param field: Discrete field under consideration
        :param min_value: lower value -> 0
        :param max_value: upper value ->1
        :return: Array with all values between 0 and 1
        """
        rel_field = (field - min_value) / (max_value - min_value)
        return np.minimum(1, np.maximum(rel_field, 0))

if __name__ == '__main__':
    from scene import Scene
    from geometry import Size, Velocity, Point

    n = 1
    scene = Scene(size=Size([100, 100]), obstacle_file='empty_scene.json', pedestrian_number=n)
    dyn_plan = DynamicPlanner(scene)
    ped = scene.pedestrian_list[0]
    ped.manual_move(Point([55.38, 91.66]))
    ped.velocity = Velocity([1, -0.5])
    print(ped.velocity)
    dyn_plan.compute_density_and_velocity_field()
    print("Density:\n%s\n\n" % dyn_plan.density)
    for direction in DynamicPlanner.DIRECTIONS:
        dyn_plan.compute_speed_field(direction)
        print(dyn_plan.speed_field_dict[direction])
