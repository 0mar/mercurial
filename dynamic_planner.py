__author__ = 'omar'

import numpy as np


class DynamicPlanner:
    HORIZONTAL_DIRECTIONS = ['left', 'right']
    VERTICAL_DIRECTIONS = ['up', 'down']
    DIRECTIONS = {'left': [-1, 0], 'right': [1, 0], 'up': [1, 0], 'down': [-1, 0]}

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

        self.cell_center_dims = self.grid_dimension
        self.horizontal_faces_dims = (self.grid_dimension[0], self.grid_dimension[1] + 1)
        self.vertical_faces_dims = (self.grid_dimension[0] + 1, self.grid_dimension[1])
        self.max_velocity = 2

        self.speed_field_weight = 1
        self.discomfort_field_weight = 1
        self.path_length_weight = 1

        self.min_density = 0
        self.max_density = 5

        self.density_exponent = 2
        self.density_threshold = (1 / 2) ** self.density_exponent

    def obtain_density_and_velocity_field(self):
        """
        Compute the density and velocity field in cell centers as done in Treuille et al. (2004).
        Density and velocity are splatted to only surrounding cell centers.
        This is a naive implementation, looping over all pedestrians
        :return: (density, velocity_x, velocity_y) as 2D arrays
        """
        # Todo: Integrate and sync with grid_computer upon remerging project
        dens = np.zeros(self.grid_dimension) + 0.001
        # Initialize density with an epsilon to facilitate division
        v_x = np.zeros(self.grid_dimension)
        v_y = np.zeros(self.grid_dimension)
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
                            dens[x_coord, y_coord] += density_contributions[x][y][pedestrian.counter]
                            v_x[x_coord, y_coord] += \
                                density_contributions[x][y][pedestrian.counter] * pedestrian.velocity.x
                            v_y[x_coord, y_coord] += \
                                density_contributions[x][y][pedestrian.counter] * pedestrian.velocity.y
        # For each pedestrian, and for each surrounding cell center, add the corresponding density distribution.
        v_x /= dens
        v_y /= dens

        return dens, v_x, v_y

    def obtain_speed_field(self, direction):
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


if __name__ == '__main__':
    from scene import Scene
    from geometry import Size, Velocity, Point

    n = 1
    scene = Scene(size=Size([100, 100]), obstacle_file='empty_scene.json', pedestrian_number=n)
    dyn_plan = DynamicPlanner(scene)
    ped = scene.pedestrian_list[0]
    ped.position = Point([55.38, 91.66])
    ped.velocity = Velocity([1, 1])
    (density, v_x, v_y) = dyn_plan.obtain_density_and_velocity_field()
    print(ped.position)
    for i in range(scene.number_of_cells[0]):
        for j in range(scene.number_of_cells[1]):
            if v_x[i, j] or v_y[i, j]:
                print("cell (%d,%d)" % (i, j))
                print("Density: %.4f" % density[i, j])
                print("v_x: %.4f" % v_x[i, j])
                print("v_y: %.4f" % v_y[i, j])
