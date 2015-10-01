__author__ = 'omar'
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from scipy.interpolate import RectBivariateSpline as Rbs
import functions as ft
from geometry import Point


class DynamicPlanner:
    """
    This class implements staggered grids.
    Density and measured velocity are defined on centers of cells.
    These fields are called center fields.

    Maximum speed and unit costs are defined on faces of cells
    These fields are called face fields.

    These class stores all arrays internally.
    This is (supposedly) beneficial for performance.
    Note that this means the methods in this class are all stateful
    and that order of execution is important.

    For now, since the potential is a center field, we need an exit that covers cell centers.
    This means exits need a certain width.
    As a consequence, obstacles should be introduced next to thick exits.
    """
    # Todo (after merge): We should build a wrapper around internal numpy 2D arrays.
    # Functionalities: Time steps, printing, updating, getting
    HORIZONTAL_DIRECTIONS = ['left', 'right']
    VERTICAL_DIRECTIONS = ['up', 'down']
    DIRECTIONS = {'left': [-1, 0], 'right': [1, 0], 'up': [0, 1], 'down': [0, -1]}

    def __init__(self, scene, show_plot):
        """
        Initializes a dynamic planner object. Takes a scene as argument.
        Parameters are initialized in this constructor, still need to be validated.
        :param scene: scene object to impose planner on
        :return: dynamic planner object
        """
        # Initialize depending on scene or on grid_computer?
        self.scene = scene
        self.grid_dimension = (20, 20)
        self.show_plot = show_plot
        self.dx, self.dy = self.scene.size.array / self.grid_dimension
        self.x_hor_face_range = np.linspace(self.dx / 2, self.scene.size.width - self.dx / 2, self.grid_dimension[0])
        self.y_hor_face_range = np.linspace(self.dy, self.scene.size.height - self.dy, self.grid_dimension[1] - 1)
        self.x_ver_face_range = np.linspace(self.dx, self.scene.size.width - self.dx, self.grid_dimension[0] - 1)
        self.y_ver_face_range = np.linspace(self.dy / 2, self.scene.size.height - self.dy / 2, self.grid_dimension[1])
        # Todo (after merge): Replace with general eps
        self.density_epsilon = 0.001
        self.cell_center_dims = self.grid_dimension

        self.max_speed = 2

        self.path_length_weight = 1
        self.time_weight = 1
        self.discomfort_field_weight = 1

        self.min_density = 0
        # This is likely on another scale than in grid computer
        self.max_density = 5

        self.density_exponent = 2
        self.density_threshold = (1 / 2) ** self.density_exponent

        self.density = None
        self.v_x = self.v_y = None
        self.potential_field = self.discomfort_field = None
        self.potential_grad_x = np.zeros((self.grid_dimension[0] - 1, self.grid_dimension[1]))
        self.potential_grad_y = np.zeros((self.grid_dimension[0], self.grid_dimension[1] - 1))
        self.unit_field_dict = {direction: None for direction in DynamicPlanner.DIRECTIONS}
        self.speed_field_dict = {direction: None for direction in DynamicPlanner.DIRECTIONS}

        self._compute_initial_interface()
        if self.show_plot:
            # Plotting hooks
            self.hor_mesh_x, self.hor_mesh_y = np.meshgrid(self.x_hor_face_range, self.y_hor_face_range, indexing='ij')
            self.ver_mesh_x, self.ver_mesh_y = np.meshgrid(self.x_ver_face_range, self.y_ver_face_range, indexing='ij')
            norm_x = np.linspace(self.dx / 2, self.scene.size.width - self.dx / 2, self.grid_dimension[0])
            norm_y = np.linspace(self.dy / 2, self.scene.size.height - self.dy / 2, self.grid_dimension[1])
            self.mesh_x, self.mesh_y = np.meshgrid(norm_x, norm_y, indexing='ij')
            f, self.graphs = plt.subplots(2, 2)
            plt.show(block=False)

    def _compute_initial_interface(self):
        """
        Compute the initial zero interface; a potential field with zeros on exits
        and infinity elsewhere. Stores inside object.
        This method also validates exits. If no exit is found, the method raises an error.
        :return: None
        """
        self.initial_interface = np.ones(self.grid_dimension)
        valid_exits = {goal: False for goal in self.scene.exit_set}
        goals = self.scene.exit_set
        for i, j in np.ndindex(self.grid_dimension):
            cell_center = Point([(i + 0.5) * self.dx, (j + 0.5) * self.dy])
            for goal in goals:
                if cell_center in goal:
                    self.initial_interface[i, j] = 0
                    valid_exits[goal] = True
        if not any(valid_exits.values()):
            raise RuntimeError("No cell centers in exit. Redo the scene")
        if not all(valid_exits.values()):
            ft.warn("%s not properly processed" % "/"
                    .join([repr(goal) for goal in self.scene.exit_set if not valid_exits[goal]]))

    def _exists(self, index, max_index=None):
        """
        Checks whether an index exists
        :param index: 2D index tuple
        :param max_index: max index tuple
        :return: true if lower than tuple, false otherwise
        """
        if not max_index:
            max_index = self.grid_dimension
        return (0 <= index[0] < max_index[0]) and (0 <= index[1] < max_index[1])

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
        # Todo: Move to C++.
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
        Outline:
        Obtain density field relative to rho_min and rho_max and between 0 and 1
        Use these to compute f_max + rel_dens * (f_flow - f_max) for each direction
        :param direction: which face is evaluated.
        :return: Discretized field as a nested numpy array
        """
        normal = DynamicPlanner.DIRECTIONS[direction]
        rel_density = self.get_normalized_field(self.density, self.min_density, self.max_density)
        measured_rel_dens = DynamicPlanner._get_center_field_with_offset(rel_density, direction)
        if direction in DynamicPlanner.HORIZONTAL_DIRECTIONS:
            avg_dir_speed = np.maximum(DynamicPlanner._get_center_field_with_offset(self.v_x, direction) * normal[0], 0)
        elif direction in DynamicPlanner.VERTICAL_DIRECTIONS:
            avg_dir_speed = np.maximum(DynamicPlanner._get_center_field_with_offset(self.v_y, direction) * normal[1], 0)
        else:
            raise ValueError("Direction %s not recognized" % direction)

        speed_field = self.max_speed + measured_rel_dens * (avg_dir_speed - self.max_speed)
        self.speed_field_dict[direction] = speed_field

    def compute_discomfort_field(self):
        """
        Obtain discomfort field G.
        Not prescribed in paper how to choose this.
        Proposal (density-dependent):
        * Discomfort is 0 for densities below min_density
        * Discomfort is 1 for densities over max_density
        * Discomfort increases linearly between these values.
        :return: None
        """
        self.discomfort_field = DynamicPlanner.get_normalized_field(self.density, self.min_density, self.max_density)


    def conpute_random_discomfort_field(self):
        """
        Obtain discomfort field G.
        We pick a random field so we can see the influence of this field to the paths
        :return: None
        """
        # Something with continuity would be better.
        self.discomfort_field = np.random.random(self.grid_dimension)

    def compute_unit_cost_field(self, direction):
        """
        Compute the unit cost vector field in the provided direction
        Stores face field in class object
        :return: None
        """
        alpha = self.path_length_weight
        beta = self.time_weight
        gamma = self.discomfort_field_weight
        f = self.speed_field_dict[direction]
        g = DynamicPlanner._get_center_field_with_offset(self.discomfort_field, direction)
        # Todo: Process obstacles
        # Todo (after merge): Change to EPS
        self.unit_field_dict[direction] = alpha + (f + beta + gamma * g) / (f + 0.001)


    def compute_potential_field(self):
        """
        Compute the potential field as a function of the unit cost.
        Also computes the gradient of the potential field
        Implemented using the fast marching method

        :return:
        """
        opposites = {'left': 'right', 'right': 'left', 'up': 'down', 'down': 'up'}
        # Experiment with a different (maybe smaller) grid,
        # so that you can define potentials on faces (and walls)
        # Example: Grid with shape + [1,1]
        # This implementation can be naive: it's costly and should be implemented in C(++)
        # But maybe do a heap structure first?
        self.potential_grad_x.fill(1)
        self.potential_grad_y.fill(1)
        where_zero = np.where(self.initial_interface ==0)
        potential_field = np.ones_like(self.initial_interface) * np.Inf
        potential_field[where_zero] = 0
        zeros = np.vstack(where_zero)
        all_cells = {(i, j) for i, j in np.ndindex(self.grid_dimension)}
        known_cells = {tuple(array) for array in zeros.T}
        unknown_cells = all_cells - known_cells

        def get_new_candidate_cells(new_known_cells):
            new_candidate_cells = set()
            for cell in new_known_cells:
                for direction in DynamicPlanner.DIRECTIONS.values():
                    nb_cell = (cell[0] + direction[0], cell[1] + direction[1])
                    if self._exists(nb_cell) and nb_cell in unknown_cells:
                        new_candidate_cells.add(nb_cell)
            return new_candidate_cells

        def compute_potential(cell):
            # Find the minimal directions along a grid cell.
            # Assume left and below are best, then overwrite with right and up if they are better
            neighbour_pots = {direction: np.Inf for direction in DynamicPlanner.DIRECTIONS}

            hor_face = tuple()
            hor_potential = ver_potential = 0
            ver_face = tuple()
            hor_cost = ver_cost = np.Inf

            for direction in DynamicPlanner.DIRECTIONS:
                normal = DynamicPlanner.DIRECTIONS[direction]
                # numerical direction
                nb_cell = (cell[0] + normal[0], cell[1] + normal[1])
                if not self._exists(nb_cell):
                    continue
                pot = potential_field[nb_cell]
                # potential in that neighbour field
                if direction == 'right':
                    face_index = (nb_cell[0] - 1, nb_cell[1])
                elif direction == 'up':
                    face_index = (nb_cell[0], nb_cell[1] - 1)
                    # Unit cost values are defined w.r.t faces, not cells!
                else:
                    face_index = nb_cell
                cost = self.unit_field_dict[opposites[direction]][face_index]
                # Cost to go from there to here
                neighbour_pots[direction] = pot + cost
                # total potential
                if neighbour_pots[direction] < neighbour_pots[opposites[direction]]:
                    if direction in DynamicPlanner.HORIZONTAL_DIRECTIONS:
                        hor_face = face_index
                        hor_potential = pot
                        hor_cost = cost
                        # lowest in horizontal direction
                    elif direction in DynamicPlanner.VERTICAL_DIRECTIONS:
                        ver_face = face_index
                        ver_potential = pot
                        ver_cost = cost
                        # lowest in vertical direction
                    else:
                        assert False
            coef = np.empty(3)
            coef[0] = 1 / hor_cost ** 2 + 1 / ver_cost ** 2
            coef[1] = -2 * (hor_potential / hor_cost ** 2 + ver_potential / ver_cost **2)
            coef[2] = (hor_potential / hor_cost) ** 2 + (ver_potential / ver_cost) ** 2 - 1
            # Coefficients of quadratic equation
            roots = np.roots(coef)
            # Roots of equation
            return roots[0]  # Which one?

        candidate_cells = get_new_candidate_cells(known_cells)
        while unknown_cells:
            min_potential = np.Inf
            best_cell = None
            for candidate_cell in candidate_cells:
                potential = compute_potential(candidate_cell)
                if potential < min_potential:
                    min_potential = potential
                    best_cell = candidate_cell
            potential_field[best_cell] = min_potential

            unknown_cells.remove(best_cell)
            candidate_cells.remove(best_cell)
            known_cells.add(best_cell)
            candidate_cells |= get_new_candidate_cells({best_cell})
        self.potential_field = potential_field

    def compute_potential_gradient(self):
        """
        Compute a gradient component approximation of the provided field.
        Considering 2th order CD as an approximation.
        :param field: discrete field to compute the gradient of
        :param axis: 'x' or 'y'
        :return: gradient component
        """
        left_field = self.potential_field[:-1, :]
        right_field = self._get_center_field_with_offset(self.potential_field, 'right')
        assert self.potential_grad_x.shape == left_field.shape
        self.potential_grad_x = (right_field - left_field) / self.dx

        down_field = self.potential_field[:, :-1]
        up_field = self._get_center_field_with_offset(self.potential_field, 'up')
        assert self.potential_grad_y.shape == up_field.shape
        self.potential_grad_y = (up_field - down_field) / self.dy

    def assign_velocities(self):
        grad_x_func = Rbs(self.x_ver_face_range, self.y_ver_face_range, self.potential_grad_x)
        grad_y_func = Rbs(self.x_hor_face_range, self.y_hor_face_range, self.potential_grad_y)
        # How:? speed_x_func = Rbs(self.x_range,self.y_range,self.speed_field_dict)
        solved_grad_x = grad_x_func.ev(self.scene.position_array[:, 0], self.scene.position_array[:, 1])
        solved_grad_y = grad_y_func.ev(self.scene.position_array[:, 0], self.scene.position_array[:, 1])
        solved_grad = np.hstack([solved_grad_x[:, None], solved_grad_y[:, None]])
        self.scene.velocity_array = -4 * solved_grad / np.linalg.norm(solved_grad, axis=1)[:,
                                                       None]
        # todo (after merge): max_velocity

    def step(self):
        self.compute_density_and_velocity_field()
        self.compute_discomfort_field()
        for direction in DynamicPlanner.DIRECTIONS:
            self.compute_speed_field(direction)
            self.compute_unit_cost_field(direction)

        self.compute_potential_field()
        self.compute_potential_gradient()
        self.assign_velocities()
        if self.show_plot:
            self.plot_grid_values()
        self.scene.time += self.scene.dt
        self.scene.move_pedestrians()
        for ped in self.scene.pedestrian_list:
            if self.scene.alive_array[ped.counter]:
                ped.correct_for_geometry()
                if ped.is_done():
                    self.scene.remove_pedestrian(ped)

    def plot_grid_values(self):
        for graph in self.graphs.flatten():
            graph.cla()
        self.graphs[0, 0].imshow(np.rot90(self.density))
        self.graphs[0, 0].set_title('Density')
        self.graphs[1, 0].imshow(np.rot90(self.discomfort_field))
        self.graphs[1, 0].set_title('Discomfort')
        self.graphs[0, 1].imshow(np.rot90(self.potential_field))
        self.graphs[0, 1].set_title('Potential field')
        # self.graphs[1, 1].imshow(np.rot90(self.discomfort_field))
        # self.graphs[1, 1].set_title('Discomfort')
        # self.graphs[1, 1].quiver(self.mesh_x, self.mesh_y, self.v_x, self.v_y, scale=1, scale_units='xy')
        # self.graphs[1, 1].set_title('Velocity field')
        # # self.graphs[1, 1].quiver(self.mesh_x, self.mesh_y, self.grad_p_x, self.grad_p_y, scale=1, scale_units='xy')
        # self.graphs[1, 1].set_title('Pressure gradient')
        plt.show(block=False)

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

#
# if __name__ == '__main__':
#     from scene import Scene
#     from geometry import Size, Velocity, Point
#
#     n = 1
#     g_scene = Scene(size=Size([100, 100]), obstacle_file='empty_scene.json', pedestrian_number=n)
#     dyn_plan = DynamicPlanner(g_scene)
#     ped = g_scene.pedestrian_list[0]
#     ped.manual_move(Point([55.38, 91.66]))
#     ped.velocity = Velocity([1, -0.5])
#     print(ped.velocity)
#     dyn_plan.compute_density_and_velocity_field()
#     dyn_plan.compute_discomfort_field()
#     for dir in DynamicPlanner.DIRECTIONS:
#         dyn_plan.compute_speed_field(dir)
#         dyn_plan.compute_unit_cost_field(dir)
