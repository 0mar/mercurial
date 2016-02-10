import time

__author__ = 'omar'
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import math
import operator
import functions as ft
from geometry import Point
from scalar_field import ScalarField as Field
# from cython_modules.dynamic_planner_cy import compute_density_and_velocity_field
from fortran_modules.grid_computer import comp_dens_velo
from cython_modules.grid_computer_cy import compute_density_and_velocity_field
from cython_modules.dynamic_planner_cy import compute_potential_cy


class DynamicPlanner:
    """
    This class implements staggered grids.
    Density and measured velocity are defined on centers of cells.
    These fields are called center fields.

    Maximum speed and unit costs are defined on faces of cells
    These fields are called face fields.

    These class stores all arrays internally, in ScalarFields
    This is (supposedly) beneficial for performance.
    Note that this means the methods in this class are all stateful
    and that order of execution is important.

    For now, since the potential is a center field, we need an exit that covers cell centers.
    This means exits have a minimum size requirement
    Ugly exit placement can be resolved by building obstacle walls.
    """

    def __init__(self, scene, show_plot=False):
        """
        Initializes a dynamic planner object. Takes a scene as argument.
        Parameters are initialized in this constructor, still need to be validated.
        :param scene: scene object to impose planner on
        :return: dynamic planner object
        """
        # Initialize depending on scene or on grid_computer?
        self.scene = scene
        self.config = scene.config
        prop_dx = self.config['general'].getfloat('cell_size_x')
        prop_dy = self.config['general'].getfloat('cell_size_y')
        self.grid_dimension = tuple((self.scene.size.array / (prop_dx, prop_dy)).astype(int))
        self.dx, self.dy = self.scene.size.array / self.grid_dimension
        self.show_plot = show_plot

        self.density_epsilon = ft.EPS
        self.max_speed = 2
        self.smoothing_length = self.config['dynamic'].getfloat('smoothing_length')
        self.path_length_weight = self.config['dynamic'].getfloat('path_length_weight')
        self.time_weight = self.config['dynamic'].getfloat('time_weight')
        self.discomfort_field_weight = self.config['dynamic'].getfloat('discomfort_weight')

        # This is dependent on cell size, because of the discretization
        self.min_density = self.config['dynamic'].getfloat('min_density')
        self.max_density = self.config['dynamic'].getfloat('max_density')

        self.density_exponent = self.config['dynamic'].getfloat('density_exponent')
        self.density_threshold = (1 / 2) ** self.density_exponent
        self.all_cells = {(i, j) for i, j in np.ndindex(self.grid_dimension)}
        self.exit_cell_set = set()
        self.obstacle_cell_set = set()
        self.part_obstacle_cell_dict = dict()  # Immediately store the fractions
        dx, dy = self.dx, self.dy
        shape = self.grid_dimension

        self.density_field = Field(shape, Field.Orientation.center, 'density', (dx, dy))
        self.v_x = Field(shape, Field.Orientation.center, 'v_x', (dx, dy))
        self.v_y = Field(shape, Field.Orientation.center, 'v_y', (dx, dy))
        self.potential_field = Field(shape, Field.Orientation.center, 'potential', (dx, dy))
        self.discomfort_field = Field(shape, Field.Orientation.center, 'discomfort', (dx, dy))
        self.obstacle_discomfort_field = np.zeros(shape)

        self.pot_grad_x = Field(shape, Field.Orientation.vertical_face, 'pot_grad_x', (dx, dy))
        self.pot_grad_y = Field(shape, Field.Orientation.horizontal_face, 'pot_grad_y', (dx, dy))
        self.unit_field_dict = {}
        self.speed_field_dict = {}
        for direction in ft.VERTICAL_DIRECTIONS:
            self.unit_field_dict[direction] = Field(shape, Field.Orientation.horizontal_face,
                                                    'Unit field %s' % direction, (dx, dy))
            self.speed_field_dict[direction] = Field(shape, Field.Orientation.horizontal_face,
                                                     'Unit field %s' % direction, (dx, dy))

        for direction in ft.HORIZONTAL_DIRECTIONS:
            self.unit_field_dict[direction] = Field(shape, Field.Orientation.vertical_face, 'Unit field %s' % direction,
                                                    (dx, dy))
            self.speed_field_dict[direction] = Field(shape, Field.Orientation.vertical_face,
                                                     'Unit field %s' % direction, (dx, dy))
        self._compute_initial_interface()
        if self.show_plot:
            # Plotting hooks
            f, self.graphs = plt.subplots(2, 2)
            plt.show(block=False)

    def _compute_initial_interface(self):
        """
        Compute the initial zero interface; a potential field with zeros on exits
        and infinity elsewhere. Stores inside object.
        This method also validates exits. If no exit is found, the method raises an error.
        :return: None
        """
        self.initial_interface = np.ones(self.grid_dimension) * np.inf
        valid_exits = {goal: False for goal in self.scene.exit_list}
        goals = self.scene.exit_list
        for i, j in np.ndindex(self.grid_dimension):
            cell_center = Point([(i + 0.5) * self.dx, (j + 0.5) * self.dy])
            for goal in goals:
                if cell_center in goal:
                    self.initial_interface[i, j] = 0
                    valid_exits[goal] = True
                    self.exit_cell_set.add((i, j))
            for obstacle in self.scene.obstacle_list:
                if not obstacle.accessible:
                    if cell_center in obstacle:
                        # self.initial_interface[i,j] = np.inf
                        self.obstacle_cell_set.add((i, j))
        if not any(valid_exits.values()):
            raise RuntimeError("No cell centers in exit. Redo the scene")
        if not all(valid_exits.values()):
            ft.warn("%s not properly processed" % "/"
                    .join([repr(goal) for goal in self.scene.exit_list if not valid_exits[goal]]))

    def get_obstacle_potential_field(self):
        """
        Compute the Gaussian-like potential field surrounding each obstacle
        We transform our kernel to an ellipsis to service the rectangular obstacles.
        :return: An 2D array with potential contributions
        """
        # h = self.smoothing_length  # Obstacle factor
        # cell_centers = self.potential_field.mesh_grid
        # potential_obstacles = np.zeros(self.grid_dimension)
        # obstacle_form = self.config['dynamic']['obstacle_form']
        # for obstacle in self.scene.obstacle_list:
        #     if not obstacle.accessible:
        #
        #         if obstacle_form == 'rectangle':
        #             distances = np.maximum(np.abs((cell_centers[0] - obstacle.center[0]) / (obstacle.size[0] / 2)),
        #                                np.abs((cell_centers[1] - obstacle.center[1]) / (obstacle.size[1] / 2)))
        #         elif obstacle_form == 'ellips':
        #             distances = np.sqrt(np.abs((cell_centers[0] - obstacle.center[0]) / (obstacle.size[0] / 2)) ** 2 +
        #                                 np.abs((cell_centers[1] - obstacle.center[1]) / (obstacle.size[1] / 2)) ** 2)
        #         else:
        #             raise ValueError("Obstacle form %s not recognized" % obstacle_form)
        #         weights = weight_function(distances / h)
        #         potential_obstacles += weights
        # return potential_obstacles

        for (i, j) in np.ndindex(self.discomfort_field.array.shape):
            location = np.array([self.discomfort_field.x_range[i], self.discomfort_field.y_range[j]])
            if self.scene.is_accessible(Point(location)):
                self.discomfort_field.array[i, j] = 0
            else:
                self.discomfort_field.array[i, j] = 3

    def _exists(self, index, max_index=None):
        """
        Checks whether an index exists within the max_index dimensions
        :param index: 2D index tuple
        :param max_index: max index tuple
        :return: true if lower than tuple, false otherwise
        """
        if not max_index:
            max_index = self.grid_dimension
        return (0 <= index[0] < max_index[0]) and (0 <= index[1] < max_index[1])

    def compute_density_and_velocity_field(self):
        """
        Compute the density and velocity field in cell centers as done in Treuille et al. (2004).
        Density and velocity are splatted to only surrounding cell centers.
        This is a naive implementation, looping over all pedestrians
        :return: (density, velocity_x, velocity_y) as 2D arrays
        """
        density_field = np.zeros(self.grid_dimension) + self.density_epsilon
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
        # For each cell center surrounding the pedestrian, add (in both dimensions) 1-\delta . or \delta .

        for pedestrian in self.scene.pedestrian_list:
            if self.scene.active_entries[pedestrian.index]:
                for x in range(2):
                    x_coord = cell_center_indices[pedestrian.index][0] + x
                    if 0 <= x_coord < self.grid_dimension[0]:
                        for y in range(2):
                            y_coord = cell_center_indices[pedestrian.index][1] + y
                            if 0 <= y_coord < self.grid_dimension[1]:
                                density_field[x_coord, y_coord] += density_contributions[x][y][pedestrian.index]
                                v_x[x_coord, y_coord] += \
                                    density_contributions[x][y][pedestrian.index] * pedestrian.velocity.x
                                v_y[x_coord, y_coord] += \
                                    density_contributions[x][y][pedestrian.index] * pedestrian.velocity.y
        # For each pedestrian, and for each surrounding cell center, add the corresponding density distribution.
        self.v_x.update(v_x / density_field)
        self.v_y.update(v_y / density_field)
        density_field -= self.density_epsilon
        self.density_field.update(density_field)

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
        normal = ft.DIRECTIONS[direction]
        rel_density = self.density_field.normalized(self.min_density, self.max_density)
        measured_rel_dens = Field.get_with_offset(rel_density, direction)
        if direction in ft.HORIZONTAL_DIRECTIONS:
            avg_dir_speed = np.maximum(Field.get_with_offset(self.v_x.array, direction) * normal[0], 0)
        elif direction in ft.VERTICAL_DIRECTIONS:
            avg_dir_speed = np.maximum(Field.get_with_offset(self.v_y.array, direction) * normal[1], 0)
        else:
            raise ValueError("Direction %s not recognized" % direction)

        speed_field = self.max_speed + measured_rel_dens * (avg_dir_speed - self.max_speed) + 0.01
        self.speed_field_dict[direction].update(speed_field)

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
        # self.discomfort_field.update(
        #    0 * self.obstacle_discomfort_field + 1.0 * self.density_field.normalized(self.min_density,
        #                                                                              self.max_density))
        pass

    def compute_random_discomfort_field(self):
        """
        Obtain discomfort field G.
        We pick a random field so we can see the influence of this field to the paths
        :return: None
        """
        # Something with continuity would be better.
        self.discomfort_field.update(np.random.random(self.grid_dimension))

    def compute_unit_cost_field(self, direction):
        """
        Compute the unit cost vector field in the provided direction
        Updates the class unit cost scalar field
        :return: None
        """
        alpha = self.path_length_weight
        beta = self.time_weight
        gamma = self.discomfort_field_weight
        f = self.speed_field_dict[direction].array
        g = self.discomfort_field.with_offset(direction)
        self.unit_field_dict[direction].update(alpha + (f + beta + gamma * g) / (f + ft.EPS))

    def compute_potential_field(self):
        """
        Compute the potential field as a function of the unit cost.
        Also computes the gradient of the potential field
        Implemented using the fast marching method

        Potential is initialized with zero on exits, and a fixed high value on inaccessible cells.
        :return:
        """
        opposites = {'left': 'right', 'right': 'left', 'up': 'down', 'down': 'up'}
        # This implementation is allowed to be naive: it's costly and should be implemented in FORTRAN
        # But maybe do a heap structure first?
        potential_field = self.initial_interface.copy()
        known_cells = self.exit_cell_set.copy()
        unknown_cells = self.all_cells - known_cells - self.obstacle_cell_set
        # All the inaccessible cells are not required.

        def get_new_candidate_cells(new_known_cells):  # Todo: Finalize list/set interfacing
            new_candidate_cells = set()
            for cell in new_known_cells:
                for direction in ft.DIRECTIONS.values():
                    nb_cell = (cell[0] + direction[0], cell[1] + direction[1])
                    if self._exists(nb_cell) and nb_cell not in known_cells and nb_cell not in self.obstacle_cell_set:
                        new_candidate_cells.add(nb_cell)
            return new_candidate_cells

        def compute_potential(cell):
            """
            Computes the potential in one cell, using potential in neighbouring cells.
            """
            # Find the minimal directions along a grid cell.
            # Assume left and below are best, then overwrite with right and up if they are better
            neighbour_pots = {direction: np.Inf for direction in ft.DIRECTIONS}

            hor_potential = ver_potential = 0
            hor_cost = ver_cost = np.Inf

            for direction in ft.DIRECTIONS:
                normal = ft.DIRECTIONS[direction]
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
                cost = self.unit_field_dict[opposites[direction]].array[face_index]
                # Cost to go from there to here
                neighbour_pots[direction] = pot + cost
                # total potential
                if neighbour_pots[direction] < neighbour_pots[opposites[direction]]:
                    if direction in ft.HORIZONTAL_DIRECTIONS:
                        hor_potential = pot
                        hor_cost = cost
                        # lowest in horizontal direction
                    elif direction in ft.VERTICAL_DIRECTIONS:
                        ver_potential = pot
                        ver_cost = cost
                        # lowest in vertical direction
                    else:
                        raise ValueError("Direction unknown")
            # Coefficients of quadratic equation
            a = 1 / hor_cost ** 2 + 1 / ver_cost ** 2
            b = -2 * (hor_potential / hor_cost ** 2 + ver_potential / ver_cost ** 2)
            c = (hor_potential / hor_cost) ** 2 + (ver_potential / ver_cost) ** 2 - 1

            D = b ** 2 - 4 * a * c
            x_high = (2 * c) / (-b - math.sqrt(D))
            # Might not be obvious, but why we take the largest root is found in report.
            return x_high

        candidate_cells = {cell: compute_potential_cy(cell, potential_field, self.unit_field_dict, opposites)
                           for cell in get_new_candidate_cells(known_cells)}

        new_candidate_cells = get_new_candidate_cells(known_cells)
        # Todo: Proposed improvement: new_candidate_cells = candidate_cells.keys()
        while unknown_cells:
            for candidate_cell in new_candidate_cells:
                if False:
                    potential = compute_potential(candidate_cell)
                else:
                    potential = compute_potential_cy(candidate_cell, potential_field, self.unit_field_dict, opposites)
                candidate_cells[candidate_cell] = potential
            sorted_candidates = sorted(candidate_cells.items(), key=operator.itemgetter(1))  # Todo: Can we reuse this?
            best_cell = sorted_candidates[0][0]
            min_potential = candidate_cells.pop(best_cell)
            potential_field[best_cell] = min_potential
            unknown_cells.remove(best_cell)
            known_cells.add(best_cell)
            new_candidate_cells = get_new_candidate_cells({best_cell})
        self.potential_field.update(potential_field)

    def compute_potential_gradient(self):
        """
        Compute a gradient component approximation of the provided field.
        Only computed for face fields.
        Gradient approximation is computed using a central difference scheme
        """
        left_field = self.potential_field.array[:-1, :]
        right_field = Field.get_with_offset(self.potential_field.array, 'right')
        assert self.pot_grad_x.array.shape == left_field.shape
        self.pot_grad_x.update((right_field - left_field) / self.dx)
        self.pot_grad_x.array[np.logical_not(np.isfinite(self.pot_grad_x.array))] =0
        down_field = self.potential_field.array[:, :-1]
        up_field = Field.get_with_offset(self.potential_field.array, 'up')
        assert self.pot_grad_y.array.shape == up_field.shape
        self.pot_grad_y.update((up_field - down_field) / self.dy)
        self.pot_grad_y.array[np.logical_not(np.isfinite(self.pot_grad_y.array))] =0


    def assign_velocities(self):
        """
        Interpolates the potential gradients for this time step and computes the velocities.
        :return: None
        """
        grad_x_func = self.pot_grad_x.get_interpolation_function()
        grad_y_func = self.pot_grad_y.get_interpolation_function()
        solved_grad_x = grad_x_func.ev(self.scene.position_array[:, 0], self.scene.position_array[:, 1])
        solved_grad_y = grad_y_func.ev(self.scene.position_array[:, 0], self.scene.position_array[:, 1])
        solved_grad = np.hstack([solved_grad_x[:, None], solved_grad_y[:, None]])
        self.scene.velocity_array = - self.scene.max_speed_array[:, None] * solved_grad / \
                                    np.linalg.norm(solved_grad + ft.EPS, axis=1)[:, None]

    def step(self):
        """
        Computes the scalar fields (in the correct order) necessary for the dynamic planner.
        If plotting is enables, updates the plot.
        :return: None
        """
        time1 = time.time()
        if True:  # FORTRAN implementation. Set False for Cython implementation
            dens_f, v_x_f, v_y_f = comp_dens_velo(self.scene.position_array, self.scene.velocity_array,
                                                  self.scene.active_entries, self.grid_dimension[0],
                                                  self.grid_dimension[1], self.dx, self.dy)
        else:
            dens_f, v_x_f, v_y_f = compute_density_and_velocity_field(self.grid_dimension, np.array([self.dx, self.dy]),
                                                                      self.scene.position_array,
                                                                      self.scene.velocity_array,
                                                                      self.scene.active_entries)
        # time2 = time.time()
        self.density_field.update(dens_f)
        self.v_x.update(v_x_f)
        self.v_y.update(v_y_f)
        self.compute_discomfort_field()
        #time3 = time.time()

        for direction in ft.DIRECTIONS:
            self.compute_speed_field(direction)
            self.compute_unit_cost_field(direction)
            # assert np.all(self.unit_field_dict[direction].array>0)
        #time4 = time.time()

        self.compute_potential_field()
        self.compute_potential_gradient()
        self.assign_velocities()
        if self.show_plot:
            self.plot_grid_values()
        #time5 = time.time()
        self.scene.move_pedestrians()  # Todo: Decide to put this here or in simulation manager
        # time6 = time.time()
        #print("Timings:\n%s"%"\n".join(["%.4f"%number for number in [time2-time1,time6-time2]]))
        self.scene.correct_for_geometry()
        self.scene.find_finished_pedestrians()

    def plot_grid_values(self):
        """
        Plots the grid values density, discomfort and potential
        :return: None
        """
        for graph in self.graphs.flatten():
            graph.cla()
        self.graphs[0, 0].imshow(np.rot90(self.density_field.array))
        self.graphs[0, 0].set_title('Density')
        self.graphs[1, 0].imshow(np.rot90(self.discomfort_field.array))
        self.graphs[1, 0].set_title('Discomfort')
        self.graphs[0, 1].imshow(np.rot90(self.potential_field.array))
        self.graphs[0, 1].set_title('Potential field')
        self.graphs[1, 1].quiver(self.potential_field.mesh_grid[0][1:-1, 1:-1],
                                 self.potential_field.mesh_grid[1][1:-1, 1:-1],
                                 ft.normalize(self.potential_field.gradient('x')[:, 1:-1]),
                                 ft.normalize(self.potential_field.gradient('y')[1:-1, :]), scale=1, scale_units='xy')
        plt.show(block=False)
