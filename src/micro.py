import time

__author__ = 'omar'
import matplotlib

matplotlib.use('TkAgg')
import numpy as np
import operator
import functions as ft
from geometry import Point
from scalar_field import ScalarField as Field
from fortran_modules.potential_computer import compute_potential


class PotentialTransporter:
    """
    This should be a combination planner.
    Same as the dynamic planner, only ignoring the density so the potential field
    is computed once and we use the pressure computer from Narain
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

        self.path_length_weight = self.config['dynamic'].getfloat('path_length_weight')
        self.time_weight = self.config['dynamic'].getfloat('time_weight')
        self.discomfort_field_weight = self.config['dynamic'].getfloat('discomfort_weight')

        self.all_cells = {(i, j) for i, j in np.ndindex(self.grid_dimension)}
        self.exit_cell_set = set()
        self.obstacle_cell_set = set()
        self.part_obstacle_cell_dict = dict()  # Immediately store the fractions
        dx, dy = self.dx, self.dy
        shape = self.grid_dimension

        self.potential_field = Field(shape, Field.Orientation.center, 'potential', (dx, dy))
        self.discomfort_field = Field(shape, Field.Orientation.center, 'discomfort', (dx, dy))
        self.obstacle_discomfort_field = np.zeros(shape)
        self.compute_obstacle_discomfort()

        self.pot_grad_x = Field(shape, Field.Orientation.vertical_face, 'pot_grad_x', (dx, dy))
        self.pot_grad_y = Field(shape, Field.Orientation.horizontal_face, 'pot_grad_y', (dx, dy))
        self.grad_x_func = self.grad_y_func = None
        self.unit_field_dict = {}
        for direction in ft.VERTICAL_DIRECTIONS:
            self.unit_field_dict[direction] = Field(shape, Field.Orientation.horizontal_face,
                                                    'Unit field %s' % direction, (dx, dy))

        for direction in ft.HORIZONTAL_DIRECTIONS:
            self.unit_field_dict[direction] = Field(shape, Field.Orientation.vertical_face, 'Unit field %s' % direction,
                                                    (dx, dy))

        if not ft.VERBOSE:
            np.seterr(invalid='ignore')
        self.obtain_potential_field()

    def obtain_potential_field(self):
        self._compute_initial_interface()
        for direction in ft.DIRECTIONS:
            self.compute_unit_cost_field(direction)
        self.compute_potential_field()
        self.compute_potential_gradient()
        self.grad_x_func = self.pot_grad_x.get_interpolation_function()
        self.grad_y_func = self.pot_grad_y.get_interpolation_function()

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

    def compute_obstacle_discomfort(self):
        """
        Create a layer of discomfort around obstacles to repel pedestrians from those locations.
        """
        for (i, j) in np.ndindex(self.discomfort_field.array.shape):
            location = np.array([self.discomfort_field.x_range[i], self.discomfort_field.y_range[j]])
            if not self.scene.is_accessible(Point(location)):
                self.obstacle_discomfort_field[i - 1:i + 2, j - 1:j + 2] = 1

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

    def compute_unit_cost_field(self, direction):
        """
        Compute the unit cost vector field in the provided direction
        Updates the class unit cost scalar field
        :return: None
        """
        alpha = self.path_length_weight
        beta = self.time_weight
        gamma = self.discomfort_field_weight
        f = 1
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


        candidate_cells = {cell: compute_potential(cell[0],cell[1], potential_field,
                                                   self.unit_field_dict['left'],self.unit_field_dict['right'],
                                                   self.unit_field_dict['up'],self.unit_field_dict['down'], 9999)
                           for cell in get_new_candidate_cells(known_cells)}

        new_candidate_cells = get_new_candidate_cells(known_cells)
        while unknown_cells:
            for candidate_cell in new_candidate_cells:
                if False:
                    potential = compute_potential(candidate_cell)
                else:
                    potential = compute_potential(candidate_cell[0], candidate_cell[1], potential_field,
                                                  self.unit_field_dict['left'], self.unit_field_dict['right'],
                                                  self.unit_field_dict['up'], self.unit_field_dict['down'], 9999)
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
        self.pot_grad_x.array[np.logical_not(np.isfinite(self.pot_grad_x.array))] = 0
        down_field = self.potential_field.array[:, :-1]
        up_field = Field.get_with_offset(self.potential_field.array, 'up')
        assert self.pot_grad_y.array.shape == up_field.shape
        self.pot_grad_y.update((up_field - down_field) / self.dy)
        self.pot_grad_y.array[np.logical_not(np.isfinite(self.pot_grad_y.array))] = 0

    def assign_velocities(self):
        """
        Interpolates the potential gradients for this time step and computes the velocities.
        :return: None
        """

        solved_grad_x = self.grad_x_func.ev(self.scene.position_array[:, 0], self.scene.position_array[:, 1])
        solved_grad_y = self.grad_y_func.ev(self.scene.position_array[:, 0], self.scene.position_array[:, 1])
        solved_grad = np.hstack([solved_grad_x[:, None], solved_grad_y[:, None]])
        self.scene.velocity_array = - self.scene.max_speed_array[:, None] * solved_grad / \
                                    np.linalg.norm(solved_grad + ft.EPS, axis=1)[:, None]

    def step(self):
        """
        Computes the scalar fields (in the correct order) necessary for the dynamic planner.
        If plotting is enables, updates the plot.
        :return: None
        """

        self.scene.move_pedestrians()
        self.scene.correct_for_geometry()

    def nudge_stationary_pedestrians(self):
        stat_ped_array = self.scene.get_stationary_pedestrians()
        num_stat = np.sum(stat_ped_array)
        if num_stat > 0:
            nudge = np.random.random((num_stat, 2)) - 0.5
            correction = self.scene.max_speed_array[stat_ped_array][:, None] * nudge * self.scene.dt
            self.scene.position_array[stat_ped_array] += correction
            self.scene.correct_for_geometry()
