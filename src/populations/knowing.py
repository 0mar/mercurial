import numpy as np
from populations.base import Population
from math_objects import functions as ft
from scipy.ndimage import gaussian_filter
from math_objects.scalar_field import ScalarField as Field
from lib.wdt import get_weighted_distance_transform, plot


class Knowing(Population):
    """
    A potential field transporter that computes the weighted distance transform
    of the scene and uses the steepest gradient to move the pedestrians towards their goal.
    Combine with a macroscopic planner for interaction (repulsion)
    """

    def __init__(self, scene, number):
        """
        Initializes a following behaviour for the given population.

        :param scene: Simulation scene
        :param number: Initial number of people
        :return: Scripted pedestrian group
        """
        super().__init__(scene, number)
        self.fire_aware_indices = []
        self.on_step_functions = []
        self.on_step_functions.append(self.assign_velocities)
        self.dx = self.dy = None
        self.potential_field = None  # Todo: These three do not need to be on class level
        self.pot_grad_x = self.pot_grad_y = None
        self.grad_x_func = self.grad_y_func = None
        self.seen_fire = None
        # self.potential_field_with_fire = None
        # self.pot_grad_fire_x = self.pot_grad_fire_y = None
        self.grad_x_fire_func = self.grad_y_fire_func = None

    def prepare(self, params):
        """
        Called before the simulation starts. Fix all parameters and bootstrap functions.

        :return: None
        """
        super().prepare(params)
        # I also want one that contains the fire as an obstacle, inserted here
        cost_field = self._add_obstacle_discomfort(radius=self.params.obstacle_clearance)
        self.grad_x_func, self.grad_y_func = self._get_potential_planner(cost_field)
        if hasattr(self.params, 'fire'):
            fire = self.params.fire.get_fire_intensity(*self.scene.env_field.shape)
            fire_threshold = 0.01
            fire[fire > fire_threshold] = np.inf
            fire[fire <= fire_threshold] = 0

            fire_cost_field = self._add_obstacle_discomfort(radius=self.params.obstacle_clearance,
                                                            cost_field=(fire + self.scene.env_field))
            self.seen_fire = np.zeros(self.scene.total_pedestrians, dtype=bool)
            self.grad_x_fire_func, self.grad_y_fire_func = self._get_potential_planner(fire_cost_field)
            self.on_step_functions.insert(0, self.set_fire_knowledge)
            self.on_step_functions.append(self.assign_post_fire_velocities)
            # Overwrite accessibility: no pedestrians should be initiated in the fire
            self.scene.direction_field = self.potential_field.array
            self._correct_pedestrian_initial_positions()

    def _correct_pedestrian_initial_positions(self):
        """
        Not particularly proud of this hack, but I need a way to get the initialized pedestrians out of any fire zones.
        :return:
        """
        for pedestrian in self.scene.pedestrian_list:
            while not self.scene.is_accessible(pedestrian.position, at_start=True):
                pedestrian.position = self.scene.size.random_internal_point()

    def _get_potential_planner(self, cost_field):
        wdt = get_weighted_distance_transform(cost_field)
        self.dx, self.dy = self.scene.size.array / wdt.shape
        self.potential_field = Field(wdt.shape, Field.Orientation.center, 'potential', (self.dx, self.dy))
        self.potential_field.array = wdt
        self.pot_grad_x = Field(wdt.shape, Field.Orientation.vertical_face, 'pot_grad_x', (self.dx, self.dy))
        np.seterr(invalid='ignore')
        self.pot_grad_y = Field(wdt.shape, Field.Orientation.horizontal_face, 'pot_grad_y', (self.dx, self.dy))
        self.compute_potential_gradient()
        grad_x_func = self.pot_grad_x.get_interpolation_function()
        grad_y_func = self.pot_grad_y.get_interpolation_function()
        return grad_x_func, grad_y_func

    def _add_obstacle_discomfort(self, radius, cost_field=None):
        """
        Use a gaussian filter (image blurring) to obtain a layer of discomfort around the obstacles
        The radius specifies how far the discomfort reaches. This radius is related to pedestrian size but can vary
        among different scenarios

        :param radius: SD of gaussian filter. Higher means lower values but longer range.
        :return: An adjusted cost field
        """
        if cost_field is None:
            cost_field = self.scene.env_field.copy()
        new_cost_field = cost_field.copy()
        new_cost_field[new_cost_field == np.inf] = 0
        new_cost_field[cost_field == np.inf] = np.max(new_cost_field) * 2
        new_cost_field = gaussian_filter(new_cost_field, sigma=radius)
        new_cost_field[cost_field == np.inf] = np.inf
        new_cost_field[cost_field == 0] = 0
        return new_cost_field

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

    def set_fire_knowledge(self):
        fire_thres = 0.001  # Assert that is this low enough so that pedestrians don't get caught in a fire zone
        sees_fire = np.where(
            np.logical_and(self.params.fire.get_fire_intensity_at(self.scene.position_array) > fire_thres,
                           self.indices))
        self.seen_fire[sees_fire] = True

    def assign_velocities(self):
        """
        Interpolates the potential gradients for this time step and computes the velocities.
        Afterwards, overwrites the velocities for people who saw the fire
        :return: None
        """  # Todo: Remove chained indexing
        path_dir_x = self.grad_x_func.ev(self.scene.position_array[:, 0][self.indices],
                                         self.scene.position_array[:, 1][self.indices])
        path_dir_y = self.grad_y_func.ev(self.scene.position_array[:, 0][self.indices],
                                         self.scene.position_array[:, 1][self.indices])
        path_dir = np.hstack([path_dir_x[:, None], path_dir_y[:, None]])
        self.scene.velocity_array[self.indices] = - self.scene.max_speed_array[:, None][
            self.indices] * path_dir / np.linalg.norm(path_dir + ft.EPS, axis=1)[:, None]

    def assign_post_fire_velocities(self):
        """
        Take the routing for the people who know where the fire is.
        :return:
        """
        post_fire_path_x = self.grad_x_fire_func.ev(self.scene.position_array[:, 0][self.seen_fire],
                                                    self.scene.position_array[:, 1][self.seen_fire])
        post_fire_path_y = self.grad_y_fire_func.ev(self.scene.position_array[:, 0][self.seen_fire],
                                                    self.scene.position_array[:, 1][self.seen_fire])
        post_fire_path = np.hstack([post_fire_path_x[:, None], post_fire_path_y[:, None]])
        self.scene.velocity_array[self.seen_fire] = - self.scene.max_speed_array[:, None][
            self.seen_fire] * post_fire_path / np.linalg.norm(post_fire_path + ft.EPS, axis=1)[:, None]


    def step(self):
        """
        Computes the scalar fields (in the correct order) necessary for the dynamic planner.
        If plotting is enabled, updates the plot.
        :return: None
        """
        [step() for step in self.on_step_functions]
