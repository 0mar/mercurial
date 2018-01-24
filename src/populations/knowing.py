import numpy as np
from populations.base import Population
from math_objects import functions as ft
from scipy.ndimage import gaussian_filter
from math_objects.scalar_field import ScalarField as Field
from lib.wdt import get_weighted_distance_transform
import params


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
        self.on_step_functions = []
        self.on_step_functions.append(self.assign_velocities)
        self.dx = self.dy = None
        self.potential_field = None
        self.pot_grad_x = self.pot_grad_y = None
        self.grad_x_func = self.grad_y_func = None

    def prepare(self):
        """
        Called before the simulation starts. Fix all parameters and bootstrap functions.

        :return: None
        """
        super().prepare()
        cost_field = self._add_obstacle_discomfort(radius=params.obstacle_clearance)
        wdt = get_weighted_distance_transform(cost_field)
        self.dx, self.dy = self.scene.size.array / wdt.shape
        self.potential_field = Field(wdt.shape, Field.Orientation.center, 'potential', (self.dx, self.dy))
        self.potential_field.array = wdt
        # self.fire_effects = self._get_fire_effects()
        # self.obstacle_discomfort_field += self.fire_effects
        # self.discomfort_field.update(self.obstacle_discomfort_field.copy())
        self.pot_grad_x = Field(wdt.shape, Field.Orientation.vertical_face, 'pot_grad_x', (self.dx, self.dy))
        np.seterr(invalid='ignore')
        self.pot_grad_y = Field(wdt.shape, Field.Orientation.horizontal_face, 'pot_grad_y', (self.dx, self.dy))
        self.compute_potential_gradient()
        self.grad_x_func = self.pot_grad_x.get_interpolation_function()
        self.grad_y_func = self.pot_grad_y.get_interpolation_function()

    def _add_obstacle_discomfort(self, radius):
        """
        Use a gaussian filter (image blurring) to obtain a layer of discomfort around the obstacles
        The radius specifies how far the discomfort reaches. This radius is related to pedestrian size but can vary
        among different scenarios

        :param radius: SD of gaussian filter. Higher means lower values but longer range.
        :return: An adjusted cost field
        """
        cost_field = self.scene.env_field.copy()
        cost_field[cost_field == np.inf] = 0
        cost_field[self.scene.env_field == np.inf] = np.max(cost_field) * 2
        cost_field = gaussian_filter(cost_field, sigma=radius)
        cost_field[self.scene.env_field == np.inf] = np.inf
        cost_field[self.scene.env_field == 0] = 0
        return cost_field

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
        solved_grad_x = self.grad_x_func.ev(self.scene.position_array[:, 0][self.indices],
                                            self.scene.position_array[:, 1][self.indices])
        solved_grad_y = self.grad_y_func.ev(self.scene.position_array[:, 0][self.indices],
                                            self.scene.position_array[:, 1][self.indices])
        solved_grad = np.hstack([solved_grad_x[:, None], solved_grad_y[:, None]])
        self.scene.velocity_array[self.indices] = - self.scene.max_speed_array[:, None][
            self.indices] * solved_grad / \
                                                  np.linalg.norm(solved_grad + ft.EPS, axis=1)[:, None]

    def step(self):
        """
        Computes the scalar fields (in the correct order) necessary for the dynamic planner.
        If plotting is enabled, updates the plot.
        :return: None
        """
        [step() for step in self.on_step_functions]
