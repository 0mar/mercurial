import matplotlib

matplotlib.use('TkAgg')
import numpy as np
import operator
from math_objects import functions as ft
from math_objects.geometry import Point
from math_objects.scalar_field import ScalarField as Field
from fortran_modules.velocity_averager import average_velocity
import matplotlib.pyplot as plt

class PotentialInterpolator:
    """
    Two types of planners, based on awareness:
    Aware pedestrians are guided by a potential field
    Unaware pedestrians are guided by a swarm force.
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
        self.averaging_length = 10*self.pedestrians.size

        # Initialize classic potential transporter
        # to obtain potential field
        potential = PotentialTransporter(self.scene)
        self.potential_x = potential.grad_x_func
        self.potential_y = potential.grad_y_func

    def assign_velocities(self):
        """
        Three options for the unaware velocities:
        1. We set new velocities for followers equal to the average velocities.
        Problem if there are no people around
        dx_i/dt = v* = sum_j{w_ij * v_j)

        2. We weigh the new velocities with the densities
        dx_i/dt = alpha(rho)*(dx_i/dt+ v*)

        3. We don't change the velocity but the acceleration based on neighbours
        dv_i/dt = sum_j{w_ij(v_i-v_j)}
        |dx_i/dt| = v_i
        We choose the third, but should check what happens with the first two
        """
        # Aware velocities
        pot_descent_x = self.potential_x.ev(self.scene.position_array[:, 0], self.scene.position_array[:, 1])
        pot_descent_y = self.potential_y.ev(self.scene.position_array[:, 0], self.scene.position_array[:, 1])
        pot_descent = np.hstack([pot_descent_x[:, None], pot_descent_y[:, None]])
        self.scene.velocity_array[self.scene.aware_pedestrians] = - pot_descent[self.scene.aware_pedestrians]
        # Unaware velocities
        swarm_force = get_swarm_force(self.scene.positions, self.scene.velocities,self.scene.size.x,self.scene.size.y,self.averaging_length)
        self.velocities[np.logical_not(self.scene.aware_pedestrians)] = swarm_force[np.logical_not(self.scene.aware_pedestrians)]*self.dt
        # Normalizing velocities
        self.velocities *= self.max_speed_array[:,None] / np.linalg.norm(self.velocities+ft.EPS,axis=1)[:,None]
        #TODO: Check sign in this function
        #TODO: Insert a boolean array in fortran code to avoid unnecessary comps

    def step(self):
        """
        Computes the scalar fields (in the correct order) necessary for the dynamic planner.
        If plotting is enabled, updates the plot.
        :return: None
        """
        self.assign_velocities()
        self.scene.move_pedestrians()
        self.scene.correct_for_geometry()

    def nudge_stationary_pedestrians(self):
        stat_ped_array = self.scene.get_stationary_pedestrians()
        num_stat = np.sum(stat_ped_array)
        if num_stat > 0:
            nudge = (np.random.random((num_stat, 2)) - 0.5)*2
            correction = self.scene.max_speed_array[stat_ped_array][:, None] * nudge * self.scene.dt
            self.scene.position_array[stat_ped_array] += correction
            self.scene.correct_for_geometry()