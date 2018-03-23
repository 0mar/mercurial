import matplotlib
import numpy as np
import params
from lib.micro_macro import comp_dens_velo
from lib.pressure_computer import compute_pressure

from math_objects import functions as ft
from math_objects.scalar_field import ScalarField as Field

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class Repel:
    """
    Class responsible for the macroscopic calculations of the crowd.
    Main tasks:
    1. Interpolating velocity on grid points
    2. Computing pressure
    3. Adapting global velocity field with pressure gradient
    4. Adapting individual velocity by global velocity field.
    """

    def __init__(self, scene):
        """
        Constructs a grid computer, responsible for the continuum calculations.
        The grid computer takes several parameters in its constructor.
        Numerical parameters are specified in the parameters file.
        :param scene: Scene on which we make the computations
        :return: Grid computer object.
        """
        self.scene = scene
        self.params = None
        self.on_step_functions = []
        self.basis_A = self.basis_v_x = self.basis_v_y = None
        self._last_solution = None
        self.show_plot = False
        self.grid_dimension = self.dx = self.dy = None

        # Fields to store the macroscopic quantities in
        self.density_field = self.v_x = self.v_y = self.pressure_field = None
        self.obstacle_field = None

    def prepare(self, params):
        """
        Called before the simulation starts. Fix all parameters and bootstrap functions.

        :params: Parameter object
        :return: None
        """
        self.params = params
        prop_dx = self.params.pressure_dx
        prop_dy = self.params.pressure_dy
        self.grid_dimension = (self.scene.size.array / (prop_dx, prop_dy)).astype(int)
        self.dx, self.dy = self.scene.size.array / self.grid_dimension

        self.density_field = Field(self.grid_dimension, Field.Orientation.center, 'density', (self.dx, self.dy))
        self.v_x = Field(self.grid_dimension, Field.Orientation.center, 'velocity_x', (self.dx, self.dy))
        self.v_y = Field(self.grid_dimension, Field.Orientation.center, 'velocity_y',
                         (self.dx, self.dy))  # Todo: We can probably easily stagger this
        self.pressure_field = Field((self.grid_dimension[0] + 2, self.grid_dimension[1] + 2), Field.Orientation.center, 'pressure', (self.dx, self.dy))
        self.on_step_functions.append(self.apply_repulsion)
        self.obstacle_field = self.scene.get_obstacles(*self.grid_dimension)
        if self.show_plot:
            # Plotting hooks
            f, self.graphs = plt.subplots(2, 2)
            plt.show(block=False)
            self.on_step_functions.append(self.plot_grid_values)


    def plot_grid_values(self):
        """
        Plot the density, velocity field, pressure, and pressure gradient.
        Plots are opened in a separate window and automatically updated.
        :return:
        """
        for graph in self.graphs.flatten():
            graph.cla()
        self.graphs[0, 0].imshow(np.rot90(self.density_field.array))
        self.graphs[0, 0].set_title('Density')
        self.graphs[1, 0].imshow(np.rot90(self.pressure_field.array))
        self.graphs[1, 0].set_title('Pressure')
        self.graphs[0, 1].quiver(self.v_x.mesh_grid[0], self.v_x.mesh_grid[1], self.v_x.array, self.v_y.array, scale=1,
                                 scale_units='xy')
        self.graphs[0, 1].set_title('Velocity field')
        self.graphs[1, 1].quiver(self.v_x.mesh_grid[0], self.v_x.mesh_grid[1],
                                 self.pressure_field.gradient('x')[:, 1:-1],
                                 self.pressure_field.gradient('y')[1:-1, :], scale=1, scale_units='xy')
        self.graphs[1, 1].set_title('Pressure gradient')
        plt.show(block=False)

    def get_macro_fields(self):
        """
        Compute the macroscopic (continuum) fields like density and velocity by interpolation of the
        microsopic pedestrian quantities.
        :return:
        """
        n_x, n_y = self.grid_dimension
        dx, dy = self.scene.size.array / self.grid_dimension
        density_field, v_x, v_y = comp_dens_velo(self.scene.position_array, self.scene.velocity_array,
                                                 self.scene.active_entries, n_x, n_y, dx, dy,
                                                 self.params.smoothing_length)
        self.density_field.update(density_field)
        self.v_x.update(v_x)
        self.v_y.update(v_y)
        ft.debug("Max allowed density: %.4f" % self.params.max_density)
        ft.debug("Max observed density: %.4f" % np.max(self.density_field.array))

    def compute_pressure(self):
        """
        Compute the pressure term
        We pad the pressure with an extra boundary
        so that the gradient is defined for each cell in the scene.
        """

        pressure = compute_pressure(self.density_field.array + 0.1, self.v_x.array, self.v_y.array,
                                    self.dx, self.dy, self.params.dt, self.params.max_density)
        dim_p = np.reshape(pressure, (self.grid_dimension[0], self.grid_dimension[1]), order='F')
        dim_p[self.obstacle_field.astype(bool)] = self.params.boundary_pressure
        # dim_p[self.scene.gutter_cells.astype(bool)] = self.gutter_pressure
        padded_dim_p = np.pad(dim_p, (1, 1), 'constant', constant_values=self.params.boundary_pressure)
        self.pressure_field.update(padded_dim_p)

    def adjust_velocity(self):
        """
        Adjusts the velocity field for the pressure gradient.
        :return: None
        """
        # Not using the update method.
        well_shaped_x_grad = self.pressure_field.gradient('x')[:, 1:-1]
        well_shaped_y_grad = self.pressure_field.gradient('y')[1:-1, :]
        self.v_x.array -= well_shaped_x_grad
        self.v_y.array -= well_shaped_y_grad

    def put_micro_changes(self):
        """
        Method that reconverts the velocity field to individual pedestrian positions.
        Input are the velocity x/y fields. We use a bivariate spline interpolation method
        to interpolate the velocities at the pedestrians position.
        These velocities are then weighed to the densities, normalized
        and added to the velocity field.
        :return: None
        """
        v_x_func = self.v_x.get_interpolation_function()
        v_y_func = self.v_y.get_interpolation_function()
        dens_func = self.density_field.get_interpolation_function()
        solved_v_x = v_x_func.ev(self.scene.position_array[:, 0], self.scene.position_array[:, 1])
        solved_v_y = v_y_func.ev(self.scene.position_array[:, 0], self.scene.position_array[:, 1])
        local_dens = np.minimum(dens_func.ev(self.scene.position_array[:, 0], self.scene.position_array[:, 1]),
                                self.params.max_density)
        solved_velocity = np.hstack((solved_v_x[:, None], solved_v_y[:, None]))
        self.scene.velocity_array = self.scene.velocity_array + local_dens[:, None] / self.params.max_density * (
            solved_velocity - self.scene.velocity_array) + ft.EPS
        self.scene.velocity_array /= \
            np.linalg.norm(self.scene.velocity_array, axis=1)[:, None] / (self.scene.max_speed_array[:, None] + ft.EPS)

    def step(self):
        """
        Performs one time step for the fluid simulator.
        Computes and interpolates the grid fields (density and velocity)
        Solves the LCP to compute the pressure field
        Adjusts the velocity field according to the pressure
        Adjusts the velocities of the pedestrians according to the velocity field
        :return: None
        """
        [step() for step in self.on_step_functions]

    def apply_repulsion(self):
        """
        The full iteration. Interpolate macrofields, solve the PDE,
        adjust the velocity field, and impose this on the pedestrians
        :return:
        """
        self.get_macro_fields()
        self.compute_pressure()
        self.adjust_velocity()
        self.put_micro_changes()
