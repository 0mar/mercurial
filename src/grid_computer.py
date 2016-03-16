__author__ = 'omar'
import matplotlib

matplotlib.use('TkAgg')
import time
import numpy as np
import matplotlib.pyplot as plt
from fortran_modules.micro_macro import comp_dens_velo
from fortran_modules.pressure_computer import compute_pressure
from scalar_field import ScalarField as Field
import cvxopt
import functions as ft
from geometry import Point

class GridComputer:
    """
    Class responsible for the macroscopic calculations of the crowd.
    Main tasks:
    1. Interpolating velocity on grid points
    2. Computing pressure
    3. Adapting global velocity field with pressure gradient
    4. Adapting individual velocity by global velocity field.
    """

    def __init__(self, scene, show_plot, apply_interpolation, apply_pressure):
        """
        Constructs a grid computer, responsible for the continuum calculations.
        The grid computer takes several parameters in its constructor.
        Numerical parameters are specified in the parameters file.
        :param scene: Scene on which we make the computations
        :param show_plot: enable matplotlib plotting of discrete fields
        :param apply_interpolation: impose group velocity on pedestrians
        :param apply_pressure: impose pressure on velocity field (and on pedestrians)
        :return: Grid computer object.
        """
        self.scene = scene
        prop_dx = scene.config['general'].getfloat('cell_size_x')
        prop_dy = scene.config['general'].getfloat('cell_size_y')
        self.grid_dimension = (self.scene.size.array / (prop_dx, prop_dy)).astype(int)
        self.dx, self.dy = self.scene.size.array / self.grid_dimension
        self.dt = self.scene.dt
        self.packing_factor = scene.config['dynamic'].getfloat('packing_factor')
        self.max_density = 2 * self.packing_factor / \
                           (np.sqrt(3) * self.scene.core_distance ** 2)
        cvxopt.solvers.options['show_progress'] = False
        self.smoothing_length = 2 * self.scene.core_distance
        self.basis_A = self.basis_v_x = self.basis_v_y = None
        self._last_solution = None
        self.show_plot = show_plot
        self.apply_interpolation = apply_interpolation or apply_pressure
        self.apply_pressure = apply_pressure
        dx, dy = self.dx, self.dy
        shape = self.grid_dimension
        self.obstacle_correction = np.zeros(shape)
        self._get_obstacle_coverage()
        self.density_field = Field(shape, Field.Orientation.center, 'density', (dx, dy))
        self.v_x = Field(shape, Field.Orientation.center, 'velocity_x', (dx, dy))
        self.v_y = Field(shape, Field.Orientation.center, 'velocity_y', (dx, dy))
        self.pressure_field = Field((shape[0] + 2, shape[1] + 2), Field.Orientation.center, 'pressure', (dx, dy))
        # If beneficial, we could employ a staggered grid

        if self.show_plot:
            # Plotting hooks
            f, self.graphs = plt.subplots(2, 2)
            plt.show(block=False)

    def _get_obstacle_coverage(self, precision=2):
        """
        Compute the fraction of the cells covered with obstacles
        Since an exact computation involves either big shot linear algebra
        or too much case distinctions, we sample the cell space.
        :param precision: row number of samples per cell. More samples, more precision.
        :return: an array approximating the inaccessible space in the cells
        """
        ft.log("Started sampling objects (%d checks)"%(precision**2*self.grid_dimension[0]*self.grid_dimension[1]))
        obstacle_coverage = np.zeros(self.obstacle_correction.shape)
        num_samples = (precision, precision)
        size = Point([self.dx, self.dy])
        for row, col in np.ndindex(obstacle_coverage.shape):
            covered_samples = 0
            begin = Point([row * self.dx, col * self.dy])
            for i, j in np.ndindex(num_samples):
                for obstacle in self.scene.obstacle_list:
                    if not obstacle.accessible:
                        if Point(begin.array + [i + 0.5, j + 0.5] / np.array(num_samples)*size.array) in obstacle:
                            covered_samples += 1
                            break
            obstacle_coverage[row, col] = covered_samples / (num_samples[0] * num_samples[1])
        # Correct the fully covered entries
        obstacle_coverage[obstacle_coverage == 1] = 0.8
        self.obstacle_correction = 1 / (1 - obstacle_coverage)
        ft.log("Finished sampling objects")

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

    def compute_pressure(self):
        pressure = compute_pressure(self.density_field.array + 0.1, self.v_x.array, self.v_y.array,
                                    self.dx, self.dy, self.dt, self.max_density)
        dim_p = np.reshape(pressure, (self.grid_dimension[0], self.grid_dimension[1]), order='F')
        padded_dim_p1 = np.pad(dim_p, (1, 1), 'constant', constant_values=1)
        self.pressure_field.update(padded_dim_p1)


    def adjust_velocity(self):
        """
        Adjusts the velocity field for the pressure gradient. We pad the pressure with an extra boundary
        so that the gradient is defined everywhere.
        :return: None
        """
        # Not using the update method.
        time1 = time.time()
        well_shaped_x_grad = self.pressure_field.gradient('x')[:, 1:-1]
        well_shaped_y_grad = self.pressure_field.gradient('y')[1:-1, :]
        self.v_x.array -= well_shaped_x_grad
        self.v_y.array -= well_shaped_y_grad
        ft.debug("Step 4: Applying pressure %.4f"%(time.time()-time1))


    def interpolate_pedestrians(self):
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
                                self.max_density)
        solved_velocity = np.hstack((solved_v_x[:, None], solved_v_y[:, None]))
        self.scene.velocity_array = self.scene.velocity_array + local_dens[:, None] / self.max_density * (
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
        if self.apply_interpolation or self.show_plot:
            n_x, n_y = self.grid_dimension
            dx, dy = self.scene.size.array / self.grid_dimension
            density_field, v_x, v_y = comp_dens_velo(self.scene.position_array, self.scene.velocity_array,
                                                     self.scene.active_entries, n_x, n_y, dx, dy, self.smoothing_length)
            self.density_field.update(self.obstacle_correction * density_field)
            self.v_x.update(v_x)
            self.v_y.update(v_y)
            ft.debug("Max allowed density: %.4f"%self.max_density)
            ft.debug("Max observed density: %.4f"%np.max(self.density_field.array))

        if self.apply_interpolation:

            if self.apply_pressure:
                self.compute_pressure()
                self.adjust_velocity()
                if self.show_plot:
                    self.plot_grid_values()
                # self.compute_pressure()
            # print("Time took: %.4f" % (time.time() - time1))
            self.interpolate_pedestrians()

    @staticmethod
    def weight_function(array, smoothing_length=1):
        """
        Using the Wendland kernel to determine the interpolation weight
        Calculation is performed in two steps to take advantage of numpy's speed
        :param array: Array of distances to apply the kernel on.
        :param smoothing_length: Steepness factor (standard deviation) of kernel
        :return: Weights of interpolation
        """
        array /= smoothing_length
        norm_constant = 7. / (4 * np.pi * smoothing_length * smoothing_length)
        first_factor = np.maximum(1 - array / 2, 0)
        weight = first_factor ** 4 * (1 + 2 * array)
        return weight * norm_constant
