__author__ = 'omar'
import matplotlib

matplotlib.use('TkAgg')
import time
import numpy as np
import scipy.io as sio
import scipy.sparse as ss
import matplotlib.pyplot as plt
from cython_modules.grid_computer import compute_density_and_velocity_field, solve_LCP_with_pgs
from scalar_field import ScalarField as Field
import cvxopt
import functions as ft


class GridComputer:
    """
    Class responsible for the fluid-dynamic ca  lculations of the crowd.
    Main tasks:
    1. Interpolating velocity on grid points
    2. Computing pressure
    3. Adapting global velocity field with pressure gradient
    4. Adapting individual velocity by global velocity field.
    """

    def __init__(self, scene, show_plot, apply_interpolation, apply_pressure, config):
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
        prop_dx = config['general'].getfloat('cell_size_x')
        prop_dy = config['general'].getfloat('cell_size_y')
        self.grid_dimension = (self.scene.size.array / (prop_dx, prop_dy)).astype(int)
        self.dx, self.dy = self.scene.size.array / self.grid_dimension
        self.dt = self.scene.dt
        self.packing_factor = config['dynamic'].getfloat('packing_factor')
        self.min_distance = config['general'].getfloat('minimal_distance')
        self.max_density = 2 * self.packing_factor / \
                           (np.sqrt(3) * self.min_distance ** 2)
        cvxopt.solvers.options['show_progress'] = False
        self.basis_A = self.basis_v_x = self.basis_v_y = None
        self._last_solution = None
        self._create_base_matrices()

        self.show_plot = show_plot
        self.apply_interpolation = apply_interpolation or apply_pressure
        self.apply_pressure = apply_pressure
        dx, dy = self.dx, self.dy
        shape = self.grid_dimension
        self.density_field = Field(shape, Field.Orientation.center, 'density', (dx, dy))
        self.v_x = Field(shape, Field.Orientation.center, 'velocity_x', (dx, dy))
        self.v_y = Field(shape, Field.Orientation.center, 'velocity_y', (dx, dy))
        self.pressure_field = Field((shape[0] + 2, shape[1] + 2), Field.Orientation.center, 'pressure', (dx, dy))
        # If beneficial, we could employ a staggered grid

        if self.show_plot:
            # Plotting hooks
            f, self.graphs = plt.subplots(2, 2)
            plt.show(block=False)

    def _create_base_matrices(self):
        """
        Creates the matrix for solving the LCP using kronecker sums and the finite difference scheme
        The matrix returned is sparse and in cvxopt format.
        :return: \Delta t/\Delta x^2*'ones'. Matrices are ready apart from multiplication with density.
        """
        nx = self.grid_dimension[0] - 2
        ny = self.grid_dimension[1] - 2
        ex = np.ones(nx)
        ey = np.ones(ny)
        Adxx = np.diag(ex[:-1], 1) + np.diag(-ex[:-1], -1)
        Adyy = np.diag(ey[:-1], 1) + np.diag(-ey[:-1], -1)
        self.Ax = 1 / (4 * self.dx ** 2) * np.kron(Adxx, np.eye(ny))
        self.Ay = 1 / (4 * self.dy ** 2) * np.kron(np.eye(nx), Adyy)

        Bdxx = np.diag(ex[:-1], 1) + np.diag(-2 * ex) + np.diag(ex[:-1], -1)
        Bdyy = np.diag(ey[:-1], 1) + np.diag(-2 * ey) + np.diag(ey[:-1], -1)
        self.Bx = 1 / (self.dx ** 2) * np.kron(Bdxx, np.eye(ny))
        self.By = 1 / (self.dy ** 2) * np.kron(np.eye(nx), Bdyy)

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
        """
        We solve min {1/2x^TAx+x^Tq}.
        First we cut off the boundary of the 2D fields.
        Then we convert the 2D fields to vectors.
        Then we construct the matrices from the base matrices made on initialization.
        These matrices can be validated from the theory in the report
        Then we convert the matrix system to a quadratic program and throw it into cvxopt solver.
        Finally we (hopefully) find a pressure and reconvert it to a 2D store.
        We have to anticipate the case of a singular matrix
        :return:
        """
        nx = self.grid_dimension[0] - 2
        ny = self.grid_dimension[1] - 2
        flat_rho = self.density_field.without_boundary().flatten(order='F') + 0.1  # Solve when analysed the effects
        diff_rho_x = (self.density_field.with_offset('right', 2) - self.density_field.with_offset('left', 2))[:, 1:-1]
        diff_rho_y = (self.density_field.with_offset('up', 2) - self.density_field.with_offset('down', 2))[1:-1, :]
        A = (self.Ax * diff_rho_x.flatten(order='F') + self.Ay * diff_rho_y.flatten(order='F'))
        B = (self.Bx + self.By) * flat_rho
        C = A + B

        diff_v_rho_x = (self.density_field.with_offset('right', 2) * self.v_x.with_offset('right', 2) \
                        - self.density_field.with_offset('left', 2) * self.v_x.with_offset('left', 2))[:, 1:-1]

        diff_v_rho_y = (self.density_field.with_offset('up', 2) * self.v_y.with_offset('up', 2) \
                        - self.density_field.with_offset('down', 2) * self.v_y.with_offset('down', 2))[1:-1, :]

        b = self.max_density - flat_rho + (diff_v_rho_x.flatten(order='F') + diff_v_rho_y.flatten(order='F')) * self.dt
        flat_p = solve_LCP_with_pgs(-C * self.dt, b, self._last_solution)
        self._last_solution = np.reshape(flat_p, (nx * ny, 1))
        dim_p = np.reshape(flat_p, (nx, ny), order='F')

        self.pressure_field.update(np.pad(dim_p, (2, 2), 'constant', constant_values=1))

    @staticmethod
    def solve_LCP_with_quad(M, q):
        """
        Solves the linear complementarity problem w = Mz + q using a quadratic solver.
        This method is unused as we employ a Cython PGS-solver.
        :param M: nxn non-singular positive definite matrix
        :param q: length n vector
        :return: length n vector z such that z>=0, w>=0, (w,z) < eps if optimum is found, else zeros vector.
        """
        n = M.shape[0]
        cvx_P = cvxopt.matrix(2 * M, tc='d')
        cvx_q = cvxopt.matrix(q, tc='d')
        I = np.eye(n)
        O = np.zeros([n, 1])
        cvx_G = cvxopt.matrix(np.vstack([-M, -I]), tc='d')
        cvx_h = cvxopt.matrix(np.vstack([q[:, None], O]), tc='d')

        try:
            result = cvxopt.solvers.qp(P=cvx_P, q=cvx_q, G=cvx_G, h=cvx_h)
            if result['status'] == 'optimal':
                z = result['x']
                return z
            else:
                ft.warn("No optimal result found in LCP solver")
        except ValueError as e:
            ft.warn("CVXOPT Error: " + str(e))
        return O

    @staticmethod
    def solve_LCP_with_pgs(M, q, init_guess=None):
        """
        Solves the linear complementarity problem w = Mz + q using a Projected Gauss Seidel solver.
        Possible improvements:
            -Sparse matrix use
        :param M: nxn non-singular positive definite matrix
        :param q: length n vector
        :return: length n vector z such that z>=0, w>=0, (w,z)\approx 0 if optimum is found, else zeros vector.
        """
        eps = 1e-02
        max_it = 10000
        n = len(q)
        q = q[:, None]
        O = np.zeros([n, 1])
        if init_guess is not None:
            z = init_guess
        else:
            z = np.ones([n, 1])
        w = np.dot(M, z) + q
        it = 0
        while (np.abs(np.dot(w.T, z)) > eps or np.any(z < -eps) or np.any(w < -eps)) and it < max_it:
            it += 1
            for i in range(n):
                r = -q[i] - np.dot(M[i, :], z) + M[i, i] * z[i]
                z[i] = max(0, r / M[i, i])
            w = np.dot(M, z) + q
        ft.debug("Iterations: %d" % it)
        if it == max_it:
            ft.warn("Max iterations reached, no optimal result found")
            return O
        else:
            return z

    def adjust_velocity(self):
        """
        Adjusts the velocity field for the pressure gradient. We pad the pressure with an extra boundary
        so that the gradient is defined everywhere.
        :return: None
        """
        # Not using the update method.
        well_shaped_x_grad = self.pressure_field.gradient('x')[:, 1:-1]
        well_shaped_y_grad = self.pressure_field.gradient('y')[1:-1, :]
        self.v_x.array -= well_shaped_x_grad
        self.v_y.array -= well_shaped_y_grad

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
        self.scene.velocity_array = self.scene.velocity_array \
                                    + local_dens[:, None] / self.max_density * (
            solved_velocity - self.scene.velocity_array)
        self.scene.velocity_array /= \
            np.linalg.norm(self.scene.velocity_array, axis=1)[:, None] / self.scene.max_speed_array[:, None]

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
            density_field, v_x, v_y = compute_density_and_velocity_field(self.grid_dimension,
                                                                         self.scene.size.array,
                                                                         self.scene.position_array,
                                                                         self.scene.velocity_array,
                                                                         self.scene.active_entries)
            self.density_field.update(density_field)
            self.v_x.update(v_x)
            self.v_y.update(v_y)
            ft.debug(self.max_density)
            ft.debug(self.density_field)
            if self.show_plot:
                self.plot_grid_values()
        if self.apply_interpolation:
            if self.apply_pressure:
                self.compute_pressure()
                self.adjust_velocity()
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
