__author__ = 'omar'
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline as Rbs
import cvxopt
from functions import *


class GridComputer:
    """
    Class responsible for the fluid-dynamic calculations of the crowd.
    Main tasks:
    1. Interpolating velocity on grid points
    2. Computing pressure
    3. Adapting global velocity field with pressure gradient
    4. Adapting individual velocity by global velocity field.
    """

    def __init__(self, scene, show_plot, apply_interpolation, apply_pressure):
        self.scene = scene
        self.cell_dimension = self.scene.number_of_cells
        self.dx, self.dy = self.scene.cell_size
        self.dt = self.scene.dt
        self.interpolation_factor = 3
        self.packing_factor = 0.9
        self.max_density = 2 * 2 * self.packing_factor / (np.sqrt(3) * self.scene.minimal_distance ** 2)  # Needs size.
        cvxopt.solvers.options['show_progress'] = False
        self.basis_A = self.basis_v_x = self.basis_v_y = None
        self._create_matrices()

        self.show_plot = show_plot
        self.apply_interpolation = apply_interpolation
        self.apply_pressure = apply_pressure

        # If beneficial, we could employ a staggered grid
        self.rho = np.zeros(self.cell_dimension)
        self.v_x = np.zeros(self.cell_dimension)
        self.v_y = np.zeros(self.cell_dimension)
        self.p = np.zeros(self.cell_dimension)
        self.grad_p_x = np.zeros(self.cell_dimension)
        self.grad_p_y = np.zeros(self.cell_dimension)

        self.x_range = np.linspace(0, self.scene.size.width, self.cell_dimension[0])
        self.y_range = np.linspace(0, self.scene.size.height, self.cell_dimension[1])

        if self.show_plot:
            # Plotting hooks
            self.mesh_x, self.mesh_y = np.meshgrid(self.x_range, self.y_range, indexing='ij')
            f, self.graphs = plt.subplots(2, 2)
            plt.show(block=False)

    def _create_matrices(self):
        """
        Creates the matrix for solving the LCP using kronecker sums and the finite difference scheme
        The matrix returned is sparse and in cvxopt format.
        :return: \Delta t/\Delta x^2*'ones'. Matrices are ready apart from multiplication with density.
        """
        nx, ny = self.cell_dimension
        ex = np.ones(nx)
        ey = np.ones(ny)
        Adxx = np.diag(ex[:-1], 1) + np.diag(-ex[:-1], -1)
        Adyy = np.diag(ey[:-1], 1) + np.diag(-ey[:-1], -1)
        self.Ax = 1 / (4 * self.dx ** 2) * np.kron(Adxx, np.eye(len(ey)))
        self.Ay = 1 / (4 * self.dy ** 2) * np.kron(np.eye(len(ex)), Adyy)

        Bdxx = np.diag(ex[:-1], 1) + np.diag(-2 * ex) + np.diag(ex[:-1], -1)
        Bdyy = np.diag(ey[:-1], 1) + np.diag(-2 * ey) + np.diag(ey[:-1], -1)
        self.Bx = 1 / (self.dx ** 2) * np.kron(Bdxx, np.eye(len(ey)))
        self.By = 1 / (self.dy ** 2) * np.kron(np.eye(len(ex)), Bdyy)

    def get_grid_values(self):
        """
        Interpolates the grid values for the discrete density and velocity fields
        Interpolation happens per grid cell, by indexing all surrounding grid cells and summing
        over the density of each pedestrian.
        Pedestrian density is computed by convolving the mass with a gaussian kernel approximation
        :return: None
        """
        cell_dict = self.scene.cell_dict
        for cell_location in cell_dict:
            relevant_pedestrian_set = set()
            cell = cell_dict[cell_location]
            cell_row, cell_col = cell_location
            for i in range(-1, 2):
                for j in range(-1, 2):
                    neighbour_cell_location = (cell_row + i, cell_col + j)
                    if neighbour_cell_location in cell_dict:
                        # Valid neighbour cell
                        relevant_pedestrian_set |= cell_dict[neighbour_cell_location].pedestrian_set
            distance_array = np.linalg.norm(self.scene.position_array - cell.center, axis=1)
            weights = GridComputer.weight_function(distance_array / self.interpolation_factor) * self.scene.alive_array
            density = np.sum(weights) + 0.01
            self.rho[cell_location] = density

            vel_array = self.scene.velocity_array * weights[:, None]
            self.v_x[cell_location] = np.sum(vel_array[:, 0]) / density
            self.v_y[cell_location] = np.sum(vel_array[:, 1]) / density
        print(self.max_density)
        print(self.orientation_correct_str(self.rho, True))
        if self.show_plot:
            self.plot_grid_values()

    def plot_grid_values(self):
        """
        Plot the density, velocity field, pressure, and pressure gradient.
        Plots are opened in a separate window and automatically updated.
        :return:
        """
        for graph in self.graphs.flatten():
            graph.cla()
        self.graphs[0, 0].imshow(np.rot90(self.rho))
        self.graphs[0, 0].set_title('Density')
        self.graphs[1, 0].imshow(np.rot90(self.p))
        self.graphs[1, 0].set_title('Pressure')
        self.graphs[0, 1].quiver(self.mesh_x, self.mesh_y, self.v_x, self.v_y, scale=1, scale_units='xy')
        self.graphs[0, 1].set_title('Velocity field')
        self.graphs[1, 1].quiver(self.mesh_x, self.mesh_y, self.grad_p_x, self.grad_p_y, scale=1, scale_units='xy')
        self.graphs[1, 1].set_title('Pressure gradient')
        plt.show(block=False)

    def solve_LCP(self):
        """
        We solve min {1/2x^TAx+x^Tq}.
        First we convert the 2D fields to vectors.
        Then we construct the matrices from the base matrices made on initialization.
        These matrices can be validated from the theory in the report
        Then we convert the matrix system to a quadratic program and throw it into cvxopt solver.
        Finally we (hopefully) find a pressure and reconvert it to a 2D store.
        We have to anticipate the case of a singular matrix
        :return:
        """
        nx = self.cell_dimension[0]
        ny = self.cell_dimension[1]
        flat_rho = self.rho.flatten(order='F') + 0.1
        diff_rho_x = GridComputer.get_dir_difference(self.rho, 'x')
        diff_rho_y = GridComputer.get_dir_difference(self.rho, 'y')
        A = (self.Ax * diff_rho_x.flatten(order='F') + self.Ay * diff_rho_y.flatten(order='F'))
        B = (self.Bx + self.By) * flat_rho
        C = A + B
        cvx_M = cvxopt.matrix(-C * self.dt)

        diff_v_rho_x = GridComputer.get_dir_difference(self.rho * self.v_x, 'x')
        diff_v_rho_y = GridComputer.get_dir_difference(self.rho * self.v_y, 'y')

        b = self.max_density - flat_rho + (diff_v_rho_x.flatten(order='F') + diff_v_rho_y.flatten(order='F')) * self.dt
        cvx_b = cvxopt.matrix(b)
        I = np.eye(nx * ny)
        cvx_G = cvxopt.matrix(np.vstack((C * self.dt, -I)))
        zeros = np.zeros([nx * ny, 1])
        cvx_h = cvxopt.matrix(np.vstack((b[:, None], zeros)))
        try:
            result = cvxopt.solvers.qp(P=cvx_M, q=cvx_b, G=cvx_G, h=cvx_h)
            flat_p = result['x']
        except ValueError as e:
            warn("CVXOPT Error: " + str(e))
            flat_p = np.zeros([1, nx * ny])
        self.p = np.reshape(flat_p, self.cell_dimension, order='F')


    def adjust_velocity(self):
        """
        Adjusts the velocity field for the pressure gradient
        :return: None
        """
        self.grad_p_x = GridComputer.get_dir_difference(self.p, 'x') / (2 * self.dx)
        self.grad_p_y = GridComputer.get_dir_difference(self.p, 'y') / (2 * self.dx)
        self.v_x -= self.grad_p_x
        self.v_y -= self.grad_p_y

    def interpolate_pedestrians(self):
        """
        Method that reconverts the velocity field to individual pedestrian positions.
        Input are the velocity x/y fields. We use a bivariate spline interpolation method
        to interpolate the velocities at the pedestrians position.
        These velocities are then weighed to the densities
        and added to the velocity field.
        :return: None
        """
        v_x_func = Rbs(self.x_range, self.y_range, self.v_x)  # Todo:Recheck
        v_y_func = Rbs(self.x_range, self.y_range, self.v_y)
        dens_func = Rbs(self.x_range, self.y_range, self.rho)
        solved_v_x = v_x_func.ev(self.scene.position_array[:, 0], self.scene.position_array[:, 1])
        solved_v_y = v_y_func.ev(self.scene.position_array[:, 0], self.scene.position_array[:, 1])
        local_dens = np.minimum(dens_func.ev(self.scene.position_array[:, 0], self.scene.position_array[:, 1]),
                                self.max_density)
        solved_velocity = np.hstack((solved_v_x[:, None], solved_v_y[:, None]))
        self.scene.velocity_array = self.scene.velocity_array \
                                    + local_dens[:, None] / self.max_density * (
        solved_velocity - self.scene.velocity_array)
        self.scene.velocity_array /= np.linalg.norm(self.scene.velocity_array, axis=1)[:, None] / 2
        # todo: should be max vel

    def step(self):
        """
        Performs one time step for the fluid simulator.
        Computes and interpolates the grid fields (density and velocity)
        Solves the LCP to compute the pressure field
        Adjusts the velocity field according to the pressure
        Adjusts the velocities of the pedestrians according to the velocity field
        :return: None
        """
        self.get_grid_values()
        if self.apply_interpolation:
            if self.apply_pressure:
                self.solve_LCP()
                self.adjust_velocity()
            self.interpolate_pedestrians()

    @staticmethod
    def weight_function(array):
        """
        Using the Wendland kernel to determine the interpolation weight
        Calculation is performed in two steps to take advantage of numpy's speed
        :param array: Array of distances to apply the kernel on.
        :return: Weights of interpolation
        """
        norm_constant = 7. / (4 * np.pi)
        first_factor = np.maximum(1 - array / 2, 0)
        weight = first_factor ** 4 * (1 + 2 * array)
        return weight * norm_constant

    @staticmethod
    def orientation_correct_str(field, full=False):
        """
        Prints the field, but places (1,1) in the lower left corner
        and (m,n) in the upper right corner), column major indexing
        :param field: 2d array to be printed
        :return: string with correct field formatting
        """
        correct_repr = np.rot90(field)
        if not full:
            return str(correct_repr)
        else:
            field_repr = ""
            for row in correct_repr:
                field_repr += " [%s]\n" % "\t".join(["%4.2f" % val for val in row])
            return "[%s]" % field_repr[1:-1]

    @staticmethod
    def get_dir_difference(field, axis):
        """
        Computes a gradient component of the discrete 2D vector field.
        The vector field contains values of the cell centers
        We use a simple 2-point second order central difference scheme.
        :param field: vector field with a Carthesian indexing
        :param axis: 'x' or 'y'
        :return: vector field representing gradient component.
        """
        assert all(dim > 2 for dim in field.shape)
        if axis == 'x':
            grad_field_x = np.zeros(field.shape)
            grad_field_x[1:-1, :] = field[2:, :] - field[:-2, :]
            return grad_field_x
        elif axis == 'y':
            grad_field_y = np.zeros(field.shape)
            grad_field_y[:, 1:-1] = field[:, 2:] - field[:, :-2]
            return grad_field_y
        else:
            raise ValueError('Choose x or y for direction, not %s' % axis)
