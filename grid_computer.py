__author__ = 'omar'
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.sparse as ss
from scipy.interpolate import RectBivariateSpline as RBS
import cvxopt

from functions import *


class GridComputer:
    def __init__(self, scene, show_plot, apply):
        self.scene = scene
        self.cell_dimension = self.scene.number_of_cells
        self.dx,self.dy = self.scene.cell_size
        self.dt = self.scene.dt
        self.interpolation_factor = 4
        self.packing_factor = 0.8
        self.minimal_distance = 1
        self.max_pressure = 2 * self.packing_factor / (np.sqrt(3) * self.minimal_distance)
        cvxopt.solvers.options['show_progress'] = False
        self.basis_A = self.basis_v_x = self.basis_v_y = None
        self._create_matrices()

        self.show_plot = show_plot
        self.apply = apply
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
        :return: \delta t/\delta x^2*'ones'. Matrices are ready apart from multiplication with density.
        """
        nx, ny = self.cell_dimension
        ex = np.ones(nx)
        ey = np.ones(ny)
        Adxx = np.diag(-ex[:-1],1)+np.diag(ex[:-1],-1)
        Adyy =np.diag(-ey[:-1],1)+np.diag(ey[:-1],-1)
        self.Ax = 1/(4*self.dx**2)*np.kron(Adxx,np.eye(len(ey)))
        self.Ay = 1/(4*self.dy**2)*np.kron(np.eye(len(ex)),Adyy)

        Bdxx = np.diag(-ex[:-1],1)+np.diag(2*ex)+np.diag(-ex[:-1],-1)
        Bdyy =np.diag(-ey[:-1],1)+np.diag(2*ey)+np.diag(-ey[:-1],-1)
        self.Bx = 1/(self.dx**2)*np.kron(Bdxx,np.eye(len(ey)))
        self.By = 1/(self.dy**2)*np.kron(np.eye(len(ex)),Bdyy)
        # v_x = ss.dia_matrix((-ex, -1), shape=self.cell_dimension) + ss.dia_matrix((ex, 1), shape=self.cell_dimension)
        # self.basis_v_x = self.dt / self.dx * ss.kron(np.eye(len(ex)), v_x)
        # v_y = ss.dia_matrix((ey, -1), shape=self.cell_dimension) + ss.dia_matrix((-ey, 1), shape=self.cell_dimension)
        # self.basis_v_y = self.dt / self.dx * ss.kron(v_y, np.eye(len(ey)))

    def get_grid_values(self):
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
            density = np.sum(weights)
            self.rho[cell_location] = density

            vel_array = self.scene.velocity_array * weights[:, None]
            self.v_x[cell_location] = np.sum(vel_array[:, 0])
            self.v_y[cell_location] = np.sum(vel_array[:, 1])
        if self.show_plot:
            self.plot_grid_values()

    def plot_grid_values(self):
        for graph in self.graphs.flatten():
            graph.cla()
        self.graphs[0, 0].imshow(np.rot90(self.rho))
        self.graphs[0, 0].set_title('Density')
        self.graphs[1, 0].imshow(np.rot90(self.p))
        self.graphs[0, 1].set_title('Pressure')
        self.graphs[0, 1].quiver(self.mesh_x, self.mesh_y, self.v_x, self.v_y, scale=1, scale_units='xy')
        self.graphs[1, 0].set_title('Velocity field')
        self.graphs[1, 1].quiver(self.mesh_x, self.mesh_y, self.grad_p_x, self.grad_p_y, scale=1, scale_units='xy')
        self.graphs[1, 1].set_title('Pressure gradient')
        plt.show(block=False)

    def solve_LCP(self):
        """
        We solve min {1/2x^TAx+x^Tq}

        :return:
        """
        nx = self.cell_dimension[0]
        ny = self.cell_dimension[1]
        flat_rho = self.rho.flatten(order='F') + 0.1
        grad_rho_x = GridComputer.get_gradient(self.rho,'x')
        grad_rho_y = GridComputer.get_gradient(self.rho,'y')
        A = self.Ax * grad_rho_x.flatten(order='F') + self.Ay * grad_rho_y.flatten(order='F')
        B = (self.Bx+self.By)*flat_rho
        cvx_M = cvxopt.matrix(A+B)

        grad_v_rho_x = GridComputer.get_gradient(self.rho*self.v_x,'x')
        grad_v_rho_y = GridComputer.get_gradient(self.rho*self.v_y,'y')

        b = self.max_pressure - self.dt*(grad_v_rho_x.flatten(order='F')+grad_v_rho_y.flatten(order='F')    )
        cvx_b = cvxopt.matrix(b)
        I = np.eye(nx * ny)
        cvx_G = cvxopt.matrix(np.vstack((A+B, -I)))
        zeros = np.zeros([nx * ny, 1])
        cvx_h = cvxopt.matrix(np.vstack((b[:, None], zeros)))
        result = cvxopt.solvers.qp(P=cvx_M, q=cvx_b, G=cvx_G, h=cvx_h)
        flat_p = result['x']
        self.p = np.reshape(flat_p, self.cell_dimension)

    def adjust_velocity(self):
        self.grad_p_x = GridComputer.get_gradient(self.p,'x')
        self.grad_p_y = GridComputer.get_gradient(self.p,'y')
        self.v_x -= self.grad_p_x
        self.v_y -= self.grad_p_y

    def interpolate_pedestrians(self):
        v_x_func = RBS(self.x_range, self.y_range, self.v_x)
        v_y_func = RBS(self.x_range, self.y_range, self.v_y)
        solved_v_x = v_x_func.ev(self.scene.position_array[:, 0], self.scene.position_array[:, 1])
        solved_v_y = v_y_func.ev(self.scene.position_array[:, 0], self.scene.position_array[:, 1])
        solved_velocity = np.hstack((solved_v_x[:, None], solved_v_y[:, None]))
        self.scene.velocity_array = (self.scene.velocity_array + solved_velocity) / 2
        self.scene.velocity_array /= np.linalg.norm(self.scene.velocity_array, axis=1)[:, None] / 5
        # todo: should be max vel

    def step(self):
        self.get_grid_values()
        if self.apply:
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
    def print_field_with_orientation(field):
        """
        Prints the field, but places (1,1) in the lower left corner
        and (m,n) in the upper right corner), column major indexing
        :param field: 2d array to be printed
        :return: string with correct field formatting
        """
        return str(np.rot90(field))

    @staticmethod
    def get_gradient(field,direction):
            assert all(dim > 2 for dim in field.shape)
            if direction=='x':
                grad_field_x = np.zeros(field.shape)
                grad_field_x[1:-1,:] = field[2:,:] - field[:-2,:]
                return grad_field_x
            elif direction=='y':
                grad_field_y = np.zeros(field.shape)
                grad_field_y[:,1:-1] = field[:,2:] - field[:,:-2]
                return grad_field_y
            else:
                raise ValueError('Choose x or y for direction, not %s'%direction)