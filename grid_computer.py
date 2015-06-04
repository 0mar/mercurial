__author__ = 'omar'
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from functions import *


class GridComputer:
    def __init__(self, scene, show_plot=True):
        self.scene = scene
        self.cell_dimension = self.scene.number_of_cells
        self.correction_factor = 4
        self.show_plot = show_plot
        self.rho = np.zeros(self.cell_dimension)
        self.v_x = np.zeros(self.cell_dimension)
        self.v_y = np.zeros(self.cell_dimension)
        if self.show_plot:
            # Plotting hooks
            self.x_range = np.linspace(0, self.scene.size.width, self.cell_dimension[0])
            self.y_range = np.linspace(0, self.scene.size.height, self.cell_dimension[1])
            self.mesh_x, self.mesh_y = np.meshgrid(self.x_range, self.y_range, indexing='ij')
            graph1 = plt.figure()
            self.rho_graph = graph1.add_subplot(111)
            graph2 = plt.figure()
            self.v_graph = graph2.add_subplot(111)
            plt.show(block=False)

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
            weights = GridComputer.weight_function(distance_array / self.correction_factor) * self.scene.alive_array
            density = np.sum(weights)
            self.rho[cell_location] = density

            vel_array = self.scene.velocity_array * weights[:, None]
            self.v_x[cell_location] = np.sum(vel_array[:, 0])
            self.v_y[cell_location] = np.sum(vel_array[:, 1])
        if self.show_plot:
            self.plot_grid_values()

    def plot_grid_values(self):
        self.rho_graph.cla()
        self.rho_graph.imshow(np.rot90(self.rho))
        self.v_graph.cla()
        self.v_graph.quiver(self.mesh_x, self.mesh_y, self.v_x, self.v_y, scale=1, scale_units='xy')
        plt.draw()

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
