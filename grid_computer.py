__author__ = 'omar'
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
from functions import *


class GridComputer:
    def __init__(self, scene):
        self.scene = scene
        self.cell_dimension = self.scene.number_of_cells
        self.correction_factor = 4
        self.rho = np.zeros(self.cell_dimension)
        self.v_x = np.zeros(self.cell_dimension)
        self.v_y = np.zeros(self.cell_dimension)

        # Plotting hooks
        # self.x_range = np.linspace(0,self.scene.size.width,self.cell_dimension[0])
        # self.y_range = np.linspace(0,self.scene.size.height,self.cell_dimension[1])
        # self.mesh_x,self.mesh_y = np.meshgrid(self.x_range,self.y_range,indexing='ij')
        # graph1 = plt.figure()
        # self.rho_graph = graph1.add_subplot(111)
        # graph2 = plt.figure()
        # self.v_graph = graph2.add_subplot(111)

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

    def plot_grid_values(self):
        self.rho_graph.cla()
        # We need a rotate, because we have 'ij' indexing
        self.rho_graph.imshow(np.rot90(self.rho))
        self.v_graph.cla()
        self.v_graph.quiver(self.mesh_x, self.mesh_y, self.v_x, self.v_y, scale=1, scale_units='xy')
        # plt.draw()

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

#
#
# from geometry import Size
# import scene
# from visualization import VisualScene
# from planner import GraphPlanner
#
# # Default parameters
# number_of_pedestrians = 3000
# domain_width = 70
# domain_height = 70
# obstacle_file = 'demo_obstacle_list.json'
# # Initialization
# scene = scene.Scene(size=Size([domain_width, domain_height]), obstacle_file=obstacle_file,
#                     pedestrian_number=number_of_pedestrians)
# planner = GraphPlanner(scene)
# grid = GridComputer(scene)
# plt.show(block=False)
# def step():
#     planner.collective_update()
#     grid.get_grid_values()
#     grid.plot_grid_values()
#
#
# vis = VisualScene(scene, 1500, 1000, step=step, loop=True)
# vis.loop()
# vis.window.mainloop()
