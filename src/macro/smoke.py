from math_objects.geometry import Point
from fortran_modules.smoke_machine import get_sparse_matrix, iterate_jacobi
import numpy as np
import matplotlib.pyplot as plt


class Smoker:
    """
    Class for modelling the propagation of smoke as a function of fire, as well the effects on the pedestrians
    """

    def __init__(self, scene, show_plot=True):
        """
        Creates a smoke evolver for the given scene
        :param scene: The scene to be smoked.
        """
        self.scene = scene
        config = self.scene.config['smoke']
        self.nx = config.getint('res_x')
        self.ny = config.getint('res_y')
        if self.nx == 0 or self.ny == 0:
            # Take the same resolution as the pressure machine
            prop_dx = scene.config['general'].getfloat('cell_size_x')
            prop_dy = scene.config['general'].getfloat('cell_size_y')
            self.nx, self.ny = (self.scene.size.array / (prop_dx, prop_dy)).astype(int)
        self.dx, self.dy = self.scene.size.array / (self.nx, self.ny)
        self.diff_coef = config.getfloat('diffusion')
        self.smoke_velo = (config.getfloat('velocity_x'), config.getfloat('velocity_y'))
        obstacles_without_boundary = self.scene.get_obstacles_coverage()
        self.obstacles = np.ones((self.nx + 2, self.ny + 2), dtype=int)
        self.obstacles[1:self.nx + 1, 1:self.ny + 1] = obstacles_without_boundary
        self.smoke = np.zeros(np.prod(self.obstacles.shape))

        self.max_smoke = 1.5
        # Original maximum speed. From this we compute the new speeds
        self.ref_speed = self.scene.max_speed_array.copy()
        self.sparse_disc_matrix = get_sparse_matrix(self.diff_coef, *self.smoke_velo, self.dx, self.dy, self.scene.dt,
                                                    self.obstacles)
        # Ready for use per time step
        self.source = self._get_source(self.scene.fire).flatten(order='F') * self.scene.dt
        self.smoke_2d = np.zeros(self.obstacles.shape)
        self.show_plot = show_plot
        if self.show_plot:
            f, self.graphs = plt.subplots(2, 2)
            plt.show(block=False)

    def _get_source(self, fire):
        """
        Compute the source function for the fire. 
        :param fire: The specific fire of the scene
        :return: Array with intensity of fire in every cell
        """
        source_function = np.zeros(self.obstacles.shape)
        for row, col in np.ndindex((self.nx, self.ny)):
            center = Point([(row + 0.5) * self.dx, (col + 0.5) * self.dy])
            source_function[row, col] = fire.get_fire_intensity(center)
        return source_function

    def step(self):
        """
        Use a central difference in space, implicit Euler in time scheme for the computing of smoke.
        :return: None
        """
        self.smoke = iterate_jacobi(*self.sparse_disc_matrix, self.source + self.smoke, self.smoke, self.obstacles)
        self.smoke_2d = (1-self.obstacles)*self.smoke.reshape(self.smoke_2d.shape,order='F')
        if self.show_plot:
            self.plot_grid_values()
            # self.scene.max_speed_array = self.smoke_driven_speed()

    def smoke_driven_speed(self):
        """
        Compute an increase in speed due to smoke panic.
        We postulate that the speed increase of pedestrians is linear with the concentration of smoke.
        It cannot exceed a certain speed increment factor.
        :return: New maximum speed for pedestrians
        """
        # Linear relation between smoke an increase of speed, up to a certain point.
        velo_coef = 1.2
        self.smoke / self.max_smoke
        return np.maximum(self.smoke / self.max_smoke, 1)[:, None] * velo_coef * self.ref_speed

    def plot_grid_values(self):
        """
        Plot the density, velocity field, pressure, and pressure gradient.
        Plots are opened in a separate window and automatically updated.
        :return:
        """
        for graph in self.graphs.flatten():
            graph.cla()
        self.graphs[0, 0].imshow(np.rot90(self.smoke_2d))
        self.graphs[0, 0].set_title('Smoke')
        plt.show(block=False)
