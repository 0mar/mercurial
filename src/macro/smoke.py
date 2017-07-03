from math_objects.geometry import Point
from fortran_modules.smoke_machine import get_sparse_matrix, iterate_jacobi
import numpy as np


class Smoker:
    """
    Class for modelling the propagation of smoke as a function of fire, as well the effects on the pedestrians
    """
    def __init__(self, scene):
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
        self.speed_ref = self.scene.max_speed_array
        self.smoke = np.zeros(np.prod(self.obstacles.shape))
        self.smoke_2d = np.ones((self.nx, self.ny))
        self.sparse_disc_matrix = get_sparse_matrix(self.diff_coef, *self.smoke_velo, self.dx, self.dy, self.scene.dt,
                                                    self.obstacles)
        # Ready for use per time step
        self.source = self._get_source(self.scene.fire) * self.scene.dt

        self.velo_unaware_lb = 0.6
        self.velo_aware_ub = 2.5

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
        print(np.max(self.smoke))
        # self.modify_speed_by_smoke()

    def modify_speed_by_smoke(self):
        smoke_upper_bound = 3
        velo_modifier = self.smoke/smoke_upper_bound
        # Positive for all aware, negative for all unaware
        self.scene.max_speed_array = self.speed_ref + self.scene.aware_pedestrians*velo_modifier*self.velo_aware_ub
        + (self.scene.aware_pedestrians - 1)*velo_modifier*self.velo_unaware_lb

