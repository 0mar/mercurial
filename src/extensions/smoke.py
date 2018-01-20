from lib.smoke_machine import get_sparse_matrix, iterate_jacobi
import numpy as np
from math_objects.scalar_field import ScalarField as Field
import params


class Smoke:
    """
    Class for modelling the propagation of smoke as a function of fire, as well the effects on the pedestrians
    """

    def __init__(self, fire):
        """
        Creates a smoke propagator for the fire in the associated scene.

        :param fire: The source of the smoke.
        """
        self.scene = fire.scene
        self.fire = fire
        self.obstacles = self.speed_ref = self.smoke = None
        self.smoke_field = self.sparse_disc_matrix = None
        self.source = None

    def prepare(self):
        prop_dx = params.smoke_dx
        prop_dy = params.smoke_dy
        nx, ny = (self.scene.size.array / (prop_dx, prop_dy)).astype(int)
        dx, dy = self.scene.size.array / (nx, ny)

        self.obstacles = np.ones((nx + 2, ny + 2), dtype=int)
        self.obstacles[1:-1, 1:-1] = self.scene.get_obstacles(nx, ny)
        self.speed_ref = self.scene.max_speed_array
        self.smoke = np.zeros(np.prod(self.obstacles.shape))
        self.smoke_field = Field(self.scene.obstacle_coverage.shape, Field.Orientation.center, 'smoke',
                                 (dx, dy))
        self.sparse_disc_matrix = get_sparse_matrix(
            params.diffusion, params.velocity_x, params.velocity_y,
            dx, dy, params.dt, self.obstacles)
        # Ready for use per time step
        self.source = self._get_source(self.fire, nx, ny, dx, dy).flatten() * params.dt

    def _get_source(self, fire, nx, ny, dx, dy):
        """
        Compute the source function for the fire. Quite inefficient, but not a bottleneck at this stage.

        :param fire: The specific fire of the scene
        :return: Array with intensity of fire in every cell
        """
        source_function = np.zeros(self.obstacles.shape)
        for row, col in np.ndindex((nx, ny)):
            center = np.array([(row + 0.5) * dx, (col + 0.5) * dy])
            source_function[row, col] = fire.get_fire_experience(center)
        return source_function

    def step(self):
        """
        Use a central difference in space, implicit Euler in time scheme for the computing of smoke.

        :return: None
        """
        self.smoke = iterate_jacobi(*self.sparse_disc_matrix, self.source + self.smoke, self.smoke, self.obstacles)
        self.smoke_field.update(np.reshape(self.smoke, self.obstacles.shape)[1:-1, 1:-1])
        self.modify_speed_by_smoke()

    def modify_speed_by_smoke(self):
        """
        Choosing velocity parameters as given by the Japanese paper
        :return:
        """
        smoke_function = self.smoke_field.get_interpolation_function()
        smoke_on_positions = smoke_function.ev(self.scene.position_array[:, 0], self.scene.position_array[:, 1])
        velo_modifier = np.clip(smoke_on_positions / params.max_smoke_level, 0, 1)
        self.scene.max_speed_array = self.speed_ref - velo_modifier * params.minimum_speed
