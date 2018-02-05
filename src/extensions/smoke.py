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
        Creates a smoke propagator using the fire in the associated scene.

        :param fire: The source of the smoke.
        """
        self.scene = fire.scene
        self.params = None
        self.fire = fire
        self.obstacles = self.speed_ref = self.smoke = None
        self.smoke_field = self.sparse_disc_matrix = None
        self.source = None

    def prepare(self, params):
        """
        Called before the simulation starts. Fix all parameters and bootstrap functions.

        :return: None
        """
        self.params = params
        prop_dx = self.params.smoke_dx
        prop_dy = self.params.smoke_dy
        nx, ny = (self.scene.size.array / (prop_dx, prop_dy)).astype(int)
        dx, dy = self.scene.size.array / (nx, ny)

        self.obstacles = np.ones((nx + 2, ny + 2), dtype=int)
        self.obstacles[1:-1, 1:-1] = self.scene.get_obstacles(nx, ny)
        self.speed_ref = self.scene.max_speed_array
        self.params.smoke = True
        self.smoke = np.zeros(np.prod(self.obstacles.shape))
        self.smoke_field = Field((nx, ny), Field.Orientation.center, 'smoke', (dx, dy))
        # Note: This object is monkey patched in the params object.
        self.params.smoke_field = self.smoke_field
        self.sparse_disc_matrix = get_sparse_matrix(
            self.params.diffusion, self.params.velocity_x, self.params.velocity_y,
            dx, dy, self.params.dt, self.obstacles)
        # Ready for use per time step
        self.source = self._get_source(self.fire, nx + 2, ny + 2).flatten() * self.params.dt

    def _get_source(self, fire, nx, ny):
        """
        Compute the source function for the fire. Quite inefficient, but not a bottleneck at this stage.

        :param fire: The specific fire of the scene
        :return: Array with intensity of fire in every cell
        """

        source_function = fire.get_fire_intensity(nx, ny)
        return source_function

    def step(self):
        """
        Use a central difference in space, implicit Euler in time scheme for the computing of smoke.

        :return: None
        """
        self.smoke = iterate_jacobi(*self.sparse_disc_matrix, self.source + self.smoke, self.smoke, self.obstacles)
        self.smoke_field.update(np.reshape(self.smoke, self.obstacles.shape)[1:-1, 1:-1])
