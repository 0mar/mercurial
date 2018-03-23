import numpy as np
import params
from extensions.smoke import Smoke


class Fire:
    """
    Models a fire source in the domain. Fire sources are semipenetrable obstacles, circular.
    repelling on a large space scale.
    Walks and talks like an obstacle
    """

    def __init__(self, center, radius, scene, repelling=True, cause_smoke=True):
        """
        Fire constructor
        :param center: list/tuple/numpy array/Point: center of the fire source circle
        :param radius: float/int: radius of the actual fire
        :return: new fire source.
        """
        self.scene = scene
        self.params = None
        self.center = center
        self.radius = radius
        self.repelling = repelling
        self.cause_smoke = cause_smoke
        # intensity_level: length scale of repulsion (may turn out to depend on radius)
        self.color = 'orange'
        self.smoke_module = None
        self.on_step_functions = []

    def prepare(self, params):
        """
        Called before the simulation starts. Fix all parameters and bootstrap functions.

        :return: None
        """
        self.params = params
        if np.any(np.zeros(2) > self.center) or np.any(self.scene.size.array < self.center):
            raise ValueError("Fire coordinates %s do not lie within scene" % self.center)
        if self.cause_smoke:
            self.smoke_module = Smoke(self)
            self.smoke_module.prepare(self.params)
            self.on_step_functions.append(self.smoke_module.step)
        if self.repelling:
            self.on_step_functions.append(self._repel_pedestrians)

    def step(self):
        [step() for step in self.on_step_functions]

    def _repel_pedestrians(self):
        """
        Apply the repelling force of the fire to the pedestrians.
        Difficult factor to implement... maybe because of the assumptions on pedestrian paths,
        maybe because it is quite difficult how people react to real fire.
        :return:
        """
        return  # Todo: Fix in a proper way
        self.fire_experience = self.get_fire_repulsion_at(self.scene.position_array)
        self.scene.velocity_array = (self.scene.velocity_array - self.fire_experience) / (np.linalg.norm(
            self.scene.velocity_array - self.fire_experience, axis=1) * self.scene.max_speed_array)[:, None]

    def get_fire_intensity(self, nx, ny):
        """
        Compute the intensity of the fire for the smoke propagation and repulsion

        :param nx: Number of cells for smoke discretization in x direction
        :param ny: Number of cells for smoke discretization in y direction
        :return: The experienced intensity of the fire as a scalar, based on the distance to the center
        """
        dx, dy = self.scene.size.array / (nx, ny)
        grid_x = np.linspace(dy / 2, self.scene.size[1] - dy / 2, ny)
        grid_y = np.linspace(dx / 2, self.scene.size[0] - dx / 2, nx)
        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
        distance = np.sqrt((mesh_x - self.center[1]) ** 2 + (mesh_y - self.center[0]) ** 2)
        return np.exp(-distance / self.radius)

    def get_fire_intensity_at(self, position):
        """
        Get the heat release/temperature the pedestrians experience

        :param position: nx2 position array
        :return: Fire intensity, length n array
        """
        distance = np.sqrt((position[:, 0] - self.center[0]) ** 2 + (position[:, 1] - self.center[1]) ** 2)
        return np.exp(-distance / self.radius)

    def get_fire_repulsion_at(self, position):
        """
        Get the force the fire pushes the pedestrians towards

        :param position: Positions
        :return: Fire repulsive force, length[n-1] array
        """
        xs = self.params.fire_intensity * np.exp(-(position[:, 0] - self.center[0] / self.radius))[:, None]
        ys = self.params.fire_intensity * np.exp(-(position[:, 1] - self.center[1] / self.radius))[:, None]
        return np.hstack([xs, ys])

    def __str__(self):
        return "Fire with center: %s, radius %.2f" % (self.center, self.radius)
