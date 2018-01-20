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
        self.center = center
        self.radius = radius
        self.repelling = repelling
        self.cause_smoke = cause_smoke
        # intensity_level: length scale of repulsion (may turn out to depend on radius)
        self.intensity = params.fire_intensity
        self.color = 'orange'
        self.smoke_module = None
        self.on_step_functions = []

    def prepare(self):
        if np.any(np.zeros(2) > self.center) or np.any(self.scene.size.array < self.center):
            raise ValueError("Fire coordinates %s do not lie within scene" % self.center)
        if self.cause_smoke:
            self.smoke_module = Smoke(self)
            self.smoke_module.prepare()
            self.on_step_functions.append(self.smoke_module.step)
        if self.repelling:
            self.on_step_functions.append(self._repel_pedestrians)

    def step(self):
        [step() for step in self.on_step_functions]

    def _repel_pedestrians(self):
        self.fire_experience = self.get_fire_repulsion(self.scene.position_array)
        self.scene.velocity_array = (self.scene.velocity_array - self.fire_experience) / (np.linalg.norm(
            self.scene.velocity_array - self.fire_experience, axis=1) * self.scene.max_speed_array)[:, None]

    def get_fire_experience(self, nx, ny):
        """
        Compute the intensity for the fire for both repelling of pedestrian as for the creation of smoke

        :param position: nx array for which the intensity is computed
        :return: The experienced intensity of the fire as a scalar, based on the distance to the center
        """
        dx, dy = self.scene.size.array / (nx, ny)
        grid_x = np.linspace(dx / 2, self.scene.size[0] - dx / 2, nx)
        grid_y = np.linspace(dy / 2, self.scene.size[1] - dy / 2, ny)
        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
        intensity = np.sqrt((mesh_x - self.center[0]) ** 2 + (mesh_y - self.center[1]) ** 2)
        return np.exp(-intensity / self.radius)

    def get_fire_repulsion(self, position):
        """
        Get the force the fire pushes the pedestrians towards

        :param position: Positions
        :return:
        """
        xs = np.exp(-(position[:, 0] - self.center[0] / self.radius))[:, None]
        ys = np.exp(-(position[:, 1] - self.center[1] / self.radius))[:, None]
        return np.hstack([xs, ys])

    def __str__(self):
        return "Fire with center: %s, radius %.2f" % (self.center, self.radius)
