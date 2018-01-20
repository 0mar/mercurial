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
        if self.cause_smoke:
            self.smoke_module = Smoke(self)
            self.smoke_module.prepare()
            self.on_step_functions.append(self.smoke_module.step)
        if self.repelling:
            self.on_step_functions.append(self._repel_pedestrians)

    def step(self):
        [step() for step in self.on_step_functions]

    def _repel_pedestrians(self):
        self.fire_experience = self.get_fire_experience(self.scene.position_array)
        self.scene.velocity_array = self.scene.velocity_array - self.fire_experience

    def get_fire_experience(self, position):
        """
        Compute the intensity for the fire for both repelling of pedestrian as for the creation of smoke
        :param position: nx2 array for which the intensity is computed
        :return: The experienced intensity of the fire as a scalar, based on the distance to the center
        """
        distance = np.sqrt((position[:, 0] - self.center[0]) ** 2 + (position[:, 1] - self.center[1]) ** 2)
        return self.intensity * np.exp(-distance / self.radius)

    def __str__(self):
        return "Fire with center: %s, radius %.2f" % (self.center, self.radius)
