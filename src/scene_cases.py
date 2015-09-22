__author__ = 'omar'

import random

import numpy as np

from geometry import Size, Point
from scene import Scene, Pedestrian
from visualization import VisualScene


class ImpulseScene(Scene):
    def __init__(self, size: Size, pedestrian_number, obstacle_file,
                 impulse_location, impulse_size, mde=True, dt=0.05, cache='read'):
        """
        Initializes an impulse scene
        :param size: Size object holding the size values of the scene
        :param pedestrian_number: Number of pedestrians on initialization in the scene
        :param obstacle_file: name of the file containing the obstacles.
        :param mde: Boolean flag for enforcing minimal distance between pedestrians.
        :param dt: update time step
        :return: scene instance.
        """
        if not all(impulse_location < size.array):
            raise ValueError("Impulse location not in scene")
        self.impulse_size = impulse_size
        self.impulse_location = impulse_location
        super().__init__(size, pedestrian_number, obstacle_file, mde, dt, cache)

    def _init_pedestrians(self):
        center = np.array(self.impulse_location)
        self.pedestrian_list = []
        for counter in range(self.pedestrian_number):
            ped_loc = None
            while not ped_loc:
                x, y = (np.random.rand(2) * 2 - 1) * self.impulse_size
                ped_loc = Point(center + np.array([x, y]))
                if x ** 2 + y ** 2 > self.impulse_size or not self.is_within_boundaries(ped_loc):
                    ped_loc = None
            self.pedestrian_list.append(Pedestrian(self, counter, self.exit_obs,
                                                   position=ped_loc, color=random.choice(VisualScene.color_list)))
        self._fill_cells()
