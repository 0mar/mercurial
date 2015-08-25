__author__ = 'omar'

import random

import numpy as np

from geometry import Size, Point
from scene import Scene, Pedestrian
from visualization import VisualScene


class ImpulseScene(Scene):
    def __init__(self, size: Size, pedestrian_number, obstacle_file,
                 impulse_location, impulse_size, dt=0.05, cache='read'):
        """
        Initializes an impulse scene
        :param size: Size object holding the size values of the scene
        :param pedestrian_number: Number of pedestrians on initialization in the scene
        :param obstacle_file: name of the file containing the obstacles.
        :param dt: update time step
        :return: scene instance.
        """
        self.size = size
        if not all(impulse_location < size.array):
            raise ValueError("Impulse location not in scene")
        self.pedestrian_number = pedestrian_number
        self.dt = dt
        self.time = 0
        self.obstacle_list = []
        self._read_json_file(file_name=obstacle_file)
        self.position_array = np.zeros([self.pedestrian_number, 2])
        self.velocity_array = np.zeros([self.pedestrian_number, 2])
        self.pedestrian_cells = np.zeros([self.pedestrian_number, 2])
        self.alive_array = np.ones(self.pedestrian_number)
        self.cell_dict = {}
        self.number_of_cells = (20, 20)
        self.cell_size = Size(self.size.array / self.number_of_cells)
        if cache == 'read':
            self._load_cells()
        else:
            self._create_cells()
        if cache == 'write':
            self._store_cells()

        center = np.array(impulse_location)
        self.pedestrian_list = []
        for counter in range(self.pedestrian_number):
            ped_loc = None
            while not ped_loc:
                x, y = (np.random.rand(2) * 2 - 1) * impulse_size
                ped_loc = Point(center + np.array([x, y]))
                if x ** 2 + y ** 2 > impulse_size or not self.is_within_boundaries(ped_loc):
                    ped_loc = None
            self.pedestrian_list.append(Pedestrian(self, counter, self.exit_obs,
                                                   position=ped_loc, color=random.choice(VisualScene.color_list)))
        self._fill_cells()
        self.status = 'RUNNING'
