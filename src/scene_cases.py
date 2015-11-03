__author__ = 'omar'

import numpy as np

from geometry import Point
from scene import Scene, Pedestrian


class ImpulseScene(Scene):
    def __init__(self, size, initial_pedestrian_number, obstacle_file,
                 impulse_location, impulse_size, mde=True, cache='read'):
        """
        Initializes an impulse scene
        :param size: Size object holding the size values of the scene
        :param initial_pedestrian_number: Number of pedestrians on initialization in the scene
        :param obstacle_file: name of the file containing the obstacles.
        :param impulse_location: Center of the pedestrians impulse
        :param impulse_size: (Ellipse-like) radius of the impulse
        :param mde: Boolean flag for enforcing minimal distance between pedestrians.
        :param cache: 'read' or 'write' the cell cache to increase init speed.
        :return: scene instance.
        """
        impulse_location = Point(size * impulse_location)
        if not all(impulse_location < size.array):
            raise ValueError("Impulse location not in scene")
        self.impulse_size = impulse_size
        self.impulse_location = impulse_location
        super().__init__(size, initial_pedestrian_number, obstacle_file, mde, cache)

    def _init_pedestrians(self, initial_pedestrian_number):
        center = np.array(self.impulse_location)
        self.pedestrian_list = []
        for counter in range(initial_pedestrian_number):
            ped_loc = None
            while not ped_loc:
                x, y = (np.random.rand(2) * 2 - 1) * self.impulse_size
                ped_loc = Point(center + np.array([x, y]))
                if x ** 2 + y ** 2 > self.impulse_size or not self.is_accessible(ped_loc):
                    ped_loc = None
            self.pedestrian_list.append(Pedestrian(self, counter, self.exit_list,
                                                   position=ped_loc, size=self.pedestrian_size,
                                                   max_speed=self.max_speed_array[counter]))

        self._fill_cells()


class TwoImpulseScene(Scene):
    def __init__(self, size, initial_pedestrian_number, obstacle_file,
                 impulse_locations, impulse_size, mde=True, cache='read'):
        """
        Initializes an impulse scene
        :param size: Size object holding the size values of the scene
        :param initial_pedestrian_number: Number of pedestrians on initialization in the scene
        :param obstacle_file: name of the file containing the obstacles.
        :param impulse_locations: Center of the pedestrians impulses
        :param impulse_size: (Ellipse-like) radius of the impulses
        :param mde: Boolean flag for enforcing minimal distance between pedestrians.
        :param cache: 'read' or 'write' the cell cache to increase init speed.
        :return: scene instance.
        """
        impulse_locations = [Point(size * imp_loc) for imp_loc in impulse_locations]
        if not np.all([loc.array < size.array for loc in impulse_locations]):
            raise ValueError("Impulse location not in scene")
        self.impulse_size = impulse_size
        self.impulse_locations = impulse_locations
        super().__init__(size, initial_pedestrian_number, obstacle_file, mde, cache)

    def _init_pedestrians(self, initial_pedestrian_number):

        self.pedestrian_list = []
        for counter in range(initial_pedestrian_number):
            if counter < initial_pedestrian_number // 2:
                center = np.array(self.impulse_locations[0])
            else:
                center = np.array(self.impulse_locations[1])
            ped_loc = None
            while not ped_loc:
                x, y = (np.random.rand(2) * 2 - 1) * self.impulse_size
                ped_loc = Point(center + np.array([x, y]))
                if x ** 2 + y ** 2 > self.impulse_size or not self.is_accessible(ped_loc):
                    ped_loc = None
            self.pedestrian_list.append(Pedestrian(self, counter, self.exit_list,
                                                   position=ped_loc, size=self.pedestrian_size,
                                                   max_speed=self.max_speed_array[counter]))

        self._fill_cells()


class LoopScene(Scene):
    def __init__(self, size, initial_pedestrian_number, obstacle_file, mde=True, cache='read'):
        """
        Initializes an impulse scene
        :param size: Size object holding the size values of the scene
        :param initial_pedestrian_number: Number of pedestrians on initialization in the scene
        :param obstacle_file: name of the file containing the obstacles.
        :param mde: Boolean flag for enforcing minimal distance between pedestrians.
        :param cache: 'read' or 'write' the cell cache to increase init speed.
        :return: scene instance.
        """

        super().__init__(size, initial_pedestrian_number, obstacle_file, mde, cache)

    def remove_pedestrian(self, pedestrian):
        new_point = Point([pedestrian.position.x, self.size.height - 1])
        pedestrian.manual_move(new_point)


class TopScene(Scene):
    def __init__(self, size, initial_pedestrian_number, obstacle_file, barrier, mde=True, cache='read'):
        """
        Initializes an impulse scene
        :param size: Size object holding the size values of the scene
        :param initial_pedestrian_number: Number of pedestrians on initialization in the scene
        :param obstacle_file: name of the file containing the obstacles.
        :param barrier: Relative y coordinate from which the pedestrians should be spawned.
        :param mde: Boolean flag for enforcing minimal distance between pedestrians.
        :param cache: 'read' or 'write' the cell cache to increase init speed.
        :return: scene instance.
        """
        if not 0 <= barrier < 1:
            raise ValueError("Barrier must be between 0 and 1")
        self.barrier = barrier
        super().__init__(size, initial_pedestrian_number, obstacle_file, mde, cache)

    def _init_pedestrians(self, initial_pedestrian_number):
        self.pedestrian_list = []
        for counter in range(initial_pedestrian_number):
            ped_loc = None
            while not ped_loc:
                ped_loc = Point(self.size.array * [np.random.rand(), 1 - np.random.rand() * (1 - self.barrier)])
                if not self.is_accessible(ped_loc, True):
                    ped_loc = None
            self.pedestrian_list.append(Pedestrian(self, counter, self.exit_list,
                                                   position=ped_loc, size=self.pedestrian_size,
                                                   max_speed=self.max_speed_array[counter]))

        self._fill_cells()
