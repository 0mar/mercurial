import numpy as np

from math_objects.geometry import Point
from objects.scene import Scene, Pedestrian

"""
A list of Scene extensions where the pedestrians have a special initial distribution
"""


class ImpulseScene(Scene):
    def __init__(self, impulse_location, impulse_size, *args, **kwargs):
        """
        Initializes scene with an initial dense circular pedestrian distribution.
        :param impulse_location: Center of the pedestrians impulse
        :param impulse_size: (Ellipse-like) radius of the impulse
        :return: scene instance.
        """
        self.impulse_location = impulse_location
        self.impulse_size = impulse_size
        super().__init__(*args, **kwargs)

    def _init_pedestrians(self, initial_pedestrian_number):
        center = self.size.array * self.impulse_location
        self.pedestrian_list = []
        for counter in range(initial_pedestrian_number):
            ped_loc = None
            while not ped_loc:
                x, y = (np.random.rand(2) * 2 - 1) * self.impulse_size
                ped_loc = Point(center + np.array([x, y]))
                if x ** 2 + y ** 2 > self.impulse_size ** 2 or not self.is_accessible(ped_loc):
                    ped_loc = None
            self.pedestrian_list.append(Pedestrian(self, counter, self.exit_list, position=ped_loc))


class TwoImpulseScene(Scene):
    def __init__(self, impulse_size, impulse_locations, *args, **kwargs):
        """
        Initializes an scene with two initial circles of pedestrians
        :param impulse_locations: Center of the pedestrians impulses
        :param impulse_size: (Ellipse-like) radius of the impulses
        :return: scene instance.
        """
        self.impulse_size = impulse_size
        self.impulse_locations = impulse_locations
        super().__init__(*args, **kwargs)

    def _init_pedestrians(self, initial_pedestrian_number):

        self.pedestrian_list = []
        for counter in range(initial_pedestrian_number):
            if counter < initial_pedestrian_number // 2:
                center = np.array(self.impulse_locations[0]) * self.size.array
            else:
                center = np.array(self.impulse_locations[1]) * self.size.array
            ped_loc = None
            while not ped_loc:
                x, y = (np.random.rand(2) * 2 - 1) * self.impulse_size
                ped_loc = Point(center + np.array([x, y]))
                if x ** 2 + y ** 2 > self.impulse_size or not self.is_accessible(ped_loc):
                    ped_loc = None
            self.pedestrian_list.append(Pedestrian(self, counter, self.exit_list, position=ped_loc))


class LoopScene(Scene):
    def __init__(self, *args, **kwargs):
        """
        Initializes a scene where pedestrians that leave the scene at one side,
        return on the other side
        :return: scene instance.
        """

        super().__init__(*args, **kwargs)

    def remove_pedestrian(self, pedestrian):
        """
        Removing a pedestrian now puts him in a new position in the scene
        :param pedestrian: Unremoved pedestrian
        :return:
        """
        new_point = Point([pedestrian.position.x, self.size.height - 1])
        pedestrian.position = new_point


class TopScene(Scene):
    def __init__(self, barrier, *args, **kwargs):
        """
        Initializes a scene where initially pedestrians are only present above a certain barrier
        :param barrier: Relative y coordinate (between 0 and 1) that indicates the lowest pedestrian position
        :return: scene instance.
        """
        if not 0 <= barrier < 1:
            raise ValueError("Barrier must be between 0 and 1")
        self.barrier = barrier
        super().__init__(*args, **kwargs)

    def _init_pedestrians(self, initial_pedestrian_number):
        self.pedestrian_list = []
        for counter in range(initial_pedestrian_number):
            ped_loc = None
            while not ped_loc:
                ped_loc = Point(self.size.array * [np.random.rand(), 1 - np.random.rand() * (1 - self.barrier)])
                if not self.is_accessible(ped_loc, True):
                    ped_loc = None
            self.pedestrian_list.append(Pedestrian(self, counter, self.exit_list, position=ped_loc))
