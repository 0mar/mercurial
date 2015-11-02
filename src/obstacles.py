__author__ = 'omar'

import math
import random

import numpy as np

from geometry import Point, Size, Interval
import functions as ft


class Obstacle:
    """
    Models an rectangular obstacle within the domain. The obstacle has a starting point, a size,
    and a permeability factor.
    """

    def __init__(self, begin: Point, size: Size, name: str, permeable=False):
        """
        Constructor for the obstacle.
        :param begin: Point object with lower-left values of object
        :param size: Size object with size values of object
        :param name: name (id) for object
        :param permeable: whether pedestrians are able to go through this object
        :return: object instance.
        """
        self.begin = begin
        self.size = size
        self.end = self.begin + self.size
        self.name = name
        self.permeable = permeable
        self.color = 'black'
        self.corner_list = [Point(self.begin + Size([x, y]) * self.size) for x in range(2) for y in range(2)]
        # Safety margin for around the obstacle corners.
        self.margin_list = [Point(np.sign([x - 0.5, y - 0.5])) for x in range(2) for y in range(2)]
        self.in_interior = True
        self.center = self.begin + self.size * 0.5

    def __contains__(self, coord: Point):
        """
        Check whether a point lies in the obstacle.
        :param coord: Point under consideration
        :return: True if point in obstacle, false otherwise
        """

        return all([self.begin[dim] <= coord[dim] <= self.begin[dim] + self.size[dim] for dim in range(2)])

    def __getitem__(self, item):
        return [self.begin, self.end][item]

    def __repr__(self):
        return "%s#%s" % (self.__class__.__name__, self.name)

    def __str__(self):
        return "Obstacle %s. Bottom left: %s, Top right: %s" % (self.name, self.begin, self.end)


class Entrance(Obstacle):
    """
    Models an entrance.
    """

    def __init__(self, begin, size, name, spawn_rate=10, exit_data=None):
        super().__init__(begin, size, name, permeable=False)
        self.spawn_rate = spawn_rate
        ft.debug("begin:%s,end:%s" % (begin, self.end))
        self.exit_data = None
        self.angle_interval = Interval([0, 2 * math.pi])
        self.color = 'green'

    def convert_angle_to_vector(self, angle):
        """
        Converts an angle into a vector on the boundary of the cube.
        Uses some basic linear algebra
        """
        # Todo: Vectorize
        object_radius = math.sqrt(self.size[0] ** 2 + self.size[1] ** 2) / 2
        x = math.cos(angle) * object_radius
        abs_x = math.fabs(x)
        y = math.sin(angle) * object_radius
        abs_y = math.fabs(y)
        assert bool(abs_x > self.size[0] / 2) ^ bool(
            abs_y > self.size[1] / 2)  # Fails on equality, but how probable is that?
        if abs_x > self.size[0] / 2:
            correction = self.size[0] / (2 * abs_x) + 1e-2
        else:
            correction = self.size[1] / (2 * abs_y) + 1e-2
        return np.array([x, y]) * correction

    def poll_for_new_pedestrian(self):
        """
        param: angle
        :return Position if time for new pedestrian, zero otherwise
        """
        if self.exit_data:
            pass
        else:
            if random.random() < 1 / self.spawn_rate:
                angle = self.angle_interval.random()
                boundary_vector = self.convert_angle_to_vector(angle)
                new_position = Point(boundary_vector + self.center)
                return new_position
            else:
                return None


class Exit(Obstacle):
    """
    Model an exit obstacle. This is, unlike other obstacles, accessible and has no dodge margin.
    """

    def __init__(self, begin, size, name):
        super(Exit, self).__init__(begin, size, name, permeable=True)
        self.color = 'red'
        self.in_interior = False
        self.margin_list = [Point(np.zeros(2)) for _ in range(4)]
        self.log_list = []

    def log_pedestrian(self, pedestrian, time):
        """
        Logs the exit location and time of the pedestrian, so we can reuse it in an Entrance
        The current log format is [angle,time].
        The current log format is positional so we can easily convert the data to a numpy array.
        :param pedestrian: Pedestrian that left the exit at time epoch time
        :param time: time at which pedestrian first entered an (this) exit.
        :return: None
        """
        distance_to_center = pedestrian.position - self.center
        angle = math.atan2(distance_to_center[1], distance_to_center[0])
        # We could process more pedestrian properties, like max speed or 'class'. We omit this for now.
        self.log_list.append([angle, time])
