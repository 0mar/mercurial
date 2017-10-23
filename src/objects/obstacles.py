import math

import numpy as np

from math_objects import functions as ft
from math_objects.geometry import Point, Size, Interval


class Obstacle:
    """
    {Obstacle} models an rectangular obstacle within the domain. The obstacle has a starting point, a size,
    and a permeability factor.
    """

    def __init__(self, begin, size, name, margin, accessible=False, cell_size=None):
        """
        Constructor for the obstacle.
        :param begin: Point object with lower-left values of object
        :param size: Size object with size values of object
        :param name: name (id) for object
        :param margin: Safety margin around obstacle
        :param accessible: whether pedestrians are able to go through this object
        :return: object instance.
        """
        try:
            new_begin = np.floor(begin.array / cell_size) * cell_size
            new_end = np.ceil((begin.array + size.array) / cell_size) * cell_size
            self.begin = Point(new_begin)
            self.size = Point(new_end - new_begin)
        except TypeError:
            self.begin = begin
            self.size = size
        self.end = self.begin + self.size
        self.name = name
        self.accessible = accessible
        self.color = 'black'
        self.corner_list = [Point(self.begin + Size([x, y]) * self.size) for x in range(2) for y in range(2)]
        # Safety margin for around the obstacle corners.
        self.margin_list = [Point(margin * np.sign([x - 0.5, y - 0.5])) for x in range(2) for y in range(2)]
        self.in_interior = True
        self.center = self.begin + self.size * 0.5

    def __contains__(self, coord: Point):
        """
        Check whether a point lies in the obstacle.
        Allows for ` if point in obstacle:` constructions.
        :param coord: Point under consideration
        :return: True if point in obstacle, false otherwise
        """

        return all([self.begin[dim] <= coord[dim] <= self.begin[dim] + self.size[dim] for dim in range(2)])

    def __getitem__(self, index):
        """
        Index the obstacles as a list of two points.
        Allows for `for point in obstacle:` obstructions
        :param index: index, must be 0 or 1
        :return: begin if 0, end if 1
        """
        return [self.begin, self.end][index]

    def __repr__(self):
        """
        Unique identifier
        :return: String identifier
        """
        return "%s#%s" % (self.__class__.__name__, self.name)

    def __str__(self):
        """
        Readable representation
        :return: String representation
        """
        return "Obstacle %s. Bottom left: %s, Top right: %s" % (self.name, self.begin, self.end)


class Entrance(Obstacle):
    """
    Models an entrance.
    """

    def __init__(self, spawn_rate=0.3, max_pedestrians=8000, start_time=0, exit_data=[], *args, **kwargs):
        """
        Creates a new entrance with specified parameters.
        :param spawn_rate: The number of new pedestrians per second
        :param max_pedestrians: The total number of pedestrians entering from this entrance
        :param start_time: Number of seconds from t=0 until the entrance activates
        :param exit_data: Pedestrian exit times gathered from an earlier simulation.
        If given, pedestrians enter this simulation at the specified times, allowing for a simulation chain.
        :param args: Obstacle parameters
        :param kwargs: Obstacle parameters
        """
        super(Entrance, self).__init__(accessible=False, *args, **kwargs)
        self.spawn_rate = spawn_rate
        self.spawned_pedestrian_number = 0
        self.depleted = False
        self.start_time = start_time
        self.exit_data = exit_data
        if self.exit_data:
            self.full_data = np.zeros([0, 2])
            self._init_exit_data()
            self.max_pedestrians = len(self.full_data)
            ft.debug("Have %d max pedestrians" % self.max_pedestrians)
        else:
            self.max_pedestrians = max_pedestrians
            self.people_queue = np.random.poisson(spawn_rate, 2000)
            self.queue_iterator = 0

        self.angle_interval = Interval([0, 2 * math.pi])
        self.color = 'green'

    def _init_exit_data(self):
        """
        Converting the pedestrian exit data to the entrance data
        """
        full_data = np.zeros([0, 2])
        for exit_array in self.exit_data:
            if exit_array.size:
                full_data = np.vstack((full_data, exit_array))
        if not full_data.size:
            ft.warn("No exit data specified. Polling randomly with rate %.2f" % self.spawn_rate)
            self.exit_data = []
        else:
            self.full_data = full_data[full_data[:, 1].argsort()].tolist()

    def convert_angle_to_vector(self, angle):
        """
        Converts an {angle} into a vector on the boundary of the rectangle.
        Uses some basic linear algebra; extended on in report.
        """
        # Possible: Vectorize for exit_log implementation
        object_radius = ft.norm(self.size[0], self.size[1]) / 2
        x = math.cos(angle) * object_radius + ft.EPS
        abs_x = math.fabs(x)
        y = math.sin(angle) * object_radius + ft.EPS
        abs_y = math.fabs(y)
        if abs_x > self.size[0] / 2:
            correction = self.size[0] / (2 * abs_x) + 1e-2
        else:
            correction = self.size[1] / (2 * abs_y) + 1e-2
        return np.array([x, y]) * correction

    def get_new_number_of_pedestrians(self, time):
        """
        Poll for a new number of pedestrians entering from this entrance during {time}
        """
        if self.depleted or time < self.start_time:
            return 0
        total_number = 0
        if self.exit_data:
            poll_successful = True
            while poll_successful:
                if len(self.full_data) > 0:
                    poll_successful = ft.is_close(time, self.full_data[0][1])
                    if poll_successful:
                        self.full_data.pop(0)
                        total_number += 1
                else:
                    poll_successful = False
        else:
            total_number = self.people_queue[self.queue_iterator]
            self.queue_iterator = (self.queue_iterator + 1) % len(self.people_queue)
        self.spawned_pedestrian_number += total_number

        self.depleted = self.spawned_pedestrian_number >= self.max_pedestrians
        return total_number

    def get_spawn_location(self):
        """
        :return Position if time for new pedestrian, zero otherwise
        """
        angle = self.angle_interval.random()
        # This angle should be coming from the data, not implemented.
        boundary_vector = self.convert_angle_to_vector(angle)
        new_position = Point(boundary_vector + self.center)
        return new_position

    def __str__(self):
        return "Entrance %s. Bottom left: %s, Top right: %s" % (self.name, self.begin, self.end)


class Exit(Obstacle):
    """
    Model an exit obstacle. This is, unlike other obstacles, accessible and has no repulsive effect.
    """

    def __init__(self, cap=0, *args, **kwargs):
        """
        Create a new exit with specified parameters
        :param cap: A limit to the number of pedestrians that can exit each second.
        If {cap}=0, there is no limit
        :param args: Obstacle parameters
        :param kwargs: Obstacle parameters
        """
        super(Exit, self).__init__(accessible=True, *args, **kwargs)
        self.color = 'red'
        self.in_interior = False
        self.margin_list = [Point(np.zeros(2)) for _ in range(4)]
        self.log_list = []
        self.cap = round(cap)

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

    def __str__(self):
        return "Exit %s. Bottom left: %s, Top right: %s" % (self.name, self.begin, self.end)
