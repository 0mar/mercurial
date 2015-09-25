#!/usr/bin/env python
import math
import random

import numpy as np

import functions as ft

__author__ = 'omar'


class Coordinate(object):
    """
    Class that models a coordinate in Carthesian space.
    Used as a base class for points, velocities and sizes.
    Implemented some basic 'magic' methods to facilitate the use of basic operators
    """

    def __init__(self, x):
        """
        :param x: iterable of coordinates. Requires a list of length 2.
        """
        self.array = np.array(x)
        self.type = self.__class__.__name__

    angle = property(lambda s: math.atan2(s[1], s[0]))

    def __len__(self):
        return len(self.array)

    def __iter__(self):
        return iter(self.array)

    def __getitem__(self, i):
        return self.array[i]

    def __add__(self, other):
        return Point(self.array + other.array)

    def __sub__(self, other):
        return Point(self.array - other.array)

    def __mul__(self, other):
        return Point(self.array * other)

    def __truediv__(self, other):
        return Point(self.array / other.array)

    def __repr__(self):
        return "%s(%s)" % (self.type, ", ".join("%.2f" % f for f in self.array))

    def is_zero(self):
        """
        Check whether coordinates are within tolerance of zero point
        :return: True if 2-norm of coordinate is smaller than epsilon
        """
        return np.linalg.norm(self.array) < ft.EPS


class Size(Coordinate):
    """
    Class that models a size. Sizes are an extension of coordinates that cannot be negative.
    """

    def __init__(self, x):
        super().__init__(x)
        self.width = x[0]
        self.height = x[1]
        if self.width < 0 or self.height < 0:
            raise ValueError("Negative size specified")

    def random_internal_point(self):
        """
        Provides a random internal point within the size
        :return: Point with positive coordinates both smaller than size
        """

        return Point(np.array([random.random() * dim for dim in self.array]))


class Point(Coordinate):
    """
    Class that models a point within a plane.
    """
    x = property(lambda s: s[0])
    y = property(lambda s: s[1])


class Velocity(Coordinate):
    """
    Class that models a velocity in the plane.
    """
    x = property(lambda s: s[0])
    y = property(lambda s: s[1])

    def rescale(self, max_speed=5.):
        if not self.is_zero():
            self.array *= max_speed / np.linalg.norm(self.array)


class Interval(object):
    """
    Class that models an interval
    """
    def __init__(self, coords):
        self.array = np.array(coords)
        self.length = coords[1] - coords[0]
        if self.length < 0:
            raise ValueError("Interval start larger than interval end")

    def random(self):
        """
        Returns a random double within the interval
        :return:
        """
        return np.random.random() * self.length + self.array[0]

    begin = property(lambda s: s[0])
    end = property(lambda s: s[1])

    def __getitem__(self, item):
        return self.array[item]

    def __contains__(self, item: float):
        return self.array[0] <= item <= self.array[1]

    def __repr__(self):
        return "Interval %s" % self.array


# Todo: Moeten ook naar Points ipv arrays toe
class LineSegment(object):
    def __init__(self, coords):
        self.array = np.array(coords)
        self.color = 'gray'

    length = property(lambda s: np.linalg.norm(s[0] - s[1]))
    begin = property(lambda s: s[0])
    end = property(lambda s: s[1])

    def get_point(self, value):
        """
        Regarding the line segment as a 1D parameter equation with domain [0,1],
         this method returns the coordinates for parameter value
        :param value: parameter in [0,1]
        :return: corresponding Point on line segment
        """
        if 0 >= value <= 1:
            raise ValueError("Value %.2f must lie between 0 and 1" % value)
        return self.begin + value * (self.end - self.begin)

    def crosses_obstacle(self, obstacle, open_sets=False):
        """
        Checks whether the line crosses the rectangular obstacle.
        Novel implementation based on hyperplanes and other linear algebra (*proud*)
        :param obstacle: The rectangular obstacle under consideration
        :param open_sets: Whether it counts as an intersection
         if the line passes inside of the obstacle (open_sets = True)
         or also counts when it passes through the obstacle boundary (open_sets = False)
        :return: True if line crosses obstacle, false otherwise
        """
        line_array = np.array([self.begin, self.end])
        rect_array = np.array([[np.min(line_array[:, 0]), np.min(line_array[:, 1])],
                               [np.max(line_array[:, 0]), np.max(line_array[:, 1])]])
        obs_array = np.array([obstacle.begin.array, obstacle.end.array])
        intersects = ft.rectangles_intersect(rect_array[0], rect_array[1], obs_array[0], obs_array[1], open_sets)
        if not intersects:
            return False
        f = ft.get_hyperplane_functional(line_array[0], line_array[1])
        obs_points = np.array([point.array for point in obstacle.corner_list])
        point_result = f(obs_points[:, 0], obs_points[:, 1])
        if np.sum(np.sign(point_result)) in [-4, 4]:
            return False
        else:
            return True

    def __getitem__(self, item):
        return self.array[item]

    def __add__(self, other):
        return LineSegment(self.array + other.array)

    def __sub__(self, other):
        return LineSegment(self.array - other.array)

    def __mul__(self, other):
        return LineSegment(self.array * other)

    def __truediv__(self, other):
        return LineSegment(self.array / other)

    def __lt__(self, other):
        if not type(self) == type(other):
            raise AttributeError("Paths must be compared to other Paths")
        return self.length < other.length

    def __repr__(self):
        return "LineSegment from %s to %s" % (Point(self.begin), Point(self.end))


class Path(object):
    """
    Wrapper around a sequence of line segments.
    """
    def __init__(self, line_segment_list):
        if not isinstance(line_segment_list, list):
            raise TypeError("Line segment list must be list")
        self.list = line_segment_list
        for i in range(len(line_segment_list) - 1):
            self.check_lines_connected(line_segment_list[i], line_segment_list[i + 1])

    @property
    def length(self):
        """
        Summed length of all line segments
        :return: sum of line segments 2-norms
        """
        return sum([line.length for line in self.list])

    def __len__(self):
        return len(self.list)

    @staticmethod
    def check_lines_connected(ls1: LineSegment, ls2: LineSegment):
        """
        Returns true if last line end connects (within epsilon accuracy) to new line begin
        :param ls1: Line segment whose end matches ls2's begin
        :param ls2: Line segment whose begin matches ls1's end
        :return: True if lines connect, False otherwise
        """
        connected = np.allclose(ls1.end, ls2.begin)
        if not connected:
            raise AssertionError("%s & %s do not connect" % (ls1, ls2))

    def __iadd__(self, other):
        """
        Appends 2 paths to each other.
        :param other: path object (which start should connect to this end)
        :return: None
        :raise: AssertionError if paths do not connect
        """
        if self.list:
            Path.check_lines_connected(self.list[-1], other[0])
            self.list += other

    def __getitem__(self, item):
        return self.list[item]

    def append(self, other):
        """
        Append line segment to this path. Path has to have at least one line segment.
        :param other: Line segment which should connect to the last line segment in this path, or Point
        :return: None
        :raise: AssertionError if path is empty, AttributeError if 'other' has wrong type.
        """
        if isinstance(other, LineSegment):
            if self.list:
                Path.check_lines_connected(self.list[-1], other)
            self.list.append(other)
        elif isinstance(other, Point):
            if not self.list:
                raise AssertionError("List must have elements before adding Points")
            print("Last element: %s\n Last point of last element: %s\n" % (self.list[-1], self.list[-1][-1]))
            new_line_segment = LineSegment([Point(self.list[-1][-1]), other])
            self.list.append(new_line_segment)
        else:
            raise AttributeError("Object appended to Path must be LineSegment or Point, not %s" % other)

    def pop_next_segment(self):
        """
        Returns the first line segment of the path and removes it from the path list.
        :return: Line segment
        """
        return self.list.pop(0)

    def __bool__(self):
        """
        :return: True if Path has any elements, False otherwise.
        """
        return bool(self.list)

    __nonzero__ = __bool__

    def __str__(self):
        """
        String representation of collected line segment
        :return: Joined string representation for each line segment
        """
        return "Path:\n%s" % "\n".join([str(ls) for ls in self.list])
