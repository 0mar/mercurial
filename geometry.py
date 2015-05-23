#!/usr/bin/env python
import math
import random

from functions import *


__author__ = 'omar'


class Coordinate(object):
    """
    Class that models a coordinate in Carthesian space.
    Used as a base class for points, velocities and sizes.
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
        return np.linalg.norm(self.array) < EPS


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

    def internal_random_coordinate(self):
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
    def __init__(self, coords):
        self.array = np.array(coords)
        self.length = coords[1] - coords[0]
        if self.length < 0:
            raise ValueError("Interval start larger than interval end")

    def random(self):
        return np.random.random() * self.length + self.array[0]

    begin = property(lambda s: s[0])
    end = property(lambda s: s[1])

    def __getitem__(self, item):
        return self.array[item]

    def __contains__(self, item: float):
        return self.array[0] <= item <= self.array[1]

    def __repr__(self):
        return "Interval %s" % self.array


# Moeten ook naar Points ipv arrays toe
class LineSegment(object):
    def __init__(self, coords):
        self.array = np.array(coords)
        self.length_2 = np.linalg.norm(coords[0] - coords[1])
        self.color = 'blue'

    length = property(lambda s: np.linalg.norm(s[0] - s[1]))
    begin = property(lambda s: s[0])
    end = property(lambda s: s[1])

    @staticmethod
    def path_norm(path, t):
        return np.linalg.norm(path[0] + t * (path[1] - path[0]), ord=1)

    def get_point(self, value:float):
        return self.begin + value * (self.end - self.begin)

    def crosses_obstacle(self, obstacle, strict=False):
        """
        Checks whether the line crosses the rectangular obstacle.
        Currently done quite inefficiently, even if I say so myself.
        Very precise though.
        :param obstacle: The rectangular obstacle to be checked
        :param strict: Whether the line passes only through the obstacle (strict = True)
         or also through the boundary (strict = False)
        :return: True if line crosses obstacle, false otherwise
        """
        line = np.array([self.begin, self.end])
        obs_matrix = np.array([obstacle.begin.array, obstacle.end.array])
        # Translation to center of object
        midpoint = np.mean(obs_matrix, axis=0)
        translated_line = line - midpoint
        # Scaling with size of object
        size = np.dot(np.array([-1, 1]), obs_matrix)  # self.end - self.begin
        scale = 2 / size
        scaled_path = translated_line * scale
        # Rotating 45 degrees
        rot_path = np.dot(scaled_path, rot_mat(np.pi / 4.) / np.sqrt(2))
        # 1 norm of function

        path_function = lambda x: LineSegment.path_norm(rot_path, x)
        t = solve_convex_piece_lin_ineq(path_function, lower_bound=1, interval=Interval([0., 1.]), strict=strict)
        return self.get_point(t) in obstacle

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


# Wrapper around list of line segments
class Path(object):
    def __init__(self, line_segment_list):
        self.list = line_segment_list
        for i in range(len(line_segment_list) - 1):
            self._check_connectivity(line_segment_list[i], line_segment_list[i + 1])

    @property
    def length(self):
        return sum([line.length for line in self.list])

    def _check_connectivity(self, ls1: LineSegment, ls2: LineSegment):
        connected = np.allclose(ls1.end, ls2.begin)
        if not connected:
            raise AssertionError("%s & %s do not connect" % (ls1, ls2))

    def __iadd__(self, other):
        if self.list:
            self._check_connectivity(self.list[-1], other[0])
        self.list += other

    def __getitem__(self, item):
        return self.list[item]

    def append(self, other):
        if isinstance(other, LineSegment):
            if self.list:
                self._check_connectivity(self.list[-1], other)
        elif isinstance(other, Point):
            if not self.list:
                raise AssertionError("List must have elements before adding Points")
            print("Last element: %s\n Last point of last element: %s\n" % (self.list[-1], self.list[-1][-1]))
            new_line_segment = LineSegment([Point(self.list[-1][-1]), other])
            self.list.append(new_line_segment)
        else:
            raise AttributeError("Object appended to Path must be LineSegment or Point, not %s" % other)
        self.list.append(other)

    def pop_next_segment(self):
        return self.list.pop(0)

    def __bool__(self):
        return bool(self.list)

    __nonzero__ = __bool__

    def __str__(self):
        return "Path:\n%s" % "\n".join([str(ls) for ls in self.list])
