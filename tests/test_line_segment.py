__author__ = 'omar'
import sys

sys.path.insert(1, '../src')

from objects.scene import Obstacle
from math_objects.geometry import Size, Point, LineSegment

demo_file_name = '../scenes/demo_obstacle_list.json'
empty_file_name = '../scenes/empty_scene.json'


class TestLineSegment:
    def __init__(self):
        self.obs = Obstacle(Point([5, 7]), Size([2, 3]), 'test obs')

    def test_line_boundaries(self):
        line = LineSegment([Point([5, 5]), Point([6, 11])])
        assert all(line.get_point(0) == line.begin)
        assert all(line.get_point(1) == line.end)

    def test_line_passing_through_obstacle(self):
        line = LineSegment([Point([6, 6]), Point([6, 11])])
        assert line.begin not in self.obs and line.end not in self.obs
        assert line.crosses_obstacle(self.obs)

    def test_line_starting_in_obstacle(self):
        line = LineSegment([Point([6, 8]), Point([7, 15])])
        assert line.begin in self.obs
        assert line.crosses_obstacle(self.obs)

    def test_line_starting_in_corner(self):
        line = LineSegment([Point([7, 10]), Point([7, 15])])
        assert line.begin in self.obs
        assert line.crosses_obstacle(self.obs)
        assert not line.crosses_obstacle(self.obs, open_sets=True)

    def test_line_ending_in_corner(self):
        line = LineSegment([Point([3, 4]), Point([5, 7])])
        assert line.end in self.obs
        assert line.crosses_obstacle(self.obs, open_sets=False)
        assert not line.crosses_obstacle(self.obs, open_sets=True)

    def test_line_far_from_obstacle(self):
        line = LineSegment([Point([4, 5]), Point([2, 3])])
        assert line.begin not in self.obs and line.end not in self.obs
        assert not line.crosses_obstacle(self.obs)

    def test_line_close_to_border(self):
        line = LineSegment([Point([4, 5]), Point([5, 6.94])])
        assert line.begin not in self.obs and line.end not in self.obs
        assert not line.crosses_obstacle(self.obs)

    def test_line_starting_close_to_corner(self):
        line = LineSegment([Point([7.01, 10.01]), Point([7, 15])])
        assert line.begin not in self.obs
        assert not line.crosses_obstacle(self.obs)

    def test_decreasing_line(self):
        line = LineSegment([Point([3, 5]), Point([6, 2])])
        assert line.begin not in self.obs
        assert not line.crosses_obstacle(self.obs)
