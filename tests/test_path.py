__author__ = 'omar'
import sys

from nose.tools import raises

sys.path.insert(1, '../src')

from geometry import Point, LineSegment, Path
import functions as ft

demo_file_name = '../scenes/demo_obstacle_list.json'
empty_file_name = '../scenes/empty_scene.json'


class TestPath:
    def __init__(self):
        self.l1 = LineSegment([Point([0, 0]), Point([6, 7])])
        self.l2 = LineSegment([Point([6, 7]), Point([7, 4])])
        self.l21 = LineSegment([Point([6 + ft.EPS, 7]), Point([7, 4])])
        self.l3 = LineSegment([Point([7, 4]), Point([5, 5])])
        self.l4 = LineSegment([Point([3, 5]), Point([6, 7])])

    def test_connectivity(self):
        path = Path([self.l1, self.l2, self.l3])

    @raises(AssertionError)
    def test_connectivity_exception(self):
        path = Path([self.l1, self.l3, self.l3])

    def test_connectivity_with_small_error(self):
        path = Path([self.l1, self.l21, self.l3])

    def test_segment_getter(self):
        path = Path([])
        path.append(self.l2)
        assert path.pop_next_segment() == self.l2

    def test_bool_operator(self):
        path = Path([])
        assert not bool(path)
        path.append(self.l1)
        path.pop_next_segment()
        assert not bool(path)
