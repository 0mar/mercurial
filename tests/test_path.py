__author__ = 'omar'
import sys

import numpy as np
from nose.tools import raises

sys.path.insert(1, '../src')

from math_objects.geometry import Point, LineSegment, Path
from math_objects import functions as ft

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
        path = Path([self.l2])
        assert path.pop_next_segment() == self.l2

    @raises(ValueError)
    def test_bool_operator(self):
        path = Path([])

    def test_exhausted_list(self):
        path = Path([self.l1])
        path.pop_next_segment()
        assert not bool(path)

    def test_path_sample_length(self):
        path = Path([self.l1, self.l21, self.l3])
        assert path.sample_points.shape[0] == 5

    def test_path_sample_distance(self):
        path = Path([self.l1, self.l21, self.l3])
        sample1 = path.sample_points[2]
        sample2 = path.sample_points[3]
        print("Distance between sample points 2 and 3: %s" % np.linalg.norm(sample1 - sample2))
        assert np.linalg.norm(sample1 - sample2) < Path.sample_length

    def test_samples_from_start_to_finish(self):
        path = Path([self.l1, self.l21, self.l3])
        assert np.allclose(path[0].begin, path.sample_points[0])
        assert np.allclose(path[-1].end, path.sample_points[-1])
