__author__ = 'omar'
from nose import with_setup
from nose.tools import raises

from functions import *
from geometry import *
from pedestrian import Pedestrian


class TestPedestrian:
    def __init__(self):
        print("Initializing the class")
        self.scene = Scene(size=Size([250, 150]), pedNumber=1000)

    def setup(self):
        print("Supposed to happen for each class")

    def test_pedestrian_location_within_domain(self):
        ped_list = []
        for i in range(1000):
            ped_list.append(Pedestrian(self.scene, 1, self.scene.obs_list[-1]))
        assert [ped.position.array < self.scene.size.array for ped in ped_list].all()

    def test_pedestrian_location_not_in_obstacle(self):
        ped_list = []
        for i in range(1000):
            ped_list.append(Pedestrian(self.scene, 1, self.scene.obs_list[-1]))
        assert [ped not in obstacle for obstacle in self.scene.obs_list for ped in ped_list].all()


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
        assert not line.crosses_obstacle(self.obs, strict=True)

    def test_line_ending_in_corner(self):
        line = LineSegment([Point([3, 4]), Point([5, 7])])
        assert line.end in self.obs
        assert line.crosses_obstacle(self.obs)
        assert not line.crosses_obstacle(self.obs, strict=True)

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


class TestPath:
    def __init__(self):
        self.l1 = LineSegment([Point([0, 0]), Point([6, 7])])
        self.l2 = LineSegment([Point([6, 7]), Point([7, 4])])
        self.l21 = LineSegment([Point([6 + EPS, 7]), Point([7, 4])])
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

from graphtesting import GT
class TestGraphPlanner:

    def __init__(self):
        self.scene = Scene(size=Size([250, 150]), pedNumber=1)
        self.gt = GT(self.scene)

    def test_line_segments_cross_no_objects(self):
        pass

    def test_path_crosses_no_obstacles(self):
        pass

    def test_path_from_pedestrian_to_finish(self):
        pass