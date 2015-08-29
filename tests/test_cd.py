__author__ = 'omar'
from nose.tools import raises

from geometry import *
from scene import *


class TestPedestrian:
    def __init__(self):
        print("Initializing the class")
        self.scene = Scene(size=Size([250, 150]), pedestrian_number=1000, obstacle_file='demo_obstacle_list.json')

    def setup(self):
        # print("Supposed to happen for each class")
        pass

    def test_pedestrian_location_within_domain(self):
        pedestrian_list = []
        for i in range(1000):
            pedestrian_list.append(Pedestrian(self.scene, 1, self.scene.obstacle_list[-1]))
        assert all([all(ped.position.array < self.scene.size.array) for ped in pedestrian_list])

    def test_pedestrian_location_not_in_obstacle(self):
        pedestrian_list = []
        for i in range(1000):
            pedestrian_list.append(Pedestrian(self.scene, 1, self.scene.obstacle_list[-1]))
        assert all([ped.position not in obstacle for obstacle in self.scene.obstacle_list for ped in pedestrian_list])


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


from planner import GraphPlanner


class TestGraphPlanner:
    def __init__(self):
        self.scene = Scene(size=Size([250, 150]), pedestrian_number=10, obstacle_file='demo_obstacle_list.json')
        self.gt = GraphPlanner(self.scene)

    def test_path_cross_no_obstacles(self):
        ped = self.scene.pedestrian_list[0]
        for obstacle in self.scene.obstacle_list:
            for line_segment in ped.path:
                assert (not line_segment.crosses_obstacle(obstacle)) or obstacle.permeable


    def test_path_from_pedestrian_to_finish(self):
        ped = self.scene.pedestrian_list[0]
        if ped.path:
            assert ped.path[-1].end in ped.goal
        else:
            assert ped.line.end in ped.goal


class TestCell:
    def __init__(self):
        self.ped_number = 1000
        self.scene = Scene(size=Size([250, 150]), pedestrian_number=self.ped_number,
                           obstacle_file='demo_obstacle_list.json')

    def test_ped_distribution(self):
        counted_ped = 0
        for cell in self.scene.cell_dict.values():
            counted_ped += len(cell.pedestrian_set)
        assert counted_ped == self.ped_number

    def test_partitioning(self):
        for _ in range(300):
            loc = self.scene.size.random_internal_point()
            found = 0
            for cell in self.scene.cell_dict.values():
                found += int(loc in cell)
            assert found == 1


class TestFunctions:
    def __init__(self):
        self.rec1 = (Point([0, 0]), Point([5, 5]))
        self.rec2 = (Point([4, 4]), Point([6, 6]))
        self.rec3 = (Point([5, 0]), Point([6, 1]))
        self.rec4 = (Point([2, 2]), Point([3, 3]))
        self.rec5 = (Point([2, -1]), Point([4, 6]))
        self.rec6 = (Point([2, -1]), Point([4, 4]))

    def test_overlapping_rectangles(self):
        start1, end1 = self.rec1
        start2, end2 = self.rec2
        assert rectangles_intersect(start1, end1, start2, end2)

    def test_adjacent_rectangles(self):
        start1, end1 = self.rec1
        start2, end2 = self.rec3
        assert rectangles_intersect(start1, end1, start2, end2, open_sets=False)
        assert not rectangles_intersect(start1, end1, start2, end2, open_sets=True)

    def test_containing_rectangles(self):
        start1, end1 = self.rec1
        start2, end2 = self.rec4
        assert rectangles_intersect(start1, end1, start2, end2)

    def test_unorderable_rectangles(self):
        start1, end1 = self.rec1
        start2, end2 = self.rec5
        start3, end3 = self.rec6
        assert rectangles_intersect(start1, end1, start2, end2)
        assert rectangles_intersect(start1, end1, start3, end3)

    def test_non_overlapping_rectangles(self):
        start1, end1 = self.rec3
        start2, end2 = self.rec5
        assert not rectangles_intersect(start1, end1, start2, end2)


class TestScene:
    def __init__(self):
        self.scene_obj = Scene(size=Size([20, 20]), obstacle_file='empty_obstacle_file.json',
                               pedestrian_number=50)

    def test_create_cells(self):
        assert self.scene_obj.cell_dict[(1, 1)].begin == Point([0, 0])
