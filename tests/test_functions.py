__author__ = 'omar'
import sys

sys.path.insert(1, '../src')

from geometry import Point
import functions as ft

demo_file_name = '../scenes/demo_obstacle_list.json'
empty_file_name = '../scenes/empty_scene.json'


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
        assert ft.rectangles_intersect(start1, end1, start2, end2)

    def test_adjacent_rectangles(self):
        start1, end1 = self.rec1
        start2, end2 = self.rec3
        assert ft.rectangles_intersect(start1, end1, start2, end2, open_sets=False)
        assert not ft.rectangles_intersect(start1, end1, start2, end2, open_sets=True)

    def test_containing_rectangles(self):
        start1, end1 = self.rec1
        start2, end2 = self.rec4
        assert ft.rectangles_intersect(start1, end1, start2, end2)

    def test_unorderable_rectangles(self):
        start1, end1 = self.rec1
        start2, end2 = self.rec5
        start3, end3 = self.rec6
        assert ft.rectangles_intersect(start1, end1, start2, end2)
        assert ft.rectangles_intersect(start1, end1, start3, end3)

    def test_non_overlapping_rectangles(self):
        start1, end1 = self.rec3
        start2, end2 = self.rec5
        assert not ft.rectangles_intersect(start1, end1, start2, end2)
