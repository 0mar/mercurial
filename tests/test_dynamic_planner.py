__author__ = 'omar'
import sys

import numpy as np

sys.path.insert(1, '..')
from scene import Scene
from geometry import Size, Point
from dynamic_planner import DynamicPlanner

empty_file_name = '../empty_scene.json'


class TestDynamicPlanner:
    def __init__(self):
        self.scene = Scene(size=Size([100, 100]), obstacle_file=empty_file_name, pedestrian_number=1)
        self.dyn_plan = DynamicPlanner(self.scene)
        self.dyn_plan.grid_dimension = (20, 20)
        self.pedestrian = self.scene.pedestrian_list[0]
        self.ped_x = np.random.randint(3, self.scene.size.width - 2)
        self.ped_y = np.random.randint(3, self.scene.size.height - 2)
        self.pedestrian.manual_move(Point([self.ped_x, self.ped_y]))

    def test_cell_size(self):
        assert self.dyn_plan.dx == self.dyn_plan.dy == 5

    def test_density_never_negative(self):
        n = 5
        scene = Scene(size=Size([100, 100]), obstacle_file=empty_file_name, pedestrian_number=n)
        dyn_plan = DynamicPlanner(scene)
        (density, _, _) = dyn_plan.obtain_density_and_velocity_field()
        assert np.all(density >= 0)

    def test_velocity_never_negative(self):
        n = 5
        scene = Scene(size=Size([100, 100]), obstacle_file=empty_file_name, pedestrian_number=n)
        dyn_plan = DynamicPlanner(scene)
        (_, v_x, v_y) = dyn_plan.obtain_density_and_velocity_field()
        assert np.all(v_x >= 0)
        assert np.all(v_y >= 0)

    def test_local_contributions_cross_threshold(self):
        (density, _, _) = self.dyn_plan.obtain_density_and_velocity_field()
        center_cell = np.around(np.array([self.ped_x / self.dyn_plan.dx, self.ped_y / self.dyn_plan.dy]) - 1).astype(
            int)
        pedestrian_cell = np.floor(np.array([self.ped_x, self.ped_y]) / [self.dyn_plan.dx, self.dyn_plan.dy])
        for x in range(2):
            for y in range(2):
                assert density[tuple(center_cell + [x, y])] > 0
                if tuple(center_cell + [x, y]) == tuple(pedestrian_cell):
                    assert density[tuple(center_cell + [x, y])] >= self.dyn_plan.density_threshold

    def test_non_local_contributions_below_threshold(self):
        (density, _, _) = self.dyn_plan.obtain_density_and_velocity_field()
        center_cell = np.around(np.array([self.ped_x / self.dyn_plan.dx, self.ped_y / self.dyn_plan.dy]) - 1).astype(
            int)
        pedestrian_cell = np.floor(np.array([self.ped_x, self.ped_y]) / [self.dyn_plan.dx, self.dyn_plan.dy])
        for x in range(2):
            for y in range(2):
                assert density[tuple(center_cell + [x, y])] > 0
                if tuple(center_cell + [x, y]) != tuple(pedestrian_cell):
                    assert density[tuple(center_cell + [x, y])] <= self.dyn_plan.density_threshold

    def test_other_contributions_vanish(self):
        (density, _, _) = self.dyn_plan.obtain_density_and_velocity_field()
        center_cell = np.around(np.array([self.ped_x / self.dyn_plan.dx, self.ped_y / self.dyn_plan.dy]) - 1).astype(
            int)
        for i, j in np.ndindex(self.dyn_plan.grid_dimension):
            if i not in [center_cell[0], center_cell[0] + 1] or j not in [center_cell[1], center_cell[1] + 1]:
                assert density[i, j] == 0

    def test_boundary_processed_correctly(self):
        upper_left_corner = Point([self.dyn_plan.dx / 2, self.scene.size[1] - self.dyn_plan.dy / 2])
        self.pedestrian.manual_move(upper_left_corner)
        (density, _, _) = self.dyn_plan.obtain_density_and_velocity_field()
        assert density is not None
