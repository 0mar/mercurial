__author__ = 'omar'
import sys

import numpy as np

sys.path.insert(1, '..')
from scene import Scene
from geometry import Size, Point, Velocity
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
        self.pedestrian.velocity = Velocity([1, -1])

    def test_cell_size(self):
        assert self.dyn_plan.dx == self.dyn_plan.dy == 5

    def test_density_never_negative(self):
        n = 5
        scene = Scene(size=Size([100, 100]), obstacle_file=empty_file_name, pedestrian_number=n)
        dyn_plan = DynamicPlanner(scene)
        dyn_plan.compute_density_and_velocity_field()
        density = dyn_plan.density
        assert np.all(density >= 0)

    def test_velocity_never_negative(self):
        n = 5
        scene = Scene(size=Size([100, 100]), obstacle_file=empty_file_name, pedestrian_number=n)
        dyn_plan = DynamicPlanner(scene)
        dyn_plan.compute_density_and_velocity_field()
        v_x, v_y = dyn_plan.v_x, dyn_plan.v_y
        assert np.all(v_x >= 0)
        assert np.all(v_y >= 0)

    def test_local_contributions_cross_threshold(self):
        self.dyn_plan.compute_density_and_velocity_field()
        density = self.dyn_plan.density
        center_cell = np.around(np.array([self.ped_x / self.dyn_plan.dx, self.ped_y / self.dyn_plan.dy]) - 1).astype(
            int)
        pedestrian_cell = np.floor(np.array([self.ped_x, self.ped_y]) / [self.dyn_plan.dx, self.dyn_plan.dy])
        for x in range(2):
            for y in range(2):
                assert density[tuple(center_cell + [x, y])] > 0
                if tuple(center_cell + [x, y]) == tuple(pedestrian_cell):
                    assert density[tuple(center_cell + [x, y])] >= self.dyn_plan.density_threshold

    def test_non_local_contributions_below_threshold(self):
        self.dyn_plan.compute_density_and_velocity_field()
        density = self.dyn_plan.density
        center_cell = np.around(np.array([self.ped_x / self.dyn_plan.dx, self.ped_y / self.dyn_plan.dy]) - 1).astype(
            int)
        pedestrian_cell = np.floor(np.array([self.ped_x, self.ped_y]) / [self.dyn_plan.dx, self.dyn_plan.dy])
        for x in range(2):
            for y in range(2):
                assert density[tuple(center_cell + [x, y])] > 0
                if tuple(center_cell + [x, y]) != tuple(pedestrian_cell):
                    print("center cell: %s" % center_cell)
                    print("pedestrian %s" % self.pedestrian)
                    print("Density in cell:  %.5f" % density[tuple(center_cell + [x, y])])
                    print("Density treshold: %.5f" % self.dyn_plan.density_threshold)
                    assert density[tuple(center_cell + [x, y])] <= self.dyn_plan.density_threshold

    def test_other_contributions_vanish(self):
        self.dyn_plan.compute_density_and_velocity_field()
        density = self.dyn_plan.density
        center_cell = np.around(np.array([self.ped_x / self.dyn_plan.dx, self.ped_y / self.dyn_plan.dy]) - 1).astype(
            int)
        for i, j in np.ndindex(self.dyn_plan.grid_dimension):
            if i not in [center_cell[0], center_cell[0] + 1] or j not in [center_cell[1], center_cell[1] + 1]:
                assert density[i, j] <= self.dyn_plan.density_epsilon

    def test_boundary_processed_correctly(self):
        upper_left_corner = Point([self.dyn_plan.dx / 2, self.scene.size[1] - self.dyn_plan.dy / 2])
        self.pedestrian.manual_move(upper_left_corner)
        self.dyn_plan.compute_density_and_velocity_field()
        assert self.dyn_plan.density is not None

    def test_normalize_field_non_negative_entries(self):
        field = np.random.random([20, 20]) * 3 - 1  # between 1 and 2
        rel_field = DynamicPlanner.get_normalized_field(field, 0, 1)
        assert np.all(rel_field >= 0)

    def test_normalize_field_smaller_equal_one(self):
        field = np.random.random([20, 20]) * 3 - 1  # between 1 and 2
        rel_field = DynamicPlanner.get_normalized_field(field, 0, 1)
        assert np.all(rel_field <= 1)

    def test_normalize_field_relative(self):
        field = np.random.random([5, 5]) * 3  # between 0 and 3
        rel_field = DynamicPlanner.get_normalized_field(field, 0, 4)

        assert np.allclose(rel_field, field / 4)

    def test_speed_contribution_to_neighbour_cells(self):
        self.dyn_plan.compute_density_and_velocity_field()
        density = self.dyn_plan.density
        center_cell = np.around(np.array([self.ped_x / self.dyn_plan.dx, self.ped_y / self.dyn_plan.dy]) - 1).astype(
            int)
        pedestrian_cell = np.floor(np.array([self.ped_x, self.ped_y]) / [self.dyn_plan.dx, self.dyn_plan.dy])
        for dir in DynamicPlanner.DIRECTIONS:
            self.dyn_plan.compute_speed_field(dir)
            for x in range(2):
                if 0 <= x + center_cell[0] < self.dyn_plan.speed_field_dict[dir].shape[0]:
                    for y in range(2):
                        if 0 <= y + center_cell[y] < self.dyn_plan.speed_field_dict[dir].shape[1]:
                            assert self.dyn_plan.speed_field_dict[dir][tuple(center_cell + [x, y])] <= \
                                   self.dyn_plan.max_speed
                            print("center cell: %s" % center_cell)
                            print("pedestrian %s" % self.pedestrian)

    def test_speed_no_larger_than_max_speed(self):
        n = 10
        scene = Scene(size=Size([100, 100]), obstacle_file=empty_file_name, pedestrian_number=n)
        dyn_plan = DynamicPlanner(scene)
        dyn_plan.compute_density_and_velocity_field()
        for dir in DynamicPlanner.DIRECTIONS:
            dyn_plan.compute_speed_field(dir)
            assert np.all(dyn_plan.speed_field_dict[dir] <= dyn_plan.max_speed)

    def test_alignment_makes_speed_max_speed(self):
        direction = 'down'
        self.pedestrian.max_speed = self.dyn_plan.max_speed
        self.pedestrian.velocity = Velocity(DynamicPlanner.DIRECTIONS[direction])
        self.dyn_plan.compute_density_and_velocity_field()
        self.dyn_plan.compute_speed_field(direction)
        print("Velocity v_x\n%s\n\n" % self.dyn_plan.v_x)
        print("Velocity v_y\n%s\n\n" % self.dyn_plan.v_y)
        print("Speed field %s:\n%s\n" % (direction, self.dyn_plan.speed_field_dict[direction]))
        assert np.allclose(self.dyn_plan.speed_field_dict[direction], self.dyn_plan.max_speed, 0.01)
