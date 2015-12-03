__author__ = 'omar'
import sys

import numpy as np

sys.path.insert(1, '../src')
from scene import Scene
from simulation_manager import SimulationManager
from geometry import Size, Point, Velocity
from dynamic_planner import DynamicPlanner
from scalar_field import ScalarField as Field
import functions as ft

empty_scene_file = '../scenes/large_exit.json'
test_fractions_file = '../scenes/test_fractions.json'

class TestDynamicPlanner:
    def __init__(self):
        self.config=SimulationManager.get_default_config()
        self.config['general']['obstacle_file']=empty_scene_file
        self.scene = Scene(config=self.config, initial_pedestrian_number=1)
        self.dyn_plan = DynamicPlanner(self.scene,self.config)
        self.pedestrian = self.scene.pedestrian_list[0]
        self.ped_x = np.random.randint(3, self.scene.size.width - 2)
        self.ped_y = np.random.randint(3, self.scene.size.height - 2)
        self.pedestrian.position = Point([self.ped_x, self.ped_y])
        self.pedestrian.velocity = Velocity([1, -1])

    def test_cell_size(self):
        assert self.dyn_plan.dx == self.dyn_plan.dy == 10

    def test_density_never_negative(self):
        n = 5
        scene = Scene(config=self.config, initial_pedestrian_number=n)
        dyn_plan = DynamicPlanner(scene, self.config)
        dyn_plan.compute_density_and_velocity_field()
        density = dyn_plan.density_field.array
        assert np.all(density >= 0)

    def test_velocity_never_negative(self):
        n = 5
        scene = Scene(config=self.config, initial_pedestrian_number=n)
        dyn_plan = DynamicPlanner(scene,self.config)
        dyn_plan.compute_density_and_velocity_field()
        v_x, v_y = dyn_plan.v_x.array, dyn_plan.v_y.array
        assert np.all(v_x >= 0)
        assert np.all(v_y >= 0)

    def test_local_contributions_cross_threshold(self):
        self.dyn_plan.compute_density_and_velocity_field()
        density = self.dyn_plan.density_field.array
        center_cell = np.around(np.array([self.ped_x / self.dyn_plan.dx, self.ped_y / self.dyn_plan.dy]) - 1).astype(
            int)
        pedestrian_cell = np.floor(np.array([self.ped_x, self.ped_y]) / [self.dyn_plan.dx, self.dyn_plan.dy])
        for x in range(2):
            for y in range(2):
                assert density[tuple(center_cell + [x, y])] > 0
                if tuple(center_cell + [x, y]) == tuple(pedestrian_cell):
                    print("Local density: %.4f, threshold %.4f" % (
                        density[tuple(center_cell + [x, y])], self.dyn_plan.density_threshold))
                    assert density[tuple(
                        center_cell + [x, y])] >= self.dyn_plan.density_threshold - self.dyn_plan.density_epsilon

    def test_non_local_contributions_below_threshold(self):
        self.dyn_plan.compute_density_and_velocity_field()
        density = self.dyn_plan.density_field.array
        center_cell = np.around(np.array([self.ped_x / self.dyn_plan.dx, self.ped_y / self.dyn_plan.dy]) - 1).astype(
            int)
        pedestrian_cell = np.floor(np.array([self.ped_x, self.ped_y]) / [self.dyn_plan.dx, self.dyn_plan.dy])
        print("Pedestrian cell %s"%pedestrian_cell)
        for x in range(2):
            for y in range(2):
                print("Dens in (%d,%d):\t%s"%(center_cell[0]+x,center_cell[1]+y,density[tuple(center_cell + [x, y])]))
                assert density[tuple(center_cell + [x, y])] > 0
                if tuple(center_cell + [x, y]) != tuple(pedestrian_cell):
                    print("center cell: %s" % center_cell)
                    print("pedestrian %s" % self.pedestrian)
                    print("Density in cell:  %.5f" % density[tuple(center_cell + [x, y])])
                    print("Density threshold: %.5f" % self.dyn_plan.density_threshold)
                    assert density[tuple(center_cell + [x, y])] <= self.dyn_plan.density_threshold

    def test_other_contributions_vanish(self):
        self.dyn_plan.compute_density_and_velocity_field()
        density = self.dyn_plan.density_field.array
        center_cell = np.around(np.array([self.ped_x / self.dyn_plan.dx, self.ped_y / self.dyn_plan.dy]) - 1).astype(
            int)
        for i, j in np.ndindex(self.dyn_plan.grid_dimension):
            if i not in [center_cell[0], center_cell[0] + 1] or j not in [center_cell[1], center_cell[1] + 1]:
                assert density[i, j] <= self.dyn_plan.density_epsilon

    def test_boundary_processed_correctly(self):
        upper_left_corner = Point([self.dyn_plan.dx / 2, self.scene.size[1] - self.dyn_plan.dy / 2])
        self.pedestrian.position = upper_left_corner
        self.dyn_plan.compute_density_and_velocity_field()
        assert self.dyn_plan.density_field is not None

    def test_normalize_field_non_negative_entries(self):
        field = Field((20, 20), Field.Orientation.center, '')
        field.update(np.random.random([20, 20]) * 3 - 1)
        rel_field = field.normalized(0, 1)
        assert np.all(rel_field >= 0)

    def test_normalize_field_smaller_equal_one(self):
        field = Field((20, 20), Field.Orientation.center, '')
        field.update(np.random.random([20, 20]) * 3 - 1)
        rel_field = field.normalized(0, 1)
        assert np.all(rel_field <= 1)

    def test_normalize_field_relative(self):
        field = Field((20, 20), Field.Orientation.center, '')
        field.update(np.random.random([20, 20]) * 4)
        rel_field = field.normalized(0, 4)
        assert np.allclose(rel_field, field.array / 4)

    def test_speed_contribution_to_neighbour_cells(self):
        self.dyn_plan.compute_density_and_velocity_field()
        density = self.dyn_plan.density_field.array
        center_cell = np.around(np.array([self.ped_x / self.dyn_plan.dx, self.ped_y / self.dyn_plan.dy]) - 1).astype(
            int)
        pedestrian_cell = np.floor(np.array([self.ped_x, self.ped_y]) / [self.dyn_plan.dx, self.dyn_plan.dy])
        for dir in ft.DIRECTIONS:
            self.dyn_plan.compute_speed_field(dir)
            for x in range(2):
                if 0 <= x + center_cell[0] < self.dyn_plan.speed_field_dict[dir].array.shape[0]:
                    for y in range(2):
                        if 0 <= y + center_cell[y] < self.dyn_plan.speed_field_dict[dir].array.shape[1]:
                            assert self.dyn_plan.speed_field_dict[dir].array[tuple(center_cell + [x, y])] <= \
                                   self.dyn_plan.max_speed
                            print("center cell: %s" % center_cell)
                            print("pedestrian %s" % self.pedestrian)

    def test_speed_no_larger_than_max_speed(self):
        n = 10
        scene = Scene(config=self.config, initial_pedestrian_number=n)
        dyn_plan = DynamicPlanner(scene,self.config)
        dyn_plan.compute_density_and_velocity_field()
        for dir in ft.DIRECTIONS:
            dyn_plan.compute_speed_field(dir)
            assert np.all(dyn_plan.speed_field_dict[dir].array <= dyn_plan.max_speed)

    def test_alignment_makes_speed_max_speed(self):
        for direction in ft.DIRECTIONS:
            self.pedestrian.max_speed = self.dyn_plan.max_speed
            self.pedestrian.velocity = Velocity(ft.DIRECTIONS[direction])
            self.dyn_plan.compute_density_and_velocity_field()
            self.dyn_plan.compute_speed_field(direction)
            print("Velocity v_x\n%s\n\n" % self.dyn_plan.v_x.array)
            print("Velocity v_y\n%s\n\n" % self.dyn_plan.v_y.array)
            print("Speed field %s:\n%s\n" % (direction, self.dyn_plan.speed_field_dict[direction]))
            assert np.allclose(self.dyn_plan.speed_field_dict[direction].array, self.dyn_plan.max_speed, 0.01)

    def test_discomfort_field_normalized(self):
        self.dyn_plan.compute_density_and_velocity_field()
        self.dyn_plan.compute_discomfort_field()
        assert np.all(self.dyn_plan.discomfort_field.array >= 0) and np.all(self.dyn_plan.discomfort_field.array <= 1)

    def test_unit_cost_field_always_positive(self):
        # Might be zero if self.time_weight == 0. Zero is bad.
        self.dyn_plan.compute_density_and_velocity_field()
        self.dyn_plan.compute_discomfort_field()
        for direction in ft.DIRECTIONS:
            self.dyn_plan.compute_speed_field(direction)
            self.dyn_plan.compute_unit_cost_field(direction)
            assert np.all(self.dyn_plan.unit_field_dict[direction].array > 0)

    def test_initial_potential_respects_exits(self):
        mesh = self.dyn_plan.potential_field.mesh_grid
        for i, j in np.ndindex(self.dyn_plan.grid_dimension):
            interface_val = self.dyn_plan.initial_interface[(i, j)]
            is_in_exit = any([Point([mesh[0][(i, j)], mesh[1][(i, j)]]) in exit for exit in self.scene.exit_list])
            assert (interface_val == 0) == is_in_exit

    def test_no_obstacle_means_no_fraction(self):
        config = SimulationManager.get_default_config()
        config['general']['obstacle_file']=test_fractions_file
        scene = Scene(1,config)
        dyn_plan = DynamicPlanner(scene,config)
        # Cell (7,10) is a free cell.
        assert (7, 10) not in dyn_plan.obstacle_cell_set and (7, 10) not in dyn_plan.part_obstacle_cell_dict

    # def test_fully_covered_means_high_potential(self):
    #     config = SimulationManager.get_default_config()
    #     config['general']['obstacle_file']=test_fractions_file
    #     scene = Scene(1,config)
    #     dyn_plan = DynamicPlanner(scene,config)
    #     # Cell (9,9) is a fully covered cell.
    #     assert (9, 9) in dyn_plan.obstacle_cell_set
    #     dyn_plan.compute_density_and_velocity_field()
    #     dyn_plan.compute_discomfort_field()
    #     for direction in ft.DIRECTIONS:
    #         dyn_plan.compute_speed_field(direction)
    #         dyn_plan.compute_unit_cost_field(direction)
    #     dyn_plan.compute_potential_field()
    #     print("Real potential: %.2f" % (np.max(dyn_plan.potential_field.array[9, 9])))
    #     print("Theoretical potential: %.2f" % (np.max(dyn_plan.potential_field.array)))
    #     assert dyn_plan.potential_field.array[9, 9] == np.max(dyn_plan.potential_field.array)

    # def test_fraction_method(self):
    #     config = SimulationManager.get_default_config()
    #     config['general']['obstacle_file']=test_fractions_file
    #     scene = Scene(1,config)
    #     dyn_plan = DynamicPlanner(scene,config)
    #     # Cell (8,10) is a partly covered cell.
    #     assert (8, 10) in dyn_plan.part_obstacle_cell_dict
    #     frac_val = dyn_plan.part_obstacle_cell_dict[(8, 10)]
    #     assert np.allclose(frac_val, 0.25)
    #     dyn_plan.compute_density_and_velocity_field()
    #     dyn_plan.compute_discomfort_field()
    #     for direction in ft.DIRECTIONS:
    #         dyn_plan.compute_speed_field(direction)
    #         dyn_plan.compute_unit_cost_field(direction)
    #     dyn_plan.compute_potential_field()
    #     print("Theoretical potential: %.2f" % (np.max(dyn_plan.potential_field.array) * frac_val))
    #     print("Real potential: %.2f" % (dyn_plan.potential_field.array[8, 10]))
    #     assert np.allclose(
    #         dyn_plan.potential_field.array[8, 10], np.max(dyn_plan.potential_field.array) * frac_val)

    def test_max_index_method(self):
        dim = (5, 5)
        cell_list = [(-1, 4), (0, -1), (5, 4), (-1, 4)]
        for cell in cell_list:
            assert not self.dyn_plan._exists(cell, dim)

    def test_updating_dynamic_planner(self):
        self.dyn_plan.step()
        for member in self.dyn_plan.__dict__:
            if isinstance(member, Field):
                assert member.time_step == 1
