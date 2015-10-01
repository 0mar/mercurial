__author__ = 'omar'
import sys

import numpy as np

sys.path.insert(1, '../src')

from scene import Scene
from geometry import Size

demo_file_name = '../scenes/demo_obstacle_list.json'
empty_file_name = '../scenes/empty_scene.json'

from static_planner import GraphPlanner


class TestGraphPlanner:
    def __init__(self):
        self.filled_scene = Scene(size=Size([250, 150]), pedestrian_number=10, obstacle_file=demo_file_name)
        self.empty_scene = Scene(size=Size([250, 150]), pedestrian_number=10, obstacle_file=empty_file_name)
        self.gt1 = GraphPlanner(self.filled_scene)
        self.gt2 = GraphPlanner(self.empty_scene)

    def test_path_cross_no_obstacles(self):
        ped = self.filled_scene.pedestrian_list[0]
        for obstacle in self.filled_scene.obstacle_list:
            for line_segment in ped.path:
                assert (not line_segment.crosses_obstacle(obstacle)) or obstacle.permeable

    def test_path_from_pedestrian_to_finish(self):
        ped = self.filled_scene.pedestrian_list[0]
        if ped.path:
            assert ped.path[-1].end in ped.goal
        else:
            assert ped.line.end in ped.goal

    def test_path_at_least_distance(self):
        for pedestrian in self.filled_scene.pedestrian_list:
            distance = np.linalg.norm(pedestrian.position.array -
                                      self.gt1.get_closest_goal_position(pedestrian.position,
                                                                         self.filled_scene.exit_obs))
            path_length = GraphPlanner.get_path_length(pedestrian)
            assert distance <= path_length

    def test_path_equal_to_distance_without_obstacles(self):
        for pedestrian in self.empty_scene.pedestrian_list:
            distance = np.linalg.norm(pedestrian.position.array -
                                      self.gt2.get_closest_goal_position(pedestrian.position,
                                                                         self.empty_scene.exit_obs))
            path_length = GraphPlanner.get_path_length(pedestrian)
            assert distance == path_length
