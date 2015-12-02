__author__ = 'omar'
import sys

import numpy as np

sys.path.insert(1, '../src')

from scene import Scene
from geometry import Size
from simulation_manager import SimulationManager
demo_file_name = '../scenes/demo_obstacle_list.json'
empty_file_name = '../scenes/empty_scene.json'

from static_planner import GraphPlanner


class TestGraphPlanner:
    def __init__(self):
        config_1 = SimulationManager.get_default_config()
        config_2 = SimulationManager.get_default_config()
        config_1['general']['obstacle_file']=demo_file_name
        config_2['general']['obstacle_file']=empty_file_name
        self.filled_scene = Scene(config=config_1,initial_pedestrian_number=10)
        self.empty_scene = Scene(config=config_2,initial_pedestrian_number=10)
        self.gt1 = GraphPlanner(self.filled_scene, config_1)
        self.gt2 = GraphPlanner(self.empty_scene,config_2)

    def test_path_cross_no_obstacles(self):
        ped = self.filled_scene.pedestrian_list[0]
        for obstacle in self.filled_scene.obstacle_list:
            for line_segment in ped.path:
                assert (not line_segment.crosses_obstacle(obstacle)) or obstacle.permeable

    def test_path_from_pedestrian_to_finish(self):
        ped = self.filled_scene.pedestrian_list[0]
        if ped.path:
            assert any([ped.path[-1].end in exit for exit in self.filled_scene.exit_list])
        else:
            assert any([ped.line.end in exit for exit in self.filled_scene.exit_list])

    def test_path_at_least_distance(self):
        distances = []
        for pedestrian in self.filled_scene.pedestrian_list:
            for exit in self.filled_scene.exit_list:
                print(exit)
                distance = np.linalg.norm(pedestrian.position.array -
                                          self.gt1.get_closest_goal_position(pedestrian.position, exit))
                path_length = GraphPlanner.get_path_length(pedestrian)
                print("Distance: %.2f " % distance)
                print("path_length: %.2f " % path_length)
                distances.append(distance)
            assert any([distance <= path_length for distance in distances])

    def test_path_equal_to_distance_without_obstacles(self):
        for pedestrian in self.empty_scene.pedestrian_list:
            for exit in self.empty_scene.exit_list:
                distance = np.linalg.norm(pedestrian.position.array -
                                          self.gt2.get_closest_goal_position(pedestrian.position, exit))
                path_length = GraphPlanner.get_path_length(pedestrian)
                assert distance >= path_length
