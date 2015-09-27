__author__ = 'omar'
import sys

sys.path.insert(1, '../src')

from scene import Scene, Pedestrian
from geometry import Size

demo_file_name = '../scenes/demo_obstacle_list.json'
empty_file_name = '../scenes/empty_scene.json'


class TestPedestrian:
    def __init__(self):
        print("Initializing the class")
        self.scene = Scene(size=Size([250, 150]), pedestrian_number=1000, obstacle_file=demo_file_name)

    def setup(self):
        # print("Supposed to happen for each class")
        pass

    def test_pedestrian_location_within_domain(self):
        pedestrian_list = []
        for i in range(1000):
            pedestrian_list.append(
                Pedestrian(self.scene, 1, self.scene.obstacle_list[-1], size=Size([1, 1]), max_speed=1))
        assert all([all(ped.position.array < self.scene.size.array) for ped in pedestrian_list])

    def test_pedestrian_location_not_in_obstacle(self):
        pedestrian_list = []
        for i in range(1000):
            pedestrian_list.append(
                Pedestrian(self.scene, 1, self.scene.obstacle_list[-1], size=Size([1, 1]), max_speed=1))
        assert all([ped.position not in obstacle for obstacle in self.scene.obstacle_list for ped in pedestrian_list])
