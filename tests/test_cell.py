__author__ = 'omar'
import sys

sys.path.insert(1, '../src')

from scene import Scene
from geometry import Size

demo_file_name = '../scenes/demo_obstacle_list.json'
empty_file_name = '../scenes/empty_scene.json'


class TestCell:
    def __init__(self):
        self.ped_number = 1000
        self.scene = Scene(size=Size([250, 150]), pedestrian_number=self.ped_number,
                           obstacle_file=demo_file_name)

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
