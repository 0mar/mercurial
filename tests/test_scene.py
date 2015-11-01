__author__ = 'omar'
import sys

sys.path.insert(1, '../src')

from scene import Scene
from geometry import Size, Point

demo_file_name = '../scenes/demo_obstacle_list.json'
empty_file_name = '../scenes/empty_scene.json'


class TestScene:
    def __init__(self):
        self.scene_obj = Scene(size=Size([20, 20]), obstacle_file=demo_file_name,
                               initial_pedestrian_number=50)

    def test_create_cells(self):
        assert (self.scene_obj.cell_dict[(0, 0)].begin - Point([0, 0])).is_zero()
