__author__ = 'omar'
import sys

sys.path.insert(1, '../src')

from objects.scene import Scene
from simulation_manager import SimulationManager

demo_file_name = '../scenes/demo_obstacle_list.json'
empty_file_name = '../scenes/empty_scene.json'


class TestScene:
    def __init__(self):
        config = SimulationManager.get_default_config()
        self.scene_obj = Scene(config=config,initial_pedestrian_number=50)

