import scene as scene_module
import functions
from macro import MacroTransporter
from micro import PotentialTransporter


class Planner:
    def __init__(self, scene):
        self.scene = scene
        self.config = scene.config
        self.step_functions = []
        micro_planner = PotentialTransporter  # Name will be changed
        micro_dict = {'PotentialTransporter': PotentialTransporter}  # Remember boys: eval is evil!
        # dictionary with values of the local planners
        if 'micro' in self.config['planner']:
            micro_planner = micro_dict[self.config['planner']['micro']]
        self.micro = micro_planner(scene)

        macro_planner = MacroTransporter
        macro_dict = {'MacroTransporter': MacroTransporter, 'None': None}
        self.macro = None
        if 'global' in self.config['planner']:
            macro_planner = macro_dict[self.config['planner']['global']]
        if not macro_planner:
            self.macro = macro_planner(scene)

    def step(self):
        self.micro.assign_velocities()
        self.macro.step()
        self.micro.step()
        self.scene.find_finished_pedestrians()
