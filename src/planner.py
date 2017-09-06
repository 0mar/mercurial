from macro.pressure import PressureTransporter
from macro.smoke import Smoker
from micro.potential import PotentialTransporter
from micro.potential_and_awareness import PotentialInterpolator


class Planner:
    def __init__(self, scene):
        self.scene = scene
        self.config = scene.config
        self.step_functions = []
        micro_planner = PotentialTransporter  # Name will be changed
        micro_dict = {'PotentialTransporter': PotentialTransporter,
                      'PotentialInterpolator': PotentialInterpolator }  # Remember boys: eval is evil!
        # dictionary with values of the local planners
        if 'micro' in self.config['planner']:
            micro_planner = micro_dict[self.config['planner']['micro']]
        self.micro = micro_planner(scene)

        macro_planner = PressureTransporter
        macro_dict = {'PressureTransporter': PressureTransporter, 'None': None}
        self.macro = None
        if 'macro' in self.config['planner']:
            macro_planner = macro_dict[self.config['planner']['macro']]
        if macro_planner:
            self.macro = macro_planner(scene)
        if self.scene.fire:
            self.smoke = Smoker(self.scene)

    def step(self):
        self.micro.assign_velocities()
        self.macro.step()
        if self.scene.fire:
            self.smoke.step()
        self.micro.step()
        self.scene.find_finished_pedestrians()
