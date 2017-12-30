import numpy as np
import params
from lib.mde import compute_mde


class Separate:
    """
    MDE class. With MDE Extensions like logging violations and stuff
    """

    def __init__(self,scene):
        self.scene = scene
        self.store_violations=False
        self.violations = []
        self.mde = None
        self.on_step_functions = []

    def prepare(self):
        self.on_step_functions.append(self.separate)
        if self.store_violations:
            self.on_step_functions.append(self.compute_violations)

    def separate(self):
        print(self.scene.position_array)
        self.mde = compute_mde(self.scene.position_array, self.scene.size[0], self.scene.size[1],
                               self.scene.active_entries, params.minimal_distance)
        self.scene.position_array += self.mde

    def compute_violations(self):
        mde_found = np.where(np.sum(np.abs(self.mde[self.scene.active_entries]), axis=1) > 0.001)[0]
        self.violations.append(len(mde_found) / np.sum(self.scene.active_entries))

    def step(self):
        [step() for step in self.on_step_functions]