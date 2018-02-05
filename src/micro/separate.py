import numpy as np
import params
from lib.mde import compute_mde


class Separate:
    """
    Implements a Minimal Distance Enforcement (MDE). Using this class, any pedestrians that violate a minimal distance can be separated (locally)
    """

    def __init__(self, scene):
        """
        Applies minimal distance enforcements among all agents.
        If wanted, we can log the violations.
        :param scene: current scene
        """
        self.scene = scene
        self.params = None
        self.store_violations = False
        self.violations = []
        self.mde = None
        self.on_step_functions = []

    def prepare(self, params):
        """
        Called before the simulation starts. Fix all parameters and bootstrap functions.

        :params: Parameter object
        :return: None
        """
        self.params = params
        self.on_step_functions.append(self.separate)
        if self.store_violations:
            self.on_step_functions.append(self.compute_violations)

    def separate(self):
        """
        Separation (performed in the fortran module). The necessary corrections are computed and applied.
        :return:
        """
        self.mde = compute_mde(self.scene.position_array, self.scene.size[0], self.scene.size[1],
                               self.scene.active_entries, self.params.minimal_distance)
        self.scene.position_array += self.mde

    def compute_violations(self):
        """
        In case we have different methods of enforcing distance (like a crowd pressure machine),
        we can measure if it satisfies the minimal distances.
        :return:
        """
        mde_found = np.where(np.sum(np.abs(self.mde[self.scene.active_entries]), axis=1) > 0.001)[0]
        self.violations.append(len(mde_found) / np.sum(self.scene.active_entries))

    def step(self):
        [step() for step in self.on_step_functions]
