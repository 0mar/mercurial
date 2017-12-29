import numpy as np
from objects.pedestrian import Pedestrian


class Population:
    """
    Base class that models a population.
    Can be overridden with populations following different rules
    """

    def __init__(self, scene, number):
        self.scene = scene
        self.number = number
        self.on_step_functions = []
        self.indices = np.zeros(self.number,dtype=int)
        self.scene.total_pedestrians += self.number

    def prepare(self):
        self._init_pedestrians()

    def _init_pedestrians(self):
        """
        Initializes the populations. Can be overridden for a different initial distribution
        or for different properties.

        Concatenates this population with the previous ones in the scene and computes the indices
         that correspond to this population.
        :return:
        """
        old_len = len(self.scene.pedestrian_list)
        for i in range(self.number):
            pedestrian = Pedestrian(self.scene,old_len+i)
            self.indices[i] = old_len + i
            self.scene.pedestrian_list.append(pedestrian)

    def step(self):
        """
        Compute all step functions for the population (like computing acceleration, velocities and positions).
        :return: None
        """
        [step() for step in self.on_step_functions]

    def find_finished(self):
        """
        Finds all pedestrians that have reached the goal in this time step.
        If there is a cap on the number of pedestrians allowed to exit,
        we randomly sample that number of pedestrians and leave the rest in the exit.
        If any pedestrians are unable to exit, we set the exit to inaccessible.
        The pedestrians that leave are processed and removed from the scene
        :return: None
        """
        cells = (self.scene.position_array // (self.scene.dx, self.scene.dy)).astype(int)
        in_goal = self.scene.env_field[cells[:, 0], cells[:, 1]] == 0
        finished = np.logical_and(in_goal, self.scene.active_entries)
        index_list = np.where(finished)[0]  # possible extension: Goal cap
        for index in index_list:
            finished_pedestrian = self.scene.index_map[index]
            self.scene.remove_pedestrian(finished_pedestrian)
