import numpy as np
from objects.pedestrian import Pedestrian


class Population:
    """
    Base class that models a population.
    Todo: Perhaps merge with behaviour class? Or make it override this one.

    Can be overridden with populations following different rules
    """

    def __init__(self, scene, number):
        self.scene = scene
        self.number = number
        self.indices = np.zeros(self.number, dtype=int)
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
            pedestrian = Pedestrian(self.scene, old_len + i)
            self.indices[i] = old_len + i
            self.scene.pedestrian_list.append(pedestrian)
            self.scene.index_map[old_len + i] = pedestrian
