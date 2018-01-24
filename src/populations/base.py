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
        self.indices = []
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
            self.indices.append(old_len + i)
            self.scene.pedestrian_list.append(pedestrian)
            self.scene.index_map[old_len + i] = pedestrian

    def create_new_pedestrian(self):
        index = np.where(self.scene.active_entries == 0)[0][0]
        new_pedestrian = Pedestrian(self.scene, counter=self.scene.total_pedestrians, index=index)
        self.scene.add_pedestrian(new_pedestrian)
        self.indices.append(index)
