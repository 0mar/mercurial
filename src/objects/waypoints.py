import numpy as np

from objects.pedestrian import Pedestrian
from math_objects.geometry import Point, Velocity


class Waypoint(Pedestrian):
    """
    Class for modelling waypoints that help follower pedestrians to reach their goals.
    Because of their similarity to pedestrian objects, implemented as an extension of the Pedestrian class.
    """

    def __init__(self, scene, position, direction):
        super().__init__(scene=scene, counter=-1, goals=[], position=position)
        self.direction = direction
        self.center = position
        # Visual properties
        self.radius = self.scene.config['general'].getfloat('pedestrian_size') * 2
        self.color = "#%02x%02x%02x" % (0, 200, 0)  # Green

    @property
    def velocity(self):
        return self.direction

    @velocity.setter
    def velocity(self, value):
        self.direction = value

    @property
    def position(self):
        return self.center

    @position.setter
    def position(self, point):
        self.center = point
