import math
import matplotlib.pyplot as plt
import numpy as np

from math_objects import functions as ft
from math_objects.geometry import Point, Size, Interval


class Fire:
    """
    Models a fire source in the domain. Fire sources are semipenetrable obstacles, circular.
    repelling on a large space scale.
    Walks and talks like an obstacle
    """

    def __init__(self, center, radius, intensity):
        """
        Fire constructor
        :param center: center of the fire source circle
        :param radius: Radius of the actual fire
        :param intensity: length scale of repulsion (may turn out to depend on radius)
        :return: new fire source.
        """
        self.center = center
        self.radius = radius
        self.intensity = intensity
        self.color = 'orange'
        self.accessible = True
        self.in_interior = True

    def __contains__(self, coord):
        return (coord[0] - self.center[0]) ** 2 + (coord[1] - self.center[1]) ** 2 < self.radius ** 2

    def __repr__(self):
        return "Fire with center: %s, radius %.2f" % (self.center, self.radius)

    def get_fire_intensity(self, point):
        """
        Compute the intensity for the fire for both repelling of pedestrian as for the creation of smoke
        :param point: Point for which the intensity is computed
        :return: The intensity of the fire as a double, based on the distance to the center
        """
        distance = np.sqrt((point[0] - self.center[0]) ** 2 + (point[1] - self.center[1]) ** 2) - self.radius
        return self.intensity * np.exp(-distance / self.radius)
