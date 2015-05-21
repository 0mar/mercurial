#!/usr/bin/env python
import random

from functions import *
from geometry import Point, Size, Velocity, Interval


__author__ = 'omar'


class Pedestrian(object):
    """
    Class for modeling a pedestrian. Apart from physical properties like position and velocity,
    this class also contains agent properties like planned path, (NYI:) goal, social forces,
    and maximum speed.
    """
    def __init__(self, scene, counter, goal, position=Point([0, 0]), color=None):
        """
        Initializes the pedestrian
        :param scene: Scene instance for the pedestrian to walk in
        :param counter: number of the pedestrian in the scene list. Not relied upon currently
        :param goal: Goal obstacle (Exit instance)
        :param position: Current position of the pedestrian. Position [0,0] will create a randomized position
        :param color: Color the pedestrian should be drawn with. Currently random, can be changed to
        observe effect of social forces.
        :return: Pedestrian instance
        """
        self.scene = scene
        self._position = position
        self.counter = counter
        self.size = Size(np.array([1.0, 1.0]))
        self.color = color
        self.max_speed = Interval([3, 10]).random()
        self._velocity = None
        self.goal = goal
        self.radius = 10
        while self._position.is_zero():
            new_position = scene.size.internal_random_coordinate()
            if scene.is_accessible(new_position, at_start=True):
                self._position = new_position
        if not scene.is_accessible(self.position):
            warn("Ped %s has no accessible coordinates. Check your initialization" % self)
        self.origin = self.position

    def __str__(self):
        if self._velocity:
            return "Moving pedestrian %d\tPosition: %s\tAngle %.2f pi" % \
               (self.counter, self._position, self._velocity.angle / np.pi)
        else:
            return "Standing pedestrian %d\tPosition: %s" % \
               (self.counter, self._position)

    def __repr__(self):
        return "Instance: Pedestrian#%d" % self.counter

    def update_position(self, dt):
        """
        Direct updating of the pedestrian position by moving to x+v*dt
        Could be made private, but that depends on the planner implementation.
        Currently, it is not checked whether the position is available, e.g. free from obstacles
        or other pedestrians. This is because we use a path that does not lead into other obstacles,
        and other pedestrians are ignored. If we implement the fluid model, we should perform a check on this
        :param dt: Time step
        :return: None
        """
        new_position = self._position + self._velocity * dt
        # if self.scene.is_accessible(new_position): # We can only remove this because of the graph planner.
        self.position = new_position

    def move_to_position(self, position: Point, dt):
        """
        Higher level updating of the pedestrian. Checks whether the position is reachable this time step.
        If so, moves to that position. Directly attaining the position within the current radius enables us
        to be less numerically accurate with the velocity directing to the goal.
        This has not yet proven to be a problem, luckily.
        :param position: Position that should be attained
        :param dt: time step
        :return: True when position is attained, false otherwise
        """
        distance = position - self.position
        if np.linalg.norm(distance.array) < self.max_speed * dt: # should be enough to avoid small numerical error
            if self.scene.is_accessible(position): # Might be removed, but is barely called.
               self.position = position
               return True
            else:
                warn("%s is directed into stuff" % self)
                return False
        else:
            if not self.velocity:
                self.velocity = Velocity(distance) # Consumes 6/19 of move_to_position time
            self.update_position(dt)
            return False

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        """
        Sets velocity value and automatically rescales it to maximum speed.
        This is an assumption that can be dropped when we implement the UIC
        :param value: 2D velocity with the correct direction
        :return: None
        """
        self._velocity = value
        if value:
            self._velocity.rescale(self.max_speed)

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, point):
        self._position = point

    def is_done(self):
        """
        Determines whether the pedestrian has reached its goal.
        Provides a warning when it has left the scene without exiting through its exit object.
        :return: True when the pedestrian has left the scene, false otherwise.
        """
        if self.position in self.goal:
            return True
        elif any(self.position.array < 0) or any(self.position.array > self.scene.size.array):
            warn("Dirty exit of %s, leaving on %s" % (self, self.position))
            return True
        else:
            return False