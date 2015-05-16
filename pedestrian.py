#!/usr/bin/env python
from functions import *

__author__ = 'omar'


class Pedestrian(object):
    def __init__(self, scene, counter, goal, position=Point([0,0]), color=None):
        self.scene = scene
        self._position = position
        self.counter = counter
        self.size = Size(np.array([1.0, 1.0]))
        self.color = color
        self.max_speed = Interval([3, 10]).random()
        self._velocity = Velocity([random.random() - 0.5, random.random() - 0.5])  # dangerous
        self._velocity.rescale(self.max_speed)
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
        return "Pedestrian %d\tPosition: %s\tAngle %.2f pi" % \
               (self.counter, self._position, self._velocity.angle / np.pi)

    def __repr__(self):
        return "Instance: Pedestrian#%d" % self.counter

    def update_position(self, dt):
        new_position = self._position + self._velocity * dt
        if self.scene.is_accessible(new_position):
            self.position = new_position

    # Todo: Profile the pedestrian methods.
    def move_to_position(self, position: Point, dt):
        distance = position - self.position
        if np.linalg.norm(distance.array) < self.max_speed * dt:
            if self.scene.is_accessible(position):
                self.position = position
                return True
            else:
                warn("%s is directed into stuff"%self)
                return False
        else:
            self.velocity = Velocity(distance)
            self.update_position(dt)
            return False

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        self._velocity = value
        self._velocity.rescale(self.max_speed)

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, point):
        self._position = point

    def is_done(self):
        if self.position in self.goal:
            return True
        elif any(self.position.array < 0) or any(self.position.array > self.scene.size.array):
            warn("Dirty exit of %s, leaving on %s" % (self, self.position))
            return True
        else:
            return False