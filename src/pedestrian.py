#!/usr/bin/env python

import math

import numpy as np

import functions as ft
from geometry import Point, Velocity

__author__ = 'omar'


class Pedestrian(object):
    """
    Class for modeling a pedestrian. Apart from physical properties like position and velocity,
    this class also contains agent properties like planned path, (NYI:) goal, social forces,
    and maximum speed.
    """

    def __init__(self, scene, counter, goals, size, max_speed, position=Point([0, 0]), index=-1):
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
        self.counter = counter
        if index < 0:
            self.index = counter
        else:
            self.index = index
        self._position = None
        self._velocity = Velocity([0, 0])
        self.position = position
        self.size = size
        self.max_speed = max_speed
        self.color = self._convert_speed_to_color()
        self.goals = goals
        self.cell = None
        while self.position.is_zero() and type(self) == Pedestrian:
            new_position = scene.size.random_internal_point()
            self.manual_move(new_position, at_start=True)
        if not scene.is_accessible(self.position) and type(self) == Pedestrian:
            ft.warn("%s has no accessible coordinates. Check your initialization" % self)
        self.origin = self.position
        self.scene.position_array[self.index] = self._position.array

    def __str__(self):
        """
        String representation for pedestrian having position, angle,
        and whether or not pedestrian is still in the scene
        :return: string representation
        """
        return "Pedestrian %d (index %d) \tPosition: %s\tAngle %.2f pi" % \
               (self.counter, self.index, self.position, self._velocity.angle / np.pi)

    def __repr__(self):
        """
        :return: Unique string identifier using counter integer
        """
        return "Pedestrian#%d (%d)" % (self.counter, self.index)

    def _convert_speed_to_color(self):
        """
        Computes a color between red and blue based on pedestrian max velocity.
        Red is fast, blue is slow.
        :return: tkinter RGB code for color
        """
        start = np.min(self.scene.max_speed_array)
        end = np.max(self.scene.max_speed_array)
        if start == end:
            return 'blue'
        max_val = 255
        speed = self.max_speed
        red = max_val * (speed - start) / (end - start)
        green = 0
        blue = max_val * (speed - end) / (start - end)
        return "#%02x%02x%02x" % (red, green, blue)

    def correct_for_geometry(self):
        """
        Updates the position of the pedestrian from the scene position array
        by checking its accessibility in the corresponding cell.
        If the position is not accessible, the pedestrian does not move
        and the scene position array entry is reset.
        :return: None
        """
        new_point = Point(self.scene.position_array[self.index])
        if self.scene.is_within_boundaries(new_point) and self.cell.is_accessible(new_point):
            self._position = new_point
        else:
            self.scene.position_array[self.index] = self._position.array

    def manual_move(self, position, at_start=False):
        """
        Move the pedestrian to the give position manually; independent on the scene position array.
        :param position: new pedestrian position (will still be checked)
        :param at_start: Time of moving
        :return: True when move is allowed and executed, false otherwise
        """
        # Should a whole scene check take too much time, then this should be replaced
        if self.scene.is_accessible(position, at_start):
            self.position = position
            self.scene.position_array[self.index] = self.position.array
            if self.cell:
                self.cell.remove_pedestrian(self)
                self.scene.get_cell_from_position(self.position).add_pedestrian(self)
            return True
        return False

    def move_to_position(self, position: Point, dt):
        """
        Higher level updating of the pedestrian. Checks whether the position is reachable this time step.
        If so, moves to that position. Directly attaining the position within the current radius enables us
        to be less numerically accurate with the velocity directing to the goal.
        :param position: Position that should be attained
        :param dt: time step
        :return: True when position is attained, false otherwise
        """
        distance = position - self.position
        if math.sqrt(distance.array[0] ** 2 + distance.array[
            1] ** 2) < self.max_speed * dt:  # should be enough to avoid small numerical error
            moved_to_position = self.manual_move(position)
            return moved_to_position
        else:
            return False

    @property
    def velocity(self):
        """
        Velocity getter
        :return:
        """
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
            self.scene.velocity_array[self.index] = self._velocity.array

    @property
    def position(self):
        """
        Position getter
        :return: Current position
        """
        return self._position

    @position.setter
    def position(self, point):
        """
        Position setter. Eases checks and debugging.
        Note that setting a point this way does not update the scene position array.
        Use manual_move() for that.
        :param point: New position
        :return: None
        """
        self._position = point

    def is_done(self):
        """
        Determines whether the pedestrian has reached its goal.
        Provides a warning when it has left the scene without exiting through its exit object.
        :return: True when the pedestrian has left the scene, false otherwise.
        """
        for goal in self.goals:
            if self.position in goal:
                goal.log_pedestrian(self, self.scene.time)
                return True
        if any(self.position.array < 0) or any(self.position.array > self.scene.size.array):
            ft.warn("Dirty exit of %s, leaving on %s" % (self, self.position))
            return True
        else:
            return False