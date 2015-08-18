#!/usr/bin/env python

from functions import *
from geometry import Point, Size, Interval

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
        self.counter = counter
        self._position = self._velocity = None
        self.position = position
        self.size = Size(np.array([0.5, 0.5]))
        self.color = color
        self.max_speed = Interval([1,2]).random()
        self.goal = goal
        self.cell = None
        while self.position.is_zero() and type(self) == Pedestrian:
            new_position = scene.size.random_internal_point()
            self.manual_move(new_position,at_start=True)
        if not scene.is_accessible(self.position) and type(self) == Pedestrian:
            warn("Ped %s has no accessible coordinates. Check your initialization" % self)
        self.origin = self.position
        self.scene.position_array[self.counter] = self._position.array

    def __str__(self):
        return "Moving pedestrian %d\tPosition: %s\tAngle %.2f pi" % \
               (self.counter, self.position, self._velocity.angle / np.pi)

    def __repr__(self):
        return "Pedestrian#%d" % self.counter

    def update_position(self):
        """
        Updates the position of the pedestrian from the scene position array
        by checking its accessibility in the corresponding cell.
        If the position is not accessible, the pedestrian does not move
        and the scene position array entry is reset.
        :return: None
        """
        new_point = Point(self.scene.position_array[self.counter])
        if self.scene.is_within_boundaries(new_point) and self.cell.is_accessible(new_point):
            self._position = new_point
        else:
            self.scene.position_array[self.counter] = self._position.array

    def manual_move(self, position,at_start = False):
        """
        Move the pedestrian to the give position manually; independent on the scene position array.
        :param position: new pedestrian position (will still be checked)
        :param at_start: Time of moving
        :return: True when move is allowed and executed, false otherwise
        """
        # Should a whole scene check take too much time, then this should be replaced
        if self.scene.is_accessible(position,at_start):
            self.position = position
            self.scene.position_array[self.counter] = self.position.array
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
        if np.linalg.norm(distance.array) < self.max_speed * dt:  # should be enough to avoid small numerical error
            moved_to_position = self.manual_move(position)
            return moved_to_position
        else:
            return False

    @property
    def velocity(self):
        # return Velocity(self.scene.velocity_array[self.counter])
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
            self.scene.velocity_array[self.counter] = self._velocity.array

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


class EmptyPedestrian(Pedestrian):
    def __init__(self, scene, counter):
        super(EmptyPedestrian, self).__init__(scene=scene, counter=counter, goal=None)

    def is_done(self):
        return True

    def __repr__(self):
        return "DonePedestrian#%d" % self.counter

    def __str__(self):
        return "Finished Pedestrian %d" % self.counter

    def update_position(self, dt):
        pass
