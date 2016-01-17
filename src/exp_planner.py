__author__ = 'omar'

import networkx as nx
import numpy as np

import functions as ft
from geometry import LineSegment, Path, Point, Interval, Velocity, Coordinate
from scene import Obstacle
from static_planner import GraphPlanner

class ExponentialPlanner(GraphPlanner):
    """
    Even Further Upgraded path planner, based on A* algorithm.
    Converts the scene to a graph, adds the pedestrians location and finds the shortest path to the nearest goal.

    Weights the points on the graph and creates smooth paths.
    Note that this only works for non-entry scenes (but easy to fix)
    """

    def __init__(self, scene):
        super().__init__(scene)
        self.path_weights = np.array([0.4,0.3,0.2,0.1])
        self.path_points = np.zeros([self.scene.position_array.shape[0],self.path_weights.shape[1]])
        self.path_dt = 1
        for pedestrian in self.scene.pedestrian_list:
            pedestrian.path_param = 0


    def compute_new_path_points(self):
        new_point_array = np.zeros((self.scene.position_array.shape[0],2))
        index_array = []
        for pedestrian in self.scene.pedestrian_list:
            if self.needs_new_points(pedestrian):
                pedestrian.param += self.path_dt
                point = pedestrian.path.get(pedestrian.param)
                new_point_array[pedestrian.index] = point.array
                index_array.append(pedestrian.index)
        pass
        # copy data into other array
        # roll axis and add new array
        # Copy on places where index changed

    def compute_new_velocities(self):
        weighted_points = self.path_weights[None,:]*self.path_points
        distances = self.scene.position_array - weighted_points
        # Compute angles with atan2
        # Adapt velocity array

    def step(self):
        """
        The update step of the static planner simulation
        # 1: Moves the pedestrians according to desired velocity, group velocity and UIC
                * Account for MDE
        # 2: Corrects for obstacles and walls
        # 3: Check all pedestrians if arrived at location
        # 3.5: Check if path is still correct (using last location and such)
        # 4: Provides the new desired velocity.
        # 5: Enforce minimal distance between pedestrians.
        :return: None
        """
        # 1
        self.compute_new_path_points()
        self.compute_new_velocities()
        self.scene.move_pedestrians()

        # 2
        self.scene.correct_for_geometry()

        stationary_pedestrian_array = self.scene.get_stationary_pedestrians()
        for pedestrian in self.scene.pedestrian_list:
            if not stationary_pedestrian_array[pedestrian.index] and hasattr(pedestrian, 'line'):
                # assert self.scene.is_accessible(pedestrian.position)
                pedestrian.move_to_position(Point(pedestrian.line.end), self.scene.dt)
                remaining_path = pedestrian.line.end - pedestrian.position
                allowed_range = 0.5  # some experimental threshold based on safety margin of obstacles
                checkpoint_reached = ft.norm(remaining_path[0], remaining_path[1]) < allowed_range
                if checkpoint_reached:  # but not done
                    if pedestrian.path:
                        pedestrian.line = pedestrian.path.pop_next_segment()
                        pedestrian.velocity = Velocity(pedestrian.line.end - pedestrian.position)
                else:
                    pedestrian.velocity = Velocity(remaining_path)  # Expensive...
            else:
                # Stationary pedestrian or new. Creating new path.
                pedestrian.path = self.create_path(pedestrian)
                pedestrian.line = pedestrian.path.pop_next_segment()
                pedestrian.velocity = Velocity(pedestrian.line.end - pedestrian.position.array)
        self.scene.find_finished_pedestrians()

