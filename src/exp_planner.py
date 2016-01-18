__author__ = 'omar'

import numpy as np

import functions as ft
from geometry import Point, Velocity
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
        self.path_weights = np.array([0.6, 0.3, 0.05, 0.05])
        self.way_point_num = len(self.path_weights)
        self.path_points = np.zeros([self.scene.position_array.shape[0], self.way_point_num, 2])
        self.compute_init_path_points()

    def compute_init_path_points(self):
        for pedestrian in self.scene.pedestrian_list:
            for i in range(self.way_point_num):
                self.path_points[pedestrian.index, i] = pedestrian.path.get_sample_point(i)
            pedestrian.path_param = self.way_point_num - 1

    def compute_new_path_points(self):
        new_point_array = np.zeros((self.scene.position_array.shape[0], 2))
        index_array = []
        for pedestrian in self.scene.pedestrian_list:
            if self.needs_new_points(pedestrian):
                pedestrian.path_param += 1
                point = pedestrian.path.get_sample_point(pedestrian.path_param)
                new_point_array[pedestrian.index] = point
                index_array.append(pedestrian.index)
        new_path_points = np.roll(self.path_points, -1, 1)
        new_path_points[:, -1, :] = new_point_array
        self.path_points[index_array] = new_path_points[index_array]

    def compute_new_velocities(self):
        """
        Doing a 'naive' vectorised implementation.
        Speedups: Splitting the x and y and using dot products.
        :return:
        """
        weighted_points = np.sum(self.path_points * self.path_weights[None, :, None], axis=1)
        distances = self.scene.position_array - weighted_points
        self.scene.velocity_array[:] = -ft.normalize(distances, safe=True) * self.scene.max_speed_array[:, None]

    def needs_new_points(self, pedestrian):
        """
        We compute whether new points are necessary by using linear algebra
        We assume that the angle between the first waypoint and the last waypoint
        should have the same 'direction', i.e. a positive inner product.
        If this is not true, the last waypoint is no longer relevant and should be updated
        :param pedestrian: Pedestrian whose waypoints are under consideration
        :return: True if update is necessary, false otherwise
        """
        dist_to_path_points = self.path_points[pedestrian.index] - pedestrian.position.array
        inner_product = np.dot(dist_to_path_points[0], dist_to_path_points[self.way_point_num - 1])
        return inner_product <= 0

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
                allowed_range = 0.5 * self.scene.config['general'].getfloat(
                    'margin')  # some experimental threshold based on safety margin of obstacles
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
