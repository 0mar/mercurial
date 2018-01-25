from math_objects.geometry import Point, Velocity
from objects.waypoints import Waypoint
from populations.base import Population

import numpy as np
from math_objects import functions as ft
from lib.local_swarm import get_swarm_force
import json


class Following(Population):
    """
    Class of pedestrian individuals that follow everyone in a certain radius.
    This radius represents a sight and can be reduced due to smoke.
    Also the maximum speed can be reduced due to the stress of the smoke
    """

    def __init__(self, scene, number):
        """
        Initializes a following behaviour for the given population.

        :param scene: Simulation scene
        :param number: Initial number of people
        :return: Scripted pedestrian group
        """
        # Todo: Add the graphing of the necessary plots as on_step_functions
        super().__init__(scene, number)
        self.waypoints = []
        # self.waypoint_positions = self.waypoint_velocities = None
        self.follow_radii = self.speed_ref = None
        self.on_step_functions = []

    def prepare(self):
        """
        Called before the simulation starts. Fix all parameters and bootstrap functions.

        :return: None
        """
        super().prepare()
        # The size of the radius the pedestrian uses to follow others.
        self.follow_radii = np.ones(self.number) * self.params.follow_radius
        # A reference to the original maximum speed of the pedestrians
        self.speed_ref = np.array(self.scene.max_speed_array)
        if self.params.smoke:
            self.on_step_functions.append(self._reduce_sight_by_smoke)
            self.on_step_functions.append(self._modify_speed_by_smoke)
        self.on_step_functions.append(self.assign_velocities)

    def _load_waypoints(self, file_name="None"):
        """
        Not yet used, not yet implemented. Future research

        :return:
        """
        with open(file_name, 'r') as wp_file:
            data = json.loads(wp_file.read())
            if not 'waypoints' in data:
                return
        for entry in data['waypoints']:
            position = Point(self.scene.size * entry['position'])
            direction = Velocity([self.potential_x.ev(*position.array), self.potential_y.ev(*position.array)]) * -1
            waypoint = Waypoint(self.scene, position, direction)
            self.waypoints.append(waypoint)
            self.scene.drawables.append(waypoint)
        self.waypoint_positions = np.zeros([len(self.waypoints), 2])
        self.waypoint_velocities = np.zeros([len(self.waypoints), 2])
        for i, waypoint in enumerate(self.waypoints):
            self.waypoint_positions[i, :] = waypoint.position.array  # Waypoints are immutable, so no copy worries
            self.waypoint_velocities[i, :] = waypoint.direction.array * 20

    def _reduce_sight_by_smoke(self):
        """
        Compute how much the smoke influences the sight radius (self.follow_radii)
        :return:
        """
        smoke_on_positions = self.params.smoke_field.get_interpolation_function().ev(
            self.scene.position_array[self.indices, 0],
            self.scene.position_array[self.indices, 1])
        self.follow_radii = self.params.follow_radius * (1 - (1 - self.params.minimal_follow_radius) * np.minimum(
            np.maximum(smoke_on_positions, 0) / self.params.smoke_limit, 1))

    def _modify_speed_by_smoke(self):
        """
        Choosing velocity parameters as given by the Japanese paper.
        :return:
        """
        smoke_function = self.params.smoke_field.get_interpolation_function()
        smoke_on_positions = smoke_function.ev(self.scene.position_array[:, 0], self.scene.position_array[:, 1])
        velo_modifier = np.clip(smoke_on_positions / self.params.max_smoke_level, 0, 1 - self.params.min_speed_ratio)
        self.scene.max_speed_array = self.speed_ref * (1 - velo_modifier)

    def assign_velocities(self):
        """
        3. We don't change the velocity but the acceleration based on neighbour velocities
        dv_i/dt = sum_j{w_ij(v_i-v_j)}
        |dx_i/dt| = v_i
        """
        if self.waypoints:
            positions = np.vstack((self.scene.position_array[self.indices], self.waypoint_positions))
            velocities = np.vstack((self.scene.velocity_array[self.indices], self.waypoint_velocities))
            actives = np.hstack(
                (self.scene.active_entries[self.indices], np.ones(len(self.waypoints), dtype=bool)))
        else:
            positions = self.scene.position_array[self.indices]
            velocities = self.scene.velocity_array[self.indices]
            actives = self.scene.active_entries[self.indices]
        swarm_force = get_swarm_force(positions, velocities, self.scene.size[0],
                                      self.scene.size[1], actives, self.follow_radii)
        random_force = np.random.randn(len(self.indices), 2) * self.params.random_force
        # fire_rep_x = self.fire_force_field_x.ev(self.scene.position_array[:, 0], self.scene.position_array[:, 1])
        # fire_rep_y = self.fire_force_field_y.ev(self.scene.position_array[:, 0], self.scene.position_array[:, 1])
        # fire_repulsion = np.hstack([fire_rep_x[:,None],fire_rep_y[:,None]])
        self.scene.velocity_array[self.indices] += (swarm_force + random_force) * self.params.dt
        # Normalizing velocities
        self.scene.velocity_array[self.indices] *= self.scene.max_speed_array[
                                                       self.indices, None] / np.linalg.norm(
            self.scene.velocity_array[self.indices] + ft.EPS, axis=1)[:, None]

    def step(self):
        """
        Performs a planner step
        :return: None
        """
        [step() for step in self.on_step_functions]
