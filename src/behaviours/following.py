from math_objects.geometry import Point, Velocity
from objects.waypoints import Waypoint

import numpy as np
from math_objects import functions as ft
from lib.local_swarm import get_swarm_force
import json

import params


class Following:
    """
    Class of pedestrian individuals that follow everyone in a certain radius.
    This radius represents a sight and can be reduced due to smoke.
    """

    def __init__(self, population):
        """
        Initializes a following behaviour for the given population.

        :param population: The group of people the behaviour is imposed upon
        :return: Scripted pedestrian group
        """
        # Todo: Add the graphing of the necessary plots as on_step_functions
        self.population = population
        self.scene = population.scene
        self.waypoints = []
        # self.waypoint_positions = self.waypoint_velocities = None
        self.follow_radii = None
        self.on_step_functions = []

    def prepare(self):
        """
        Called before the simulation starts. Fix all parameters and bootstrap functions.

        :return: None
        """
        self.population.prepare()
        self.follow_radii = np.ones(self.population.number) * params.follow_radius
        if hasattr(self.scene, 'smoke_field'): # Probably there is a nicer way
            self.on_step_functions.append(self._reduce_sight_by_smoke)
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
            self.waypoint_velocities[i, :] = waypoint.direction.array*20

    def _reduce_sight_by_smoke(self):
        """
        Todo: Integrate
        :return:
        """
        smoke_on_positions = self.scene.smoke_field.get_interpolation_function().ev(self.scene.position_array[self.population.indices, 0],
                                                                                    self.scene.position_array[self.population.indices, 1])
        self.follow_radii = params.follow_radius * (1 - (1 - params.minimal_follow_radius) *
                                                    np.minimum(np.maximum(smoke_on_positions, 0) / params.smoke_limit,
                                                               1))

    def assign_velocities(self):
        """
        3. We don't change the velocity but the acceleration based on neighbours
        dv_i/dt = sum_j{w_ij(v_i-v_j)}
        |dx_i/dt| = v_i
        """
        if self.waypoints:
            positions = np.vstack((self.scene.position_array[self.population.indices], self.waypoint_positions))
            velocities = np.vstack((self.scene.velocity_array[self.population.indices], self.waypoint_velocities))
            actives = np.hstack((self.scene.active_entries[self.population.indices], np.ones(len(self.waypoints), dtype=bool)))
        else:
            positions = self.scene.position_array[self.population.indices]
            velocities = self.scene.velocity_array[self.population.indices]
            actives = self.scene.active_entries[self.population.indices]
        swarm_force = get_swarm_force(positions, velocities, self.scene.size[0],
                                      self.scene.size[1], actives, self.follow_radii)
        random_force = np.random.randn(len(self.population.indices),2) * params.random_force
        # fire_rep_x = self.fire_force_field_x.ev(self.scene.position_array[:, 0], self.scene.position_array[:, 1])
        # fire_rep_y = self.fire_force_field_y.ev(self.scene.position_array[:, 0], self.scene.position_array[:, 1])
        # fire_repulsion = np.hstack([fire_rep_x[:,None],fire_rep_y[:,None]])
        self.scene.velocity_array[self.population.indices] += (swarm_force + random_force) * params.dt
        # Normalizing velocities
        self.scene.velocity_array[self.population.indices] *= self.scene.max_speed_array[self.population.indices, None] / np.linalg.norm(self.scene.velocity_array[self.population.indices] + ft.EPS, axis=1)[:, None]

    def step(self):
        """
        Performs a planner step
        :return: None
        """
        [step() for step in self.on_step_functions]
