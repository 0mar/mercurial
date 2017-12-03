import json

import numpy as np
from lib.mde import compute_mde
from lib.wdt import map_image_to_costs
from math_objects.geometry import Point, Size, Interval
from objects.fire import Fire
from objects.pedestrian import Pedestrian


class Scene:
    """
    Models a scene. A scene is a rectangular object with obstacles and pedestrians inside.
    """

    def __init__(self, config):
        """
        Initializes a Scene using the settings in the configuration file augmented with command line parameters
        :param config: ConfigParser instance containing all settings required for a pedestrian simulation
        :return: scene instance.
        """
        self.time = 0
        self.counter = 0
        self.total_pedestrians = 0

        # Other drawable items than pedestrians and obstacles.
        self.drawables = []
        self.on_step_functions = []
        self.on_pedestrian_exit_functions = []
        self.on_pedestrian_init_functions = []

        # Parameter initialization (will be overwritten by _load_parameters)
        self.mde = self.use_exit_logs = self.minimal_distance = self.dt = self.size = None
        self.number_of_cells = self.pedestrian_size = self.max_speed_interval = None
        self.fire_center = self.fire_intensity = self.fire_radius = None
        self.aware_percentage = 0
        self.env_field = None

        self.config = config
        self.load_config()
        self.dx, self.dy = self.size.array / self.env_field.shape
        self.core_distance = self.minimal_distance  # Distance between ped centers
        self.fire = None
        if self.fire_center:
            self.fire = Fire(self.size * Point(self.fire_center), self.size[0] * self.fire_radius, self.fire_intensity)
            self.drawables.append(self.fire)
        # self.gutter_cells = self.get_obstacle_gutter_cells()
        # Array initialization
        self.position_array = np.zeros([self.total_pedestrians, 2])
        self.last_position_array = np.zeros([self.total_pedestrians, 2])
        self.velocity_array = np.zeros([self.total_pedestrians, 2])
        self.max_speed_array = np.empty(self.total_pedestrians)
        self.active_entries = np.ones(self.total_pedestrians, dtype=bool)

        self.aware_pedestrians = np.random.random(self.total_pedestrians) < self.aware_percentage
        # self.max_speed_array = self.max_speed_interval.begin + \
        #                        np.random.random(self.position_array.shape[0]) * self.max_speed_interval.length
        self.max_speed_array = np.maximum(np.random.randn(self.position_array.shape[0]) * 0.15 + 1.4,
                                          0.3)  # Todo: Fix in config file
        self.mde_proc = []
        self.pedestrian_list = []
        self._init_pedestrians(self.total_pedestrians)
        self.index_map = {i: self.pedestrian_list[i] for i in range(self.total_pedestrians)}
        self.status = 'RUNNING'

    def _init_pedestrians(self, init_number):
        """
        Protected method that determines how the pedestrians are initially distributed,
        as well as with what properties they come. Overridable.
        :param: Initial number of pedestrians
        :return: None
        """
        self.pedestrian_list = [
            Pedestrian(self, index) for index in range(init_number)]

    # def get_obstacle_gutter_cells(self, radius=2):
    #     """
    #     Compute all the cells which lie of distance `radius` from an obstacle.
    #     Used in the macroscopic pressure, in case we want to repel/attract pedestrians from/to specific zones.
    #     :param radius: distance in cells to the obstacles
    #     :return: array of nx,ny with 1 on gutter cells and obstacle cells and 0 on rest.
    #     """
    #     if not self.snap_obstacles:
    #         ft.warn("Computing obstacle coverage: Snapping is turned off, so this is only an estimation")
    #     dx, dy = self.config['general'].getfloat('cell_size_x'), self.config['general'].getfloat('cell_size_y')
    #     nx = int(self.config['general'].getint('scene_size_x') / dx)
    #     ny = int(self.config['general'].getint('scene_size_y') / dy)
    #     obstacle_gutter = np.zeros(self.obstacle_coverage.shape)
    #
    #     for row, col in np.ndindex((nx, ny)):
    #         if not self.obstacle_coverage[row, col]:
    #             right, up = min(row + radius + 1, nx - 1), min(col + radius + 1, ny - 1)
    #             left, down = max(row - radius, 0), max(col - radius, 0)
    #             if np.any(self.obstacle_coverage[left:right, down:up]):
    #                 obstacle_gutter[row, col] = 1
    #     return obstacle_gutter

    def load_config(self):
        """
        Interpret the ConfigParser object to read the parameters from
        :return: None
        """
        section = self.config['general']
        image_file = section['scene']
        self.total_pedestrians = section.getint('number_of_pedestrians')
        self.dt = float(section['dt'])
        self.minimal_distance = section.getfloat('minimal_distance')
        self.mde = section.getboolean('minimal_distance_enforcement')
        self.pedestrian_size = Size([section.getfloat('pedestrian_size'), section.getfloat('pedestrian_size')])
        self.size = Size([section.getfloat('scene_size_x'), section.getfloat('scene_size_y')])
        self.max_speed_interval = Interval([section.getfloat('max_speed_begin'), section.getfloat('max_speed_end')])
        self.aware_percentage = self.config['aware'].getfloat('percentage', fallback=1.0)
        if self.config.has_section('fire'):
            # TODO: Move fire location from config to scene
            fire_config = self.config['fire']
            self.fire_center = (fire_config.getfloat('center_x'), fire_config.getfloat('center_y'))
            self.fire_intensity = fire_config.getfloat('intensity')
            self.fire_radius = fire_config.getfloat('radius')
        self.env_field = np.rot90(map_image_to_costs(image_file), -1)

    def _expand_arrays(self):
        """
        Increases the size (first dimension) of a numpy array with the given factor.
        Missing entries are set to zero
        :return: None
        """
        # I don't like [gs]etattr, but this is pretty explicit
        # Todo: Is it possible to to make one method that expands all arrays application-wide?
        attr_list = ["position_array", "last_position_array", "velocity_array",
                     "max_speed_array", "active_entries", "aware_pedestrians"]
        for attr in attr_list:
            array = getattr(self, attr)
            addition = np.zeros(array.shape, dtype=array.dtype)
            setattr(self, attr, np.concatenate((array, addition), axis=0))
        self.index_map.update({len(self.index_map) + i: None for i in range(len(self.index_map))})

    def get_stationary_pedestrians(self):
        """
        Computes which pedestrians have not moved since the last time step
        :return: nx1 boolean np.array, True if (existing) pedestrian is stationary, False otherwise
        """
        pos_difference = np.linalg.norm(self.position_array - self.last_position_array, axis=1)
        not_moved = np.logical_and(pos_difference == 0, self.active_entries)
        return not_moved

    def remove_pedestrian(self, pedestrian):
        """
        Removes a pedestrian from the scene.
        :param pedestrian: The pedestrian instance to be removed.
        :return: None
        """
        index = pedestrian.index
        self.index_map[index] = None
        self.pedestrian_list.remove(pedestrian)

        self.active_entries[index] = False
        for func in self.on_pedestrian_exit_functions:
            func(pedestrian)
        if self.is_done():
            self.status = 'DONE'

    def is_within_boundaries(self, coord: Point):
        """
        Check whether a single point lies within the scene.
        :param coord: Point under consideration
        :return: True if within scene, false otherwise
        """
        within_boundaries = all(np.array([0, 0]) < coord.array) and all(coord.array < self.size.array)
        return within_boundaries

    def is_accessible(self, coord: Point, at_start=False):
        """
        Checking whether the coordinate present is an accessible coordinate on the scene.
        When evaluated at the start, the exit is not an accessible object,
        to prevent pedestrians from being spawned there.
        :param coord: Coordinates to be checked
        :param at_start: Whether to be evaluated at the start
        :return: True if accessible, False otherwise.
        """
        if not self.is_within_boundaries(coord):
            return False
        cell = (int(coord[0] // self.dx), int(coord[1] // self.dy))
        if at_start:
            return 0 < self.env_field[cell] < np.inf  # Todo: Rather, you want to check the potential field
        else:
            return self.env_field[cell] < np.inf

    def step(self):
        """
        Compute all step functions in scene not related to planner functions.
        :return: None
        """
        [step() for step in self.on_step_functions]

    def move_pedestrians(self):
        """
        Performs a vectorized move of all the pedestrians.
        Assumes that all the velocities have been set accordingly.
        :return: None
        """
        self.time += self.dt
        self.counter += 1
        self.last_position_array = np.array(self.position_array)
        self.position_array += self.velocity_array * self.dt
        if self.mde:
            mde = compute_mde(self.position_array, self.size[0], self.size[1],
                              self.active_entries, self.core_distance)
            mde_found = np.where(np.sum(np.abs(mde[self.active_entries]), axis=1) > 0.001)[0]
            self.mde_proc.append(len(mde_found) / np.sum(self.active_entries))
            # print("mde percentage: %.4f" % (sum(self.mde_proc) / len(self.mde_proc)))
            self.position_array += mde

    def correct_for_geometry(self):
        """
        Performs a vectorized correction to make sure pedestrians do not run into walls
        :return: None
        """
        geq_zero = np.logical_and(self.position_array[:, 0] > 0, self.position_array[:, 1] > 0)
        leq_size = np.logical_and(self.position_array[:, 0] < self.size[0], self.position_array[:, 1] < self.size[1])
        still_correct = np.logical_and(geq_zero, leq_size)
        cells = (self.position_array // (self.dx, self.dy)).astype(int) % self.env_field.shape
        # All peds for which the modulo kicks in, are already out
        not_in_obstacle = self.env_field[cells[:, 0], cells[:, 1]] < np.inf
        still_correct = np.logical_and(still_correct, not_in_obstacle)
        still_correct = np.logical_and(still_correct, self.active_entries)
        self.position_array += np.logical_not(still_correct)[:, None] * (self.last_position_array - self.position_array)

    def find_finished_pedestrians(self):
        """
        Finds all pedestrians that have reached the goal in this time step.
        If there is a cap on the number of pedestrians allowed to exit,
        we randomly sample that number of pedestrians and leave the rest in the exit.
        If any pedestrians are unable to exit, we set the exit to inaccessible.
        The pedestrians that leave are processed and removed from the scene
        :return: None
        """
        cells = (self.position_array // (self.dx, self.dy)).astype(int)
        in_goal = self.env_field[cells[:, 0], cells[:, 1]] == 0
        finished = np.logical_and(in_goal, self.active_entries)
        index_list = np.where(finished)[0]  # possible extension: Goal cap
        for index in index_list:
            finished_pedestrian = self.index_map[index]
            self.remove_pedestrian(finished_pedestrian)
        if self.counter > 5000:
            self.status = 'DONE'  # TODO: Move to config

    def is_done(self):
        """
        Checks whether all pedestrians are done by summing the active entries array and
        checking if all entrances are depleted.
        :return: True if all pedestrians are done, False otherwise
        """
        return not np.any(self.active_entries)  # and all([entrance.depleted for entrance in self.entrance_list])
