import json

import numpy as np
from fortran_modules.mde import compute_mde

from math_objects import functions as ft
from math_objects.geometry import Point, Size, Interval
from objects.obstacles import Obstacle, Entrance, Exit
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

        self.obstacle_list = []
        self.exit_list = []
        self.entrance_list = []
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
        self.snap_obstacles = None

        self.config = config
        self.load_config()
        self.core_distance = self.minimal_distance + self.pedestrian_size[0]  # Distance between ped centers
        self.fire = None
        if self.fire_center:
            self.fire = Fire(self.size * Point(self.fire_center), self.size[0] * self.fire_radius, self.fire_intensity)
            self.drawables.append(self.fire)
        self.obstacle_coverage = self.get_obstacles_coverage()
        self.gutter_cells = self.get_obstacle_gutter_cells()
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
            Pedestrian(self, index, goals=self.exit_list) for index in range(init_number)]

    def create_new_pedestrians(self):
        """
        Creates new pedestrians according to the entrance data.
        New indices are computed for these pedestrians and if necessary,
        the arrays are increased.
        :return: None
        """
        for entrance in self.entrance_list:
            new_number = entrance.get_new_number_of_pedestrians(self.time)
            max_tries = 100 * new_number
            tries = 0  # when this becomes large, chances are the entrance can produce no valid new position.
            while new_number > 0:
                tries += 1
                new_position = entrance.get_spawn_location()
                if not self.is_accessible(new_position):
                    if tries > max_tries:
                        raise RuntimeError("Can't find new spawn locations for %s. Check the scene" % entrance)
                    continue
                free_indices = np.where(self.active_entries == False)[0]  # Not faster with np.not
                if len(free_indices):
                    new_index = free_indices[0]
                else:
                    new_index = self.active_entries.shape[0]
                    self._expand_arrays()
                    ft.debug("Indices full. doubling array size to %d" % (2 * new_index))
                new_max_speed = self.max_speed_interval.random()
                self.max_speed_array[new_index] = new_max_speed
                new_pedestrian = Pedestrian(self, self.total_pedestrians, self.exit_list, new_position, new_index)
                self.total_pedestrians += 1
                self.active_entries[new_index] = True
                self.pedestrian_list.append(new_pedestrian)
                self.index_map[new_index] = new_pedestrian
                [func(new_pedestrian) for func in self.on_pedestrian_init_functions]
                new_number -= 1

    def get_obstacles_coverage(self):
        """
        Compute which obstacles are occupied. For accurate results, obstacles must have been snapped
        ergo, self.snap_obstacles == True
        :return: Array with ones for cells under obstacles.
        """
        if not self.snap_obstacles:
            ft.warn("Computing obstacle coverage: Snapping is turned off, so this is only an estimation")
        dx, dy = self.config['general'].getfloat('cell_size_x'), self.config['general'].getfloat('cell_size_y')
        nx = int(self.config['general'].getint('scene_size_x') / dx)
        ny = int(self.config['general'].getint('scene_size_y') / dy)
        obstacle_coverage = np.zeros((nx, ny), dtype=int)
        for row, col in np.ndindex((nx, ny)):
            center = Point([(row + 0.5) * dx, (col + 0.5) * dy])
            for obstacle in self.obstacle_list:
                if not obstacle.accessible:
                    if center in obstacle:
                        obstacle_coverage[row, col] = 1
        return obstacle_coverage

    def get_obstacle_gutter_cells(self, radius=2):
        """
        Compute all the cells which lie of distance `radius` from an obstacle.
        Used in the pressure determination, in case we want to repel/attract pedestrians from/to specific zones.
        :param radius: distance in cells to the obstacles
        :return: array of nx,ny with 1 on gutter cells and obstacle cells and 0 on rest
        """
        if not self.snap_obstacles:
            ft.warn("Computing obstacle coverage: Snapping is turned off, so this is only an estimation")
        dx, dy = self.config['general'].getfloat('cell_size_x'), self.config['general'].getfloat('cell_size_y')
        nx = int(self.config['general'].getint('scene_size_x') / dx)
        ny = int(self.config['general'].getint('scene_size_y') / dy)
        obstacle_gutter = np.zeros(self.obstacle_coverage.shape)

        for row, col in np.ndindex((nx, ny)):
            if not self.obstacle_coverage[row, col]:
                right, up = min(row + radius + 1, nx - 1), min(col + radius + 1, ny - 1)
                left, down = max(row - radius, 0), max(col - radius, 0)
                if np.any(self.obstacle_coverage[left:right, down:up]):
                    obstacle_gutter[row, col] = 1
        return obstacle_gutter

    def load_config(self):
        """
        Interpret the ConfigParser object to read the parameters from
        :return: None
        """
        section = self.config['general']
        obstacle_file = section['obstacle_file']
        self.total_pedestrians = section.getint('number_of_pedestrians')
        self.dt = float(section['dt'])
        self.minimal_distance = section.getfloat('minimal_distance')
        self.mde = section.getboolean('minimal_distance_enforcement')
        self.pedestrian_size = Size([section.getfloat('pedestrian_size'), section.getfloat('pedestrian_size')])
        self.size = Size([section.getfloat('scene_size_x'), section.getfloat('scene_size_y')])
        self.max_speed_interval = Interval([section.getfloat('max_speed_begin'), section.getfloat('max_speed_end')])
        self.snap_obstacles = section.getboolean('snap_obstacles')
        self.aware_percentage = self.config['aware'].getfloat('percentage', fallback=1.0)
        if self.config.has_section('fire'):
            fire_config = self.config['fire']
            self.fire_center = (fire_config.getfloat('center_x'), fire_config.getfloat('center_y'))
            self.fire_intensity = fire_config.getfloat('intensity')
            self.fire_radius = fire_config.getfloat('radius')
        self.load_obstacle_file(obstacle_file)

    def load_obstacle_file(self, file_name: str):
        """
        Reads in a JSON file and stores the obstacle data in the scene.
        The file must consist of one JSON object with keys 'obstacles', 'exits', and 'entrances'
        Every key must have a list of instances, each having a 'name', a 'begin' and a 'size'.
        Note that sizes are fractions of the scene size. A size of 0 is converted to 1 size unit.
        :param file_name: file name string of the JSON file
        :return: None
        """
        with open(file_name, 'r') as json_file:
            data = json.loads(json_file.read())
        margin = self.config['general'].getfloat('margin')
        cell_size = None
        if self.snap_obstacles:
            cell_size = np.array([self.config['general'].getfloat('cell_size_x'),
                                  self.config['general'].getfloat('cell_size_y')])
        for obstacle_data in data["obstacles"]:
            begin = Point(self.size * obstacle_data['begin'])
            size = Size(self.size * obstacle_data['size'])
            name = obstacle_data["name"]
            self.obstacle_list.append(Obstacle(begin, size, name, margin, cell_size=cell_size))
        if len(data['exits']) == 0:
            raise AttributeError('No exits specified in %s' % file_name)
        for exit_data in data['exits']:
            begin = Point(self.size * exit_data['begin'])
            size = self.size.array * np.array(exit_data['size'])
            name = exit_data["name"]
            cap = 0
            if 'cap' in exit_data:
                cap = exit_data['cap'] * self.dt
                if cap < 1:
                    ft.warn("Exit cap of %.2f per time step is too low low; must exceed 1.0")
            for dim in range(2):
                if size[dim] == 0.:
                    size[dim] = 1.
            exit_obs = Exit(cap=cap, begin=begin, size=Size(size), name=name, margin=margin, cell_size=cell_size)
            self.exit_list.append(exit_obs)
            self.obstacle_list.append(exit_obs)
        for entrance_data in data['entrances']:
            begin = Point(self.size * entrance_data['begin'])
            size = self.size.array * np.array(entrance_data['size'])
            name = entrance_data["name"]
            spawn_rate = 2 * self.dt
            max_pedestrians = 8000
            start_time = 0
            if 'spawn_rate' in entrance_data:
                spawn_rate = entrance_data['spawn_rate'] * self.dt
            if max_pedestrians in entrance_data:
                max_pedestrians = entrance_data['max_pedestrians']
            if 'start_time' in entrance_data:
                start_time = entrance_data['start_time']
            for dim in range(2):
                if size[dim] == 0.:
                    size[dim] = 1.
            entrance_obs = Entrance(begin=begin, size=Size(size), name=name, margin=margin, spawn_rate=spawn_rate,
                                    max_pedestrians=max_pedestrians, start_time=start_time, cell_size=cell_size)
            self.entrance_list.append(entrance_obs)
            self.obstacle_list.append(entrance_obs)
        if len(data['entrances']):
            self.on_step_functions.append(self.create_new_pedestrians)

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
        for function in self.on_pedestrian_exit_functions:
            function(pedestrian)
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

    def is_accessible(self, coord: Point, at_start=False) -> bool:
        """
        Checking whether the coordinate present is an accessible coordinate on the scene.
        When evaluated at the start, the exit is not an accessible object. That would be weird.
        :param coord: Coordinates to be checked
        :param at_start: Whether to be evaluated at the start
        :return: True if accessible, False otherwise.
        """
        if not self.is_within_boundaries(coord):
            return False
        if at_start:
            return all([coord not in obstacle for obstacle in self.obstacle_list])
        else:
            return all([coord not in obstacle or obstacle.accessible for obstacle in self.obstacle_list])

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
            # self.position_array += mde

    def correct_for_geometry(self):
        """
        Performs a vectorized correction to make sure pedestrians do not run into walls
        :return: None
        """
        geq_zero = np.logical_and(self.position_array[:, 0] > 0, self.position_array[:, 1] > 0)
        leq_size = np.logical_and(self.position_array[:, 0] < self.size[0], self.position_array[:, 1] < self.size[1])
        still_correct = np.logical_and(geq_zero, leq_size)
        for obstacle in self.obstacle_list:
            if not obstacle.accessible:
                in_obstacle = np.logical_and(self.position_array > obstacle.begin, self.position_array < obstacle.end)
                in_obs = np.logical_and(in_obstacle[:, 0], in_obstacle[:, 1])  # Faster than np.all(..,axis=1)
                still_correct = np.logical_and(still_correct, np.logical_not(in_obs))
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
        for goal in self.exit_list:
            in_goal = np.logical_and(self.position_array >= goal.begin, self.position_array <= goal.end)
            in_g = np.logical_and(in_goal[:, 0], in_goal[:, 1])
            done = np.logical_and(in_g, self.active_entries)
            index_list = np.where(done)[0]
            if goal.cap:
                surplus = max(0, len(index_list) - goal.cap)
                ft.debug("Surplus %d" % surplus)
                if surplus:

                    index_list = np.random.choice(index_list, goal.cap, replace=False)
                    goal.accessible = False
                else:
                    goal.accessible = True
            for index in index_list:
                finished_pedestrian = self.index_map[index]
                goal.log_pedestrian(finished_pedestrian, self.time)
                self.remove_pedestrian(finished_pedestrian)

    def is_done(self):
        """
        Checks whether all pedestrians are done by summing the active entries array and
        checking if all entrances are depleted.
        :return: True if all pedestrians are done, False otherwise
        """
        return not np.any(self.active_entries) and all([entrance.depleted for entrance in self.entrance_list])
