__author__ = 'omar'

import pickle
import itertools
import json
import time
import numpy as np
import scipy.io as sio
import os
import functions as ft
from pedestrian import Pedestrian
from geometry import Point, Size, Interval
from obstacles import Obstacle, Entrance, Exit
from cells import Cell
from cython_modules.mde import minimum_distance_enforcement


class Scene:
    """
    Models a scene. A scene is a rectangular object with obstacles and pedestrians inside.
    """

    def __init__(self, initial_pedestrian_number, obstacle_file,
                 mde=True, cache='read', log_exits=False, use_exit_logs=False):
        """
        Initializes a Scene
        :param size: Size object holding the size values of the scene
        :param initial_pedestrian_number: Number of pedestrians on initialization in the scene
        :param obstacle_file: name of the file containing the obstacles.
        :param dt: update time step
        :return: scene instance.
        """
        self.time = 0
        self.total_pedestrians = initial_pedestrian_number

        self.obstacle_list = []
        self.exit_list = []
        self.entrance_list = []
        self.on_step_functions = []
        self.on_pedestrian_exit_functions = []
        self.on_finish_functions = []

        self.mde = mde  # Minimum Distance Enforcement
        self.use_exit_logs = use_exit_logs

        # Array initialization

        self.position_array = np.zeros([initial_pedestrian_number, 2])
        self.last_position_array = np.zeros([initial_pedestrian_number, 2])
        self.velocity_array = np.zeros([initial_pedestrian_number, 2])
        self.max_speed_array = np.empty(initial_pedestrian_number)
        self.active_entries = np.ones(initial_pedestrian_number)


        # Parameter initialization (will be overwritten by _load_parameters)
        self.minimal_distance = self.dt = 0
        self.number_of_cells = self.pedestrian_size = self.max_speed_interval = None

        self._load_parameters()
        self._read_json_file(file_name=obstacle_file)

        if log_exits:
            self.set_on_finish_functions(self.store_exit_logs)
        self.pedestrian_list = []
        self._init_pedestrians(initial_pedestrian_number)
        self.index_map = {i: self.pedestrian_list[i] for i in range(initial_pedestrian_number)}
        self.status = 'RUNNING'
        # Todo: Add handle for reuse_exits
        # Todo: Almost time to move settings to a settingsmanager
        # Todo: How about creating a scene manager?

    def _init_pedestrians(self, init_number):
        """
        Protected method that determines how the pedestrians are initially distributed,
        as well as with what properties they come. Overridable.
        :return: None
        """
        self.pedestrian_list = [
            Pedestrian(self, index, goals=self.exit_list, max_speed=self.max_speed_array[index])
            for index in range(init_number)]

    def create_new_pedestrians(self):
        for entrance in self.entrance_list:
            new_number = entrance.get_new_number_of_pedestrians(self.time)
            max_tries = 10 * new_number
            tries = 0  # when this becomes large, chances are the entrance can produce no valid new position.
            while new_number > 0:
                tries += 1
                new_position = entrance.get_spawn_location()
                if not self.is_accessible(new_position):
                    if tries > max_tries:
                        raise RuntimeError("Can't find new spawn locations for %s. Check the scene" % entrance)
                    continue
                free_indices = np.where(self.active_entries == 0)[0]
                if len(free_indices):
                    new_index = free_indices[0]
                else:
                    new_index = self.active_entries.shape[0]
                    self._expand_arrays()
                    ft.debug("Indices full. doubling array size to %d" % (2 * new_index))
                new_max_speed = self.max_speed_interval.random()
                self.max_speed_array[new_index] = new_max_speed  # todo: remove max speed
                new_pedestrian = Pedestrian(self, self.total_pedestrians, self.exit_list,
                                            new_max_speed, new_position, new_index)
                self.total_pedestrians += 1
                self.active_entries[new_index] = 1
                self.pedestrian_list.append(new_pedestrian)
                self.index_map[new_index] = new_pedestrian
                new_number -=1

    def _load_parameters(self, filename='params.json'):
        """
        Load parameters from JSON file. If file not present or damaged, load default parameters.
        :param filename: filename of file containing valid json
        :return: None
        """
        import os
        default_dict = {"dt": 0.05,
                        "width": 100,
                        "height": 100,
                        "minimal_distance": 0.7,
                        "pedestrian_size": [0.4, 0.4],
                        "max_speed_interval": [1, 2],
                        "smoothing_length": 0.75,  # times cell size
                        "packing_factor": 0.9}
        data_dict = {}
        if not os.path.isfile(filename):
            raise FileNotFoundError("Parameter file %s not found" % filename)
        with open(filename, 'r') as file:
            try:
                data = json.loads(file.read())
                for key in default_dict:
                    if key not in data.keys():
                        ft.warn("Not all keys found in %s. Using default parameters" % filename)
                        data_dict.update(default_dict)
                        break
                else:
                    data_dict.update(data)
            except ValueError:
                ft.warn("Invalid JSON in %s. Using default parameters" % filename)

        self.dt = data_dict['dt']
        self.size = Size((data_dict['width'], data_dict['height']))
        self.minimal_distance = data_dict['minimal_distance']
        self.pedestrian_size = Size(data_dict['pedestrian_size'])
        self.max_speed_interval = Interval(data_dict['max_speed_interval'])
        self.max_speed_array = self.max_speed_interval.begin + \
                               np.random.random(self.position_array.shape[0]) * self.max_speed_interval.length
        self.smoothing_length = data_dict['smoothing_length']
        self.packing_factor = data_dict['packing_factor']

    def _expand_arrays(self):
        """
        Increases the size (first dimension) of a numpy array with the given factor.
        Missing entries are set to zero
        :param factor: Multiplication factor for the size
        :return: None
        """
        # I don't like [gs]etattr, but this is pretty explicit
        attr_list = ["position_array", "last_position_array", "velocity_array",
                     "max_speed_array", "active_entries"]
        for attr in attr_list:
            array = getattr(self, attr)
            addition = np.zeros(array.shape)
            setattr(self, attr, np.concatenate((array, addition), axis=0))
        self.index_map.update({len(self.index_map) + i: None for i in range(len(self.index_map))})

    def set_on_step_functions(self, *on_step):
        """
        Adds functions to list called on each time step.
        :param on_step: functions (without arguments)
        :return: None
        """
        self.on_step_functions += on_step

    def set_on_pedestrian_exit_functions(self, *on_pedestrian_exit):
        """
        Adds functions to list called each time a pedestrian exits.
        :param on_pedestrian_exit: functions which take Pedestrian as argument
        :return: None
        """
        self.on_pedestrian_exit_functions += on_pedestrian_exit

    def set_on_finish_functions(self, *on_finish):
        """
        Adds functions to list called on simulation finish.
        :param on_finish: functions (without arguments)
        :return: None
        """
        self.on_finish_functions += on_finish

    def _read_json_file(self, file_name: str):
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
        for obstacle_data in data["obstacles"]:
            begin = Point(self.size * obstacle_data['begin'])
            size = Size(self.size * obstacle_data['size'])
            name = obstacle_data["name"]
            self.obstacle_list.append(Obstacle(begin, size, name))
        if len(data['exits']) == 0:
            raise AttributeError('No exits specified in %s' % file_name)
        for exit_data in data['exits']:
            begin = Point(self.size * exit_data['begin'])
            size = self.size.array * np.array(exit_data['size'])
            name = exit_data["name"]

            for dim in range(2):
                if size[dim] == 0.:
                    size[dim] = 1.
            exit_obs = Exit(begin, Size(size), name)
            self.exit_list.append(exit_obs)
            self.obstacle_list.append(exit_obs)
        for entrance_data in data['entrances']:  # Todo: Merge
            begin = Point(self.size * entrance_data['begin'])
            size = self.size.array * np.array(entrance_data['size'])
            name = entrance_data["name"]

            for dim in range(2):
                if size[dim] == 0.:
                    size[dim] = 1.
            entrance_obs = Entrance(begin, Size(size), name, exit_data=self.load_exit_logs())
            self.entrance_list.append(entrance_obs)
            self.obstacle_list.append(entrance_obs)
        if len(data['entrances']):
            self.set_on_step_functions(self.create_new_pedestrians)

    def get_stationary_pedestrians(self):
        """
        Computes which pedestrians have not moved since the last time step
        :return: nx1 boolean np.array, True if (existing) pedestrian is stationary, False otherwise
        """
        pos_difference = np.linalg.norm(self.position_array - self.last_position_array, axis=1)
        not_moved = np.logical_and(pos_difference == 0, self.active_entries == 1)
        return not_moved

    def remove_pedestrian(self, pedestrian):
        """
        Removes a pedestrian from the scene.
        :param pedestrian: The pedestrian instance to be removed.
        :return: None
        """
        # assert pedestrian.is_done()
        index = pedestrian.index
        self.index_map[index] = None
        self.pedestrian_list.remove(pedestrian)

        self.active_entries[index] = 0
        for function in self.on_pedestrian_exit_functions:
            function(pedestrian)
        if self.is_done():
            self.status = 'DONE'

    def is_within_boundaries(self, coord: Point):
        """
        Check whether a point lies within the scene.
        If used often, this should be either vectorized or short-circuited.
        :param coord: Point under consideration
        :return: True if within scene, false otherwise
        """
        within_boundaries = all(np.array([0, 0]) < coord.array) and all(coord.array < self.size.array)
        return within_boundaries

    def is_accessible(self, coord: Point, at_start=False) -> bool:
        """
        Checking whether the coordinate present is an accessible coordinate on the scene.
        When evaluated at the start, the exit is not an accessible object. That would be weird.
        We can eliminate this later though.
        :param coord: Coordinates to be checked
        :param at_start: Whether to be evaluated at the start
        :return: True if accessible, False otherwise.
        """
        if not self.is_within_boundaries(coord):
            return False
        if at_start:
            return all([coord not in obstacle for obstacle in self.obstacle_list])
        else:
            return all([coord not in obstacle or obstacle.permeable for obstacle in self.obstacle_list])

    def move_pedestrians(self):
        """
        Performs a vectorized move of all the pedestrians.
        Assumes that all the velocities have been set accordingly.
        :return: None
        """
        self.time += self.dt
        self.last_position_array = np.array(self.position_array)
        self.position_array += self.velocity_array * self.dt
        time1 = time.time()
        if self.mde:
            self.position_array += minimum_distance_enforcement(self.size.array, self.position_array,
                                                                self.active_entries,
                                                                self.minimal_distance)
        print("%.2e" % (time.time() - time1))
        [function() for function in self.on_step_functions]

    def correct_for_geometry(self):
        """
        Performs a vectorized correction to make sure pedestrians do not run into walls
        :return: None
        """
        geq_zero = np.logical_and(self.position_array[:, 0] > 0, self.position_array[:, 1] > 0)
        leq_size = np.logical_and(self.position_array[:, 0] < self.size[0], self.position_array[:, 1] < self.size[1])
        still_correct = np.logical_and(geq_zero, leq_size)
        for obstacle in self.obstacle_list:
            if not obstacle.permeable:
                in_obstacle = np.logical_and(self.position_array > obstacle.begin, self.position_array < obstacle.end)
                in_obs = np.logical_and(in_obstacle[:, 0], in_obstacle[:, 1])  # Faster than np.all(..,axis=1)
                still_correct = np.logical_and(still_correct, np.logical_not(in_obs))
        still_correct = np.logical_and(still_correct, self.active_entries)
        self.position_array += np.logical_not(still_correct)[:, None] * (self.last_position_array - self.position_array)

    def find_finished_pedestrians(self):
        for goal in self.exit_list:
            in_goal = np.logical_and(self.position_array >= goal.begin, self.position_array <= goal.end)
            in_g = np.logical_and(in_goal[:, 0], in_goal[:, 1])
            done = np.logical_and(in_g, self.active_entries)
            for index in np.where(done)[0]:
                finished_pedestrian = self.index_map[index]
                goal.log_pedestrian(finished_pedestrian, self.time)
                self.remove_pedestrian(finished_pedestrian)


    def store_exit_logs(self, file_name=None):
        log_dir = 'results/'
        if not file_name:
            file_name = 'logs'
        log_dict = {exit_object.name: np.array(exit_object.log_list) for exit_object in self.exit_list}
        sio.savemat(file_name=log_dir + file_name, mdict=log_dict)

    def load_exit_logs(self, file_name=None):
        """
        If flagged: Open logs, convert them to a list.
        :param file_name:
        :return:
        """
        # Todo: No pretty call.
        if self.use_exit_logs:
            log_dir = 'results/'
            if not file_name:
                file_name = 'logs'
            log_dict = sio.loadmat(file_name=log_dir + file_name)
            log_lists = [log_list for log_list in log_dict.values() if isinstance(log_list, np.ndarray)]
            return log_lists
        else:
            return []

    def is_done(self):
        return np.sum(self.active_entries) == 0 and all([entrance.depleted for entrance in self.entrance_list])

    def finish(self):
        """
        Call all methods that need to be called upon finish.
        :return: None
        """
        for finish_function in self.on_finish_functions:
            finish_function()
        ft.log('Simulation is finished. Exiting')
