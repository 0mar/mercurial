__author__ = 'omar'

import pickle
import itertools
import json

import numpy as np
import scipy.io as sio

import functions as ft
from pedestrian import Pedestrian
from geometry import Point, Size, Interval
from obstacles import Obstacle, Entrance, Exit
from cells import Cell


class Scene:
    """
    Models a scene. A scene is a rectangular object with obstacles and pedestrians inside.
    """

    def __init__(self, size: Size, initial_pedestrian_number, obstacle_file,
                 mde=True, cache='read', log_exits=False, use_exit_logs=False):
        """
        Initializes a Scene
        :param size: Size object holding the size values of the scene
        :param initial_pedestrian_number: Number of pedestrians on initialization in the scene
        :param obstacle_file: name of the file containing the obstacles.
        :param dt: update time step
        :return: scene instance.
        """
        self.size = size
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
        self._read_json_file(file_name=obstacle_file)
        self.position_array = np.zeros([initial_pedestrian_number, 2])
        self.last_position_array = np.zeros([initial_pedestrian_number, 2])
        self.velocity_array = np.zeros([initial_pedestrian_number, 2])
        self.max_speed_array = np.empty(initial_pedestrian_number)
        self.pedestrian_cells = np.zeros([initial_pedestrian_number, 2])
        self.active_entries = np.ones(initial_pedestrian_number)
        self._expand_arrays()
        self.cell_dict = {}

        # Parameter initialization (will be overwritten by _load_parameters)
        self.minimal_distance = self.dt = 0
        self.number_of_cells = self.pedestrian_size = self.max_speed_interval = None
        self._load_parameters()

        self.cell_size = Size(self.size.array / self.number_of_cells)
        if log_exits:
            self.set_on_finish_functions(self.store_exit_logs)
        if cache == 'read':
            self._load_cells()
        else:
            self._create_cells()
        if cache == 'write':
            self._store_cells()
        self.pedestrian_list = []
        self._init_pedestrians(initial_pedestrian_number)
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
            Pedestrian(self, index, goals=self.exit_list, size=self.size, max_speed=self.max_speed_array[index])
            for index in range(init_number)]
        self._fill_cells()

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
                new_pedestrian = Pedestrian(self, self.total_pedestrians, self.exit_list, self.pedestrian_size,
                                            new_max_speed, new_position, new_index)
                self.total_pedestrians += 1
                self.active_entries[new_index] = 1
                self.pedestrian_list.append(new_pedestrian)
                cell_location = np.floor(new_position / self.cell_size)
                cell_index = int(cell_location[0]), int(cell_location[1])
                self.cell_dict[cell_index].add_pedestrian(new_pedestrian)
                new_number -=1

    def _load_parameters(self, filename='params.json'):
        """
        Load parameters from JSON file. If file not present or damaged, load default parameters.
        :param filename: filename of file containing valid json
        :return: None
        """
        import os
        default_dict = {"dt": 0.05,
                        "number_of_cells": [20, 20],
                        "minimal_distance": 0.7,
                        "pedestrian_size": [0.4, 0.4],
                        "max_speed_interval": [1, 2],
                        "interpolation_factor": 3,
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
        self.number_of_cells = tuple(data_dict['number_of_cells'])
        self.minimal_distance = data_dict['minimal_distance']
        self.pedestrian_size = Size(data_dict['pedestrian_size'])
        self.max_speed_interval = Interval(data_dict['max_speed_interval'])
        self.max_speed_array = self.max_speed_interval.begin + \
                               np.random.random(self.position_array.shape[0]) * self.max_speed_interval.length
        self.interpolation_factor = data_dict['interpolation_factor']
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
                     "max_speed_array", "pedestrian_cells", "active_entries"]
        for attr in attr_list:
            array = getattr(self, attr)
            addition = np.zeros(array.shape)
            setattr(self, attr, np.concatenate((array, addition), axis=0))

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

    def _create_cells(self):
        """
        Creates the cell objects into which the scene is partitioned.
        The cell objects are stored in the scenes cell_dict as {(row, column): Cell}
        This is a time intensive operation which can be avoided by using the cache function
        :return: None
        """
        ft.log("Started preprocessing cells")
        self.cell_dict = {}
        for row, col in np.ndindex(self.number_of_cells):
            start = Point(self.cell_size.array * [row, col])
            cell = Cell(row, col, start, self.cell_size)
            self.cell_dict[(row, col)] = cell
            cell.obtain_relevant_obstacles(self.obstacle_list)
        ft.log("Finished preprocessing cells")

    def _fill_cells(self):
        """
        This method fills the cells in self.cell_dict with the pedestrians
        :return: None
        """
        cell_locations = np.floor(self.position_array / self.cell_size)
        self.pedestrian_cells = cell_locations
        for pedestrian in self.pedestrian_list:
            cell_location = (int(cell_locations[pedestrian.index, 0]), int(cell_locations[pedestrian.index, 1]))
            self.cell_dict[cell_location].add_pedestrian(pedestrian)

    def _store_cells(self, filename='cells.bin'):
        """
        This method pickles the cells dictionary (without pedestrians) into a file.
        It can be loaded using _load_cells
        :param filename: name of pickled scene file
        :return: None
        """
        with open(filename, 'wb') as pickle_file:
            pickle.dump(self.cell_dict, pickle_file)

    def _load_cells(self, filename='cells.bin'):
        """
        Opens and unpickles the file containing the scene dictionary
        If file does not exist, creates new cells.
        If cells do not correspond to the scene + obstacles, creates new cells.
        :param filename: name of pickled scene file
        :return: None
        """
        # Todo: Depends on scene
        ft.log("Loading cell objects from file")

        def reject_cells():
            ft.log("Cells cache does not correspond to this scene. Creating new cells and storing those.")
            self._create_cells()
            self._store_cells()

        import os.path
        if os.path.isfile(filename):
            with open(filename, 'rb') as pickle_file:
                self.cell_dict = pickle.load(pickle_file)
        else:
            reject_cells()
        if not self._validate_cells():
            reject_cells()

    def _validate_cells(self):
        """
        Compares the cell dictionary to the scene information.
        Polls a cells and checks its location and its size to see if it behaves as expected.
        :return: False if the method detects an inconsistency, True otherwise
        """
        cell_location = set(self.cell_dict).pop()
        correct_index = all([cell_location[dim] < self.number_of_cells[dim] for dim in range(2)])
        correct_size = np.all(np.isclose(self.cell_dict[cell_location].size.array, self.cell_size.array))
        correct_obstacle = {obs.name for obs in self.obstacle_list} \
                           == {obs.name for cell in self.cell_dict.values() for obs in cell.obstacle_set}
        return correct_index and correct_size and correct_obstacle

    def get_cell_from_position(self, position):
        """
        Obtain the cell corresponding to a certain position
        :param position: position
        :return: corresponding cell
        """
        size = Size(self.size.array / self.number_of_cells)
        cell_location = np.floor(np.array(position) / size)
        return self.cell_dict[(int(cell_location[0]), int(cell_location[1]))]

    def get_pedestrian_cells(self):
        """
        Obtain the pedestrian distribution over the cells.
        :return:array with integer values per pedestrian corresponding to its cell.
        """
        raw_cell_locations = np.floor(self.position_array / self.cell_size)
        return raw_cell_locations

    def get_stationary_pedestrians(self):
        """
        Computes which pedestrians have not moved since the last time step
        :return: nx1 boolean np.array, True if (existing) pedestrian is stationary, False otherwise
        """
        pos_difference = np.linalg.norm(self.position_array - self.last_position_array, axis=1)
        not_moved = np.logical_and(pos_difference == 0, self.active_entries == 1)
        return not_moved

    def update_cells(self):
        """
        Update all the pedestrians, but by inspecting the new situation per cell.
        :return: None
        """
        new_ped_cells = self.get_pedestrian_cells()
        needs_update = self.pedestrian_cells != new_ped_cells
        for pedestrian in self.pedestrian_list:
            index = pedestrian.index
            if any(needs_update[index]):
                cell = pedestrian.cell
                new_cell_orientation = (int(new_ped_cells[index, 0]), int(new_ped_cells[index, 1]))
                if new_cell_orientation in self.cell_dict:
                    cell.remove_pedestrian(pedestrian)
                    new_cell = self.cell_dict[new_cell_orientation]
                    new_cell.add_pedestrian(pedestrian)
                else:
                    pass
        self.pedestrian_cells = new_ped_cells

    def _minimum_distance_enforcement(self, min_distance):
        """
        Finds the pedestrian pairs that are closer than the specified distance.
        Does so by comparing the distances of all pedestrians a,b in a cell.
        Note that intercellullar pedestrian pairs are ignored,
        we might fix this later.

        :param min_distance: minimum distance between pedestrians, including their size.
        :return: list of pedestrian index pairs with distances lower than min_distance.
        """
        list_a = []
        list_b = []
        index_list = []
        for cell in self.cell_dict.values():
            for ped_combination in itertools.combinations(cell.pedestrian_set, 2):
                list_a.append(ped_combination[0].position.array)
                list_b.append(ped_combination[1].position.array)
                index_list.append([ped_combination[0].index, ped_combination[1].index])
        array_a = np.array(list_a)
        array_b = np.array(list_b)
        array_index = np.array(index_list)
        differences = array_a - array_b
        if len(differences) == 0:
            return
        distances = np.linalg.norm(differences, axis=1)
        indices = np.where(distances < min_distance)[0]

        mde_index_pairs = array_index[indices]
        mde_corrections = (min_distance / (distances[indices][:, None] + ft.EPS) - 1) * differences[indices] / 2
        ordered_corrections = np.zeros(self.position_array.shape)
        for it in range(len(mde_index_pairs)):
            pair = mde_index_pairs[it]
            ordered_corrections[pair[0]] += mde_corrections[it]
            ordered_corrections[pair[1]] -= mde_corrections[it]
        self.position_array += ordered_corrections

    def remove_pedestrian(self, pedestrian):
        """
        Removes a pedestrian from the scene.
        :param pedestrian: The pedestrian instance to be removed.
        :return: None
        """
        # assert pedestrian.is_done()
        assert self.active_entries[pedestrian.index]
        pedestrian.cell.remove_pedestrian(pedestrian)
        index = pedestrian.index
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
        if self.mde:
            self._minimum_distance_enforcement(self.minimal_distance)
        self.update_cells()
        [function() for function in self.on_step_functions]

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
