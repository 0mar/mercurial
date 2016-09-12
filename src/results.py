__author__ = 'omar'

import numpy as np
import scipy.io as sio

import functions as ft


class Result:
    def __init__(self, scene):
        """
        Results processed in this class:
        - Path length (measuring the distance between all successive points pedestrians visit).
        - Planned path length (measuring the sum of line segment lengths of the planned path).
        - Time spent (number of time steps * dt pedestrians spend inside the scene).
        - Density (total mass (=number of pedestrians) divided by the area of the scene (minus obstacles)).
        - Origin (starting point of each pedestrian, to see the results as a relative factor).
        - Path (as a sequence of followed points per pedestrian)

        Derived results:
        - Path length ratio: Path length over planned path length per pedestrian.
        - Average path length ratio: Average of all path length ratio's.
        - Mean speed: Path length over time spend in simulation.

        :param scene: Scene object under evaluation.
        :return: None
        """
        self.result_dir = scene.config['general']['result_dir']
        self.scene = scene

        ft.log("Simulations results are processed and stored in folder '%s'" % self.result_dir)
        if not hasattr(scene.pedestrian_list[0], "line"):
            self.no_paths = True
            ft.log("Storing results for Dynamic planner")
        else:
            self.no_paths = False
            ft.log("Storing results for Graph Planner")
        self.position_list = []
        self.time_spent = np.zeros(len(self.scene.pedestrian_list))
        self.mean_speed = np.zeros(len(self.scene.pedestrian_list))
        self.max_speed = np.zeros(len(self.scene.pedestrian_list))
        self.finished = np.zeros(len(self.scene.pedestrian_list), dtype=bool)
        self.started = np.zeros(len(self.scene.pedestrian_list), dtype=bool)

        self.avg_mean_speed = 0
        self.density = len(self.scene.pedestrian_list) / (self.scene.size[0] * self.scene.size[1])
        self.origins = np.zeros([len(self.scene.pedestrian_list), 2])

        self.on_init()

    def _expand_arrays(self):
        """
        Increases the size (first dimension) of a numpy array with the given factor.
        Missing entries are set to zero
        :return: None
        """
        attr_list = ["path_length", "max_speed", "mean_speed", "time_spent",
                     "started", "finished", "origins", "planned_path_length"]

        for attr in attr_list:
            array = getattr(self, attr)
            addition = np.zeros(array.shape, dtype=array.dtype)
            setattr(self, attr, np.concatenate((array, addition), axis=0))

    def on_init(self):
        """
        All pre-processing and calls that should be executed before the simulation starts.
        :return: None
        """
        for pedestrian in self.scene.pedestrian_list:
            self.on_pedestrian_entrance(pedestrian)
        self.density = len(self.scene.pedestrian_list) / np.prod(self.scene.size.array)
        self.max_speed = self.scene.max_speed_array

    def on_step(self):
        """
        All data that should be gathered on each time step:
        Currently used:
        - Distance walked per pedestrian
        :return: None
        """
        distance = np.linalg.norm(self.scene.position_array - self.scene.last_position_array, axis=1)
        index_list = [self.scene.index_map[index].counter for index in np.where(self.scene.active_entries)[0]]
        for pedestrian in self.scene.pedestrian_list:
            if self.scene.active_entries[pedestrian.index]:
                # Comment out if not needed. Lots of memory
                self.position_list.append(
                    [self.scene.position_array[pedestrian.index, 0], self.scene.position_array[pedestrian.index, 1],
                     self.scene.time])
                # Without copy(), we get a reference (even though we slice the array...)

    def on_pedestrian_entrance(self, pedestrian):
        """
        All data that should be initialized and gathered on pedestrian entrance
        :param pedestrian: Entering pedestrian (from entrance)
        :return: None
        """
        if self.origins.shape[0] < self.scene.total_pedestrians:  # Check if we need to expand
            self._expand_arrays()
            print("Expanding to %d" % self.origins.shape[0])
        assert self.origins.shape[0] >= self.scene.total_pedestrians
        self.origins[pedestrian.counter] = pedestrian.origin.array
        self.started[pedestrian.counter] = True

    def on_pedestrian_exit(self, pedestrian):
        """
        All data that should be gathered on each pedestrians exit:
        :param pedestrian: Exiting pedestrian
        :return: None
        """
        self.time_spent[pedestrian.counter] = self.scene.time
        self.finished[pedestrian.counter] = True

    def on_finish(self):
        """
        All data that should be gathered on finishing the simulation:
        :return: None
        """
        ft.log("Starting post-processing results")
        unfinished_counters = [self.scene.index_map[index].counter for index in
                               np.where(self.scene.active_entries)[0]]
        for ped_index in unfinished_counters:
            self.time_spent[ped_index] = self.scene.time
        self.paths_list = np.array(self.paths_list)
        self.position_list = np.array(self.position_list)
        self.avg_mean_speed = np.mean(self.mean_speed)
        self.write_matlab_results()
        ft.log("Finished post-processing results")

    def write_results(self):
        """
        Store results to file or to stdout
        :return: None
        """
        filename = "hoi.txt"
        with open(filename, 'w') as file:
            file.write("Mean speed\n%s\n\n" % self.mean_speed)
            file.write("Density:\n%s\n\n" % self.density)

    def write_matlab_results(self):
        """
        Stores result in binary matlab file
        :return: None
        """
        ft.log("Started writing results to binary file")
        filename = self.result_dir + 'results' + self.scene.config['general']['number_of_pedestrians']
        sio.savemat(filename, mdict={"max_speed": self.max_speed,
                                     "time_spent": self.time_spent,
                                     "origins": self.origins,
                                     "avg_mean_speed": self.avg_mean_speed,
                                     "paths_list": self.paths_list,
                                     "mean_speed": self.mean_speed,
                                     "position_list": self.position_list,
                                     "finished": self.finished,
                                     "final_positions": self.scene.position_array[self.scene.active_entries]})
        ft.log("Finished writing results to binary file")