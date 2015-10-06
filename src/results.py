__author__ = 'omar'

import numpy as np
import scipy.io as sio

import functions as ft
from static_planner import GraphPlanner


class Result:
    def __init__(self, scene, result_directory='results/', subsequent=True):
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

        This Result class depends on a GraphPlanner to capture the length of the paths.

        :param scene: Scene object under evaluation.
        :return: None
        """
        self.result_dir = result_directory
        self.scene = scene
        self.subsequent = subsequent
        self.scene.set_on_step_functions(self.on_step)
        self.scene.set_on_pedestrian_exit_functions(self.on_pedestrian_exit)
        self.scene.set_on_finish_functions(self.on_finish)

        ft.log("Simulations results are processed and stored in folder '%s'" % result_directory)
        if not scene.pedestrian_list[0].line:
            raise AttributeError("Pedestrians have no paths. Assert GraphPlanner module is used")
        self.planned_path_length = np.zeros(self.scene.pedestrian_number)
        self.path_length = np.zeros(self.scene.pedestrian_number)
        self.time_spent = np.zeros(self.scene.pedestrian_number)
        self.mean_speed = np.zeros(self.scene.pedestrian_number)
        self.path_length_ratio = np.zeros(self.scene.pedestrian_number)
        self.max_speed = np.zeros(self.scene.pedestrian_number)
        self.avg_path_length_ratio = 0
        self.avg_mean_speed = 0
        self.density = self.scene.pedestrian_number / (self.scene.size.width * self.scene.size.height)
        self.origins = np.zeros([self.scene.pedestrian_number, 2])
        self.paths_list = [[] for _ in range(self.scene.pedestrian_number)]

        self.on_init()

    def on_init(self):
        """
        All pre-processing and calls that should be executed before the simulation starts.
        :return: None
        """
        for pedestrian in self.scene.pedestrian_list:
            self.planned_path_length[pedestrian.counter] = GraphPlanner.get_path_length(pedestrian)
            self.origins[pedestrian.counter] = pedestrian.origin.array
        self.density = self.scene.pedestrian_number / np.prod(self.scene.size.array)
        self.max_speed = self.scene.max_speed_array

    def on_step(self):
        """
        All data that should be gathered on each time step:
        Currently used:
        - Distance walked per pedestrian
        :return: None
        """
        distance = np.linalg.norm(self.scene.position_array - self.scene.last_position_array, axis=1)
        self.path_length[np.where(self.scene.alive_array)] += distance[np.where(self.scene.alive_array)]
        for pedestrian in self.scene.pedestrian_list:
            if self.scene.alive_array[pedestrian.counter]:
                # Comment out if not needed.
                self.paths_list[pedestrian.counter].append(pedestrian.position.array)

    def on_pedestrian_exit(self, pedestrian):
        """
        All data that should be gathered on each pedestrians exit:
        :return: None
        """
        self.time_spent[pedestrian.counter] = self.scene.time

    def on_finish(self):
        """
        All data that should be gathered on finishing the simulation:
        :return: None
        """
        for pedestrian in self.scene.pedestrian_list:
            if self.scene.alive_array[pedestrian.counter] == 1:
                self.time_spent[pedestrian.counter] = self.scene.time
        self.path_length_ratio = self.planned_path_length[np.invert(self.scene.alive_array.astype(bool))] / \
                                 self.path_length[np.invert(self.scene.alive_array.astype(bool))]
        self.avg_path_length_ratio = np.mean(self.path_length_ratio)
        self.mean_speed = self.path_length / self.time_spent
        self.paths_list = np.array(self.paths_list)
        self.avg_mean_speed = np.mean(self.mean_speed)
        self.write_matlab_results()

    def write_results(self):
        """
        Store results to file or to stdout
        :return: None
        """
        filename = "hoi.txt"
        with open(filename, 'w') as file:
            file.write("Planned path length:\n%s\n\n" % self.planned_path_length)
            file.write("Path ratio:\n%s\n\n" % self.path_length_ratio)
            file.write("Average path length ratio\n%s\n\n" % self.avg_path_length_ratio)
            file.write("Mean speed\n%s\n\n" % self.mean_speed)
            file.write("Density:\n%s\n\n" % self.density)


    def write_matlab_results(self):
        """
        Stores result in binary matlab file
        :return: None
        """
        filename = self.result_dir + 'results'
        sio.savemat(filename, mdict={"planned_path_length": self.planned_path_length,
                                     "path_length": self.path_length,
                                     "path_length_ratio": self.path_length_ratio,
                                     "avg_path_length_ratio": self.avg_path_length_ratio,
                                     "max_speed": self.max_speed,
                                     "time_spent": self.time_spent,
                                     "origins": self.origins,
                                     "avg_mean_speed": self.avg_mean_speed,
                                     "paths_list": self.paths_list,
                                     "mean_speed": self.mean_speed})

    def write_density_results(self, filename):
        import os
        if not os.path.exists(filename):
            raise IOError("file %s not found" % filename)
        with open(filename, 'a') as o:
            o.write("%.4f\t%.4f\t%.4f\n" % (self.density, self.avg_path_length_ratio, self.avg_mean_speed))
