__author__ = 'omar'

import numpy as np
import scipy.io as sio

import functions as ft


class Result:
    def __init__(self, scene, result_directory='results/'):
        """
        Results processed in this class:
        - Path length (measuring the distance between all successive points pedestrians visit).
        - Planned path length (measuring the sum of line segment lengths of the planned path).
        - Time spent (number of time steps * dt pedestrians spend inside the scene).
        - Density (total mass (=number of pedestrians) divided by the area of the scene (minus obstacles)).

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
        self.scene.set_on_step_functions(self.on_step)
        self.scene.set_on_pedestrian_exit_functions(self.on_pedestrian_exit)
        self.scene.set_on_finish_functions(self.on_finish)

        ft.debug("Simulations results are processed and stored in folder '%s'" % result_directory)
        if not scene.pedestrian_list[0].line:
            raise AttributeError("Pedestrians have no paths. Assert Graphplanner module is used")
        self.planned_path_length = np.zeros(self.scene.pedestrian_number)
        self.path_length = np.zeros(self.scene.pedestrian_number)
        self.time_spent = np.zeros(self.scene.pedestrian_number)
        self.mean_speed = np.zeros(self.scene.pedestrian_number)
        self.path_length_ratio = np.zeros(self.scene.pedestrian_number)
        self.avg_path_length_ratio = 0
        self.density = 0

        self.on_init()

    def on_init(self):
        for pedestrian in self.scene.pedestrian_list:
            self.planned_path_length[pedestrian.counter] = self._get_planned_path_length(pedestrian)
        self.density = self.scene.pedestrian_number / np.prod(self.scene.size.array)

    def _get_planned_path_length(self, pedestrian):
        """
        Compute the sum of Euclidian lengths of the path line segments
        :param pedestrian: owner of the path
        :return: Length of planned path
        """
        length = pedestrian.line.length
        for line in pedestrian.path:
            length += line.length
        return length

    def on_step(self):
        """
        All data that should be gathered on each time step:
        Currently used:
        - Distance walked per pedestrian
        :return: None
        """
        distance = np.linalg.norm(self.scene.position_array - self.scene.last_position_array, axis=1)
        self.path_length += distance

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
        self.path_length_ratio = self.path_length / self.planned_path_length
        self.avg_path_length_ratio = np.mean(self.path_length_ratio)
        self.mean_speed = self.path_length / self.time_spent
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
        filename = "results"
        sio.savemat(filename, mdict={"planned_path_length": self.planned_path_length,
                                     "path_ratio": self.path_length_ratio,
                                     "avg_path_length_ratio": self.avg_path_length_ratio,
                                     "mean_speed": self.mean_speed})
