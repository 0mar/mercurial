import matplotlib

matplotlib.use('TkAgg')
import numpy as np
from micro import MicroTransporter
from macro import MacroTransporter


class CombiPlanner:
    """
    This should be a combination planner.
    Same as the dynamic planner, only ignoring the density so the potential field
    is computed once and we use the pressure computer from Narain
    """

    def __init__(self, scene, show_plot=False):
        """
        Initializes a dynamic planner object. Takes a scene as argument.
        Parameters are initialized in this constructor, still need to be validated.
        :param scene: scene object to impose planner on
        :return: dynamic planner object
        """
        # Initialize depending on scene or on grid_computer?
        self.scene = scene
        self.config = scene.config
        self.micro = MicroTransporter(scene)
        self.macro = MacroTransporter(self.scene, False, True, True)
        self.macro.smoothing_length = self.scene.core_distance

    def step(self):
        """
        Computes the scalar fields (in the correct order) necessary for the dynamic planner.
        If plotting is enables, updates the plot.
        :return: None
        """

        self.micro.compute_velocities()
        self.macro.step()
        self.scene.move_pedestrians()
        self.scene.correct_for_geometry()
        self.nudge_stationary_pedestrians()
        self.scene.find_finished_pedestrians()

    def nudge_stationary_pedestrians(self):
        stat_ped_array = self.scene.get_stationary_pedestrians()
        num_stat = np.sum(stat_ped_array)
        if num_stat > 0:
            nudge = np.random.random((num_stat, 2)) - 0.5
            correction = self.scene.max_speed_array[stat_ped_array][:, None] * nudge * self.scene.dt
            self.scene.position_array[stat_ped_array] += correction
            self.scene.correct_for_geometry()
