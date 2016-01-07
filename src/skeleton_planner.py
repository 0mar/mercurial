import numpy as np

from image_processing import ImageProcessor
import functions as ft
__author__ = 'omar'


class SkeletonPlanner:
    def __init__(self, scene, config):
        self.scene = scene
        self.config = config
        feature_transform = ImageProcessor.get_medial_axis(self.scene)

        self.ft_x = feature_transform[0, :, :]
        self.ft_y = feature_transform[1, :, :]
        self.ft_shape = self.ft_x.shape
        self.scale = self.ft_shape / self.scene.size.array
        self.max_distance = self._get_max_distance()

    def step(self):
        feature_locations = self._get_feature_locations()
        distance_to_exit = self._get_distance_to_exit()
        feature_coefficient = distance_to_exit / self.max_distance
        feature_directions = ft.normalize(self.scene.position_array - feature_locations)
        exit_directions = ft.normalize(distance_to_exit)
        self.scene.velocity_array[:] = feature_coefficient * feature_directions + \
                                       (1 - feature_coefficient) * exit_directions
        self.scene.velocity_array[:] = ft.normalize(self.scene.velocity_array)
        self.scene.velocity_array *= self.scene.max_speed_array[:, None]
        self.scene.move_pedestrians()
        self.scene.correct_for_geometry()
        self.scene.find_finished_pedestrians()

    def _get_max_distance(self):
        """
        Get a sense of the maximum distance to the exit in this scene
        :return: approximation of the furthest any pedestrian is located from the exit
        """
        resolution = 15
        exit_center = self.scene.obstacle_list[0].center.array
        size = self.scene.size.array
        distances = np.fromfunction(lambda i, j: np.sqrt(((i + 0.5) * size[0] / resolution - exit_center[0]) ** 2 +
                                                         ((j + 0.5) * size[1] / resolution - exit_center[1]) ** 2),
                                    (resolution, resolution))
        correction_factor = 1 + 0.5 / resolution
        return np.max(distances) * correction_factor

    def _get_feature_locations(self):
        scaled_ped_locations = self.scene.position_array * self.scale
        scaled_ped_locations_ij = np.ceil(self._location_xy_to_ij(scaled_ped_locations) - 1).astype(int)
        scaled_f_x = self.ft_x[scaled_ped_locations_ij[:, 0], scaled_ped_locations_ij[:, 1]][:, None]
        scaled_f_y = self.ft_y[scaled_ped_locations_ij[:, 0], scaled_ped_locations_ij[:, 1]][:, None]
        feature_locations_ij = np.hstack((scaled_f_x, scaled_f_y))
        feature_locations = self._location_ij_to_xy(feature_locations_ij) / self.scale
        return feature_locations

    def _get_distance_to_exit(self):
        # Naive implementation
        obstacle_center = self.scene.exit_list[0].center
        distance_to_exit = obstacle_center.array - self.scene.position_array
        return distance_to_exit

    def _location_xy_to_ij(self, indices):
        assert indices.shape[1] == 2
        return np.hstack((self.ft_shape[1] - indices[:, 1][:, None], indices[:, 0][:, None]))

    def _location_ij_to_xy(self, indices):
        assert indices.shape[1] == 2
        return np.hstack((indices[:, 1][:, None], self.ft_shape[1] - indices[:, 0][:, None]))
