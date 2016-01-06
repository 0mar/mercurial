import numpy as np

from image_processing import ImageProcessor

__author__ = 'omar'


class SkeletonPlanner:
    def __init__(self, scene, config):
        self.scene = scene
        self.config = config
        feature_transform = ImageProcessor.get_feature_transform(self.scene)

        self.ft_x = feature_transform[0, :, :]
        self.ft_y = feature_transform[1, :, :]
        self.ft_shape = self.ft_x.shape
        self.scale = self.ft_shape / self.scene.size.array

    def step(self):
        scaled_ped_locations = self.scene.position_array * self.scale
        scaled_ped_locations_ij = np.ceil(self._location_xy_to_ij(scaled_ped_locations) - 1).astype(int)
        scaled_f_x = self.ft_x[scaled_ped_locations_ij[:, 0], scaled_ped_locations_ij[:, 1]][:, None]
        scaled_f_y = self.ft_y[scaled_ped_locations_ij[:, 0], scaled_ped_locations_ij[:, 1]][:, None]
        feature_locations_ij = np.hstack((scaled_f_x, scaled_f_y))
        feature_locations = self._location_ij_to_xy(feature_locations_ij) / self.scale
        self.scene.velocity_array[:] = self.scene.position_array - feature_locations
        self.scene.velocity_array /= (np.linalg.norm(self.scene.velocity_array, axis=1)
                                      * self.scene.max_speed_array)[:, None]
        self.scene.move_pedestrians()
        self.scene.correct_for_geometry()
        self.scene.find_finished_pedestrians()

    def _location_xy_to_ij(self, indices):
        assert indices.shape[1] == 2
        return np.hstack((self.ft_shape[1] - indices[:, 1][:, None], indices[:, 0][:, None]))

    def _location_ij_to_xy(self, indices):
        assert indices.shape[1] == 2
        return np.hstack((indices[:, 1][:, None], self.ft_shape[1] - indices[:, 0][:, None]))
