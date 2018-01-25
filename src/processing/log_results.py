# import src.self.params as self.params
import time
import base64
import os
import re

try:
    import h5py

    has_h5py = True
except ImportError:
    has_h5py = False

import numpy as np


class PositionLogger:

    def __init__(self, simulation):
        if not has_h5py:
            raise ImportError("Cannot launch logger, install h5py")
        self.hash = ("%.5f" % (time.time() % 1))[2:]
        self.simulation = simulation
        self.params = self.simulation.params
        self.filename = None
        self.file = None
        self.results_folder = "results"
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)
            print("Created new folder %s" % self.results_folder)

    def prepare(self):
        base_name = re.search('/([^/]+).(png|jpe?g)$', self.params.scene_file).group(1)
        self.filename = "%s/%s%s.h5" % (self.results_folder, base_name, self.hash)
        self.file = h5py.File(self.filename, 'w')
        self.file.create_group('scene')
        self.file.attrs['timestamp'] = time.time()
        with open(self.simulation.scene_file, 'rb') as image_file:
            binary_data = image_file.read()
        image_data = self.file['scene'].create_dataset("image", (2,))
        image_data[:] = self.simulation.scene.size.array
        image_data.attrs['environment'] = np.string_(base64.b64encode(binary_data))
        if 'fire' in self.simulation.effects:
            self.file['scene'].create_dataset('fire', data=self.simulation.effects['fire'].center)
        if 'cameras' in self.simulation.effects:
            combined_data = np.hstack(
                (self.simulation.effects['cameras'].positions, self.simulation.effects['cameras'].angles[:, None]))
            self.file['scene'].create_dataset('cameras', data=combined_data)

    def step(self):
        time_step = self.file.create_group('%d' % self.simulation.scene.counter)
        time_step.attrs['dt'] = self.params.dt
        micro = ['positions', 'velocities', 'active', 'map']  # Particle characteristics
        time_step.create_dataset('positions', data=self.simulation.scene.position_array)
        time_step.create_dataset('velocities', data=self.simulation.scene.velocity_array)
        time_step.create_dataset('active', data=self.simulation.scene.active_entries)
        time_step.create_dataset('map', data=self._get_index_array())
        for tag in micro:
            time_step[tag].attrs['level'] = 'micro'

        macro = ['density', 'velo_field_x', 'velo_field_y', 'pressure']  # Continuum characteristics
        time_step.create_dataset('density', data=self.simulation.effects['repulsion'].density_field.array)
        time_step.create_dataset('velo_field_x', data=self.simulation.effects['repulsion'].v_x.array)
        time_step.create_dataset('velo_field_y', data=self.simulation.effects['repulsion'].v_y.array)
        time_step.create_dataset('pressure', data=self.simulation.effects['repulsion'].pressure_field.array)
        if 'fire' in self.simulation.effects:
            time_step.create_dataset('smoke', data=self.simulation.effects['fire'].smoke_module.smoke_field.array)
            time_step['smoke'].attrs['level'] = 'macro'
        for tag in macro:
            time_step[tag].attrs['level'] = 'macro'

    def _get_index_array(self):
        """
        Convert the index map of the pedestrians to an numpy array.
        Each index (each row) contains the unique pedestrian counter.
        Empty arrays correspond to -1, so that index_array > 0 == active_array

        :return: a numpy array, shape of active_array with pedestrian counters
        """
        index_array = np.zeros_like(self.simulation.scene.active_entries) - 1
        for key, val in self.simulation.scene.index_map.items():
            if val:
                index_array[key] = val.counter
        return index_array
