# import src.params as params
import time
import base64

try:
    import h5py

    has_h5py = True
except ImportError:
    print("Logging of the results requires storing HDF5 files. Please install h5py")
    has_h5py = False

import numpy as np


class Logger:

    def __init__(self, simulation):
        if not has_h5py:
            raise ImportError("Cannot launch logger, install h5py")
        self.hash = ("%.5f" % (time.time() % 1))[2:]
        self.simulation = simulation
        self.filename = None
        self.file = None

    def prepare(self):
        self.filename = 'trd.h5'  # self.simulation.scene_file+self.hash
        self.file = h5py.File(self.filename, 'w')
        self.file.create_group('scene')
        self.file.attrs['timestamp'] = time.time()
        with open(self.simulation.scene_file, 'rb') as image_file:
            binary_data = image_file.read()
        image_data = self.file['scene'].create_dataset("image", (2,))
        image_data[:] = self.simulation.scene.size.array
        image_data.attrs['environment'] = np.string_(base64.b64encode(binary_data))

    def step(self):
        time_step = self.file.create_group('%d' % self.simulation.scene.counter)
        time_step.attrs['dt'] = 0.1  # params.dt
        time_step.create_dataset('positions', data=self.simulation.scene.position_array)
        time_step.create_dataset('velocities', data=self.simulation.scene.velocity_array)
        time_step.create_dataset('active', data=self.simulation.scene.active_entries)
        time_step.create_dataset('map', data=self._get_index_array()

        time_step.create_dataset('density', data=self.simulation.effects['repulsion'].density_field.array)
        time_step.create_dataset('velocity_x', data=self.simulation.scene.effects['repulsion'].velocity_field_x.array)
        time_step.create_dataset('velocity_y', data=self.simulation.scene.effects['repulsion'].velocity_field_y.array)
        time_step.create_dataset('pressure', data=self.simulation.effects['repulsion'].pressure_field.array)
        time_step.create_dataset('smoke', data=self.simulation.effects['smoke'].smoke_field.array)

    def _get_index_array(self):
        index_array = np.zeros_like(self.simulation.scene.active_entries) - 1
        for key, val in self.simulation.scene.index_map.items:
            index_array[key] = val.counter
        return index_array


class Sim:
    def __init__(self):
        self.scene = 'scene'
        self.scene_file = 'scenes/test.png'
