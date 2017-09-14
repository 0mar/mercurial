#!/usr/bin/env python3
__author__ = 'omar'
import os
import sys

sys.path.insert(0, '../src')
sys.path.insert(0, '..')

import scipy.io as sio
import numpy as np
import matplotlib.colors as mc
import matplotlib.pyplot as plt
from math_objects import functions as ft
from fortran_modules.micro_macro import comp_dens_velo
from fortran_modules.mde import compute_mde

class Processor:
    def __init__(self, result=None,filename=None):
        if not result:
            if not filename:
                filename = "results.mat"
            self.s = 100
            self.norm_l = 0
            self.norm_u = 0.5
            self.alpha = 0.6
            self.dt = 0.05
            self.clip = False
            ft.log("Reading from %s" % filename)
            result_dict = sio.loadmat(filename)

            class MockResult:
                def __init__(self, res_dict):
                    for key in res_dict:
                        setattr(self, key, res_dict[key])

            self.result = MockResult(result_dict)
            self.result.finished = self.result.finished.astype(bool).flatten()
        else:
            self.result = result

    def delay_scatter_plot(self):
        if self.clip:
            norm = mc.Normalize(self.norm_l, self.norm_u, False)
        else:
            norm = None
        delay = np.maximum(1 - self.result.path_length_ratio, 0)
        positions = self.result.origins[self.result.finished]
        plt.scatter(x=positions[:, 0], y=positions[:, 1], c=delay, s=self.s, norm=norm, alpha=self.alpha)
        # plt.xlim(-10, 80)
        # plt.ylim(40, 80)
        plt.xlabel('x-coordinate of scene')
        plt.ylabel('y-coordinate of scene')
        plt.suptitle('Average relative delay as a function of initial location')
        plt.colorbar()
        plt.show()

    def time_scatter_plot(self, clip=False):
        if clip:
            norm = mc.Normalize(-0.1, 0.2, False)
        else:
            norm = None
        time = self.result.time_spent.flatten()[self.result.finished]
        positions = self.result.origins[self.result.finished]
        plt.scatter(positions[:, 0], positions[:, 1], c=time, s=self.s, norm=norm, alpha=self.alpha)
        # plt.xlim(-10, 80)
        # plt.ylim(45, 80)
        plt.xlabel('x-coordinate of scene')
        plt.ylabel('y-coordinate of scene')
        plt.suptitle('Time to exit in seconds as a function of initial location')
        plt.colorbar()
        plt.show()

    def time_spent_histogram(self):
        if self.result.time_spent.size > 1:
            time = (self.result.time_spent.flatten()[self.result.finished] / self.dt).T
            plt.hist(time, bins=50)
            plt.xlabel('Time steps to reach exit')
            plt.ylabel('Number of pedestrians')
            plt.suptitle('Histogram of pedestrian walking time')
            plt.show()
        else:
            ft.warn("No histogram made, insufficient data set (size %d)" % len(self.result.time_spent))

    def path_length_histogram(self):
        if self.result.path_length.size > 1:
            # Consistent with time histogram
            path = (self.result.path_length.flatten()[self.result.finished] / self.dt).T
            plt.hist(path, bins=50)
            plt.xlabel('Time steps to reach exit')
            plt.ylabel('Number of pedestrians')
            plt.suptitle('Histogram of pedestrian distance time')
            plt.show()
        else:
            ft.warn("No histogram made, insufficient data set (size %d)" % len(self.result.time_spent))

    def density_map(self):
        if self.result.position_list.size > 1:
            positions = self.result.position_list[:, 0:2]
            dummy_velo = np.random.random(positions.shape)
            active = np.ones(positions.shape[0], dtype=bool)
            size_x = np.max(positions[:, 0])
            size_y = np.max(positions[:, 1])
            nx, ny = 400, 400
            dx, dy = size_x / nx, size_y / ny
            dens, _, _ = comp_dens_velo(positions, dummy_velo, active, nx, ny, dx, dy, 5 * dx)
            plt.imshow(np.log(1 + np.rot90(dens)))
            plt.xlabel('x-coordinate in scene')
            plt.ylabel('y-coordinate in scene')
            plt.suptitle('Logarithmic density heatmap')
            plt.show()

    def mde_violations(self):
        active_arrays = np.ones(self.result.final_positions.shape[0])
        size_x = np.max(self.result.final_positions[:, 0]) + 1
        size_y = np.max(self.result.final_positions[:, 1]) + 1
        min_dist = 0.05
        max_dist = 0.8
        resolution = 100
        distances = np.linspace(min_dist, max_dist, resolution)
        vio_amount = np.zeros(distances.shape)
        for i, distance in enumerate(distances):
            mde = compute_mde(self.result.final_positions, size_x, size_y, active_arrays, distance)
            mde_found = np.where(np.sum(np.abs(mde), axis=1) > 0)[0]
            vio_amount[i] = len(mde_found)
        plt.plot(distances, vio_amount)
        plt.xlabel('Range r')
        plt.ylabel('Number of particles')
        plt.suptitle('Particles with other particles in range r')
        sio.savemat('vio', {'distances': distances, 'vio': vio_amount})
        plt.show()


if __name__ == '__main__':
    filename = None
    if len(sys.argv)>1:
        filename = sys.argv[1]
        if not os.path.exists(filename):
            ft.error("Result file %s does not exist"%filename)
    proc = Processor(filename=filename)
    # proc.mde_violations()
    proc.time_spent_histogram()
    # proc.path_length_histogram()
    proc.delay_scatter_plot()
    proc.time_scatter_plot()
    proc.density_map()
