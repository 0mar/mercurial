#!/usr/bin/env python3
__author__ = 'omar'
import sys

sys.path.insert(0, '../src')
import scipy.io as sio
import numpy as np
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import functions as ft


class Processor:
    def __init__(self, result=None):
        if not result:
            filename = "results.mat"
            self.s = 100
            self.norm_l = -0.1
            self.norm_u = 1
            self.alpha = 0.6
            self.dt = 0.05
            self.clip = False
            ft.log("No results passed, reading from %s" % filename)
            result_dict = sio.loadmat(filename)

            class MockResult:
                def __init__(self, res_dict):
                    for key in res_dict:
                        setattr(self, key, res_dict[key])

            self.result = MockResult(result_dict)
            # one time hack
            self.result.finished = np.ones([10000], dtype=bool)
            unfinished = [129, 715, 1575, 2166, 3630, 3713, 4513, 5163, 5850, 7753, 8220, 9379, 9573, 9764, 9804]
            self.result.finished[unfinished] = 0
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
        #plt.ylim(40, 80)
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
        #plt.ylim(45, 80)
        plt.xlabel('x-coordinate of scene')
        plt.ylabel('y-coordinate of scene')
        plt.suptitle('Time to exit in seconds as a function of initial location')
        plt.colorbar()
        plt.show()

    def time_spent_histogram(self):
        if self.result.time_spent.size > 1:
            time = self.result.time_spent.T / self.dt
            plt.hist(time, bins=50)
            plt.xlabel('Time steps to reach exit')
            plt.ylabel('Number of pedestrians')
            plt.suptitle('Histogram of pedestrian walking time')
            plt.show()
        else:
            ft.warn("No histogram made, insufficient data set (size %d)" % len(self.result.time_spent))

if __name__ == '__main__':
    proc = Processor()
    proc.time_spent_histogram()
    proc.delay_scatter_plot()
    proc.time_scatter_plot()
