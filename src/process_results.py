__author__ = 'omar'
import scipy.io as sio
import matplotlib.colors as mc
import matplotlib.pyplot as plt

import functions as ft


class Processor:
    def __init__(self, result=None):
        if not result:
            filename = "../results/results.mat"
            ft.log("No results passed, reading from %s" % filename)
            result_dict = sio.loadmat(filename)

            class MockResult:
                def __init__(self, res_dict):
                    for key in res_dict:
                        setattr(self, key, res_dict[key])

            self.result = MockResult(result_dict)
        else:
            self.result = result

    def delay_scatter_plot(self, clip=True):
        if clip:
            norm = mc.Normalize(-0.1, 0.2, False)
        else:
            norm = None
        delay = 1 - self.result.path_length_ratio
        positions = self.result.origins
        plt.scatter(positions[:, 0], positions[:, 1], c=delay, s=100, norm=norm, alpha=0.6)
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
        time = self.result.time_spent
        positions = self.result.origins
        plt.scatter(positions[:, 0], positions[:, 1], c=time, s=100, norm=norm, alpha=0.6)
        plt.xlabel('x-coordinate of scene')
        plt.ylabel('y-coordinate of scene')
        plt.suptitle('Time to exit in seconds as a function of initial location')
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    proc = Processor()
    proc.time_scatter_plot()
