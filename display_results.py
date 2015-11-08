__author__ = 'omar'
import sys, os
import numpy as np
import tkinter
import json, argparse

sys.path.insert(1, 'src')


class Displayer:
    def __init__(self, filename):

        self.delay = 10
        self.obstacle_file = 'scenes/alpha.json'
        self.results_name = 'alphas'
        self.results_folder = 'results/positions'
        self.counter = 1

        self.window = tkinter.Tk()
        self.window.title("Prototype implementation of a Hybrid Crowd Dynamics model for dense crowds")
        self.size = [800, 800]
        self.canvas = tkinter.Canvas(self.window)
        self.canvas.pack(fill=tkinter.BOTH, expand=1)
        self.read_json_file(filename)
        self.iterator = 0
        self.window.after(self.delay, self.step)
        self.window.mainloop()

    def read_json_file(self, filename):
        with open(filename, 'r') as f:
            data = json.loads(f.read())
        self.results_name = data['name']
        self.obstacle_file = data['obstacle_file']
        self.scene_size = data['size']
        self.counter = int(data['number'])
        with open(self.obstacle_file, 'r') as file:
            self.obstacle_data = json.loads(file.read())

    @property
    def size(self):
        return [self.canvas.winfo_width(), self.canvas.winfo_height()]

    @size.setter
    def size(self, value):
        self.window.geometry("%dx%d" % tuple(value))

    def open_arrays(self):
        file_name = "%s-%d" % (self.results_name, self.iterator)
        with open(file_name, 'rb') as f:
            array = np.load(f)
        return array

    def draw_obstacles(self):
        colormap = {'exits': 'red', 'entrances': 'green', 'obstacles': 'grey'}
        for key in self.obstacle_data.keys():
            for obs in self.obstacle_data[key]:
                x_0 = np.array([float(obs['begin'][0]), 1 - float(obs['begin'][1])]) * self.size
                x_1 = x_0 + np.array([float(obs['size'][0]), -float(obs['size'][1])]) * self.size
                color = colormap[key]
                self.canvas.create_rectangle(tuple(x_0) + tuple(x_1), fill=color)

    def display_arrays(self, array):
        """
        Computes the coordinates of all pedestrian relative to the visualization.
        Uses vectorized operations for speed increments
        :return: relative start coordinates, relative end coordinates.
        """
        rel_pos_array = array / self.scene_size
        rel_size_array = np.ones(array.shape) * [0.4, 0.4] / self.scene_size * self.size
        vis_pos_array = np.hstack((rel_pos_array[:, 0][:, None], 1 - rel_pos_array[:, 1][:, None])) * self.size
        start_pos_array = vis_pos_array - 0.5 * rel_size_array
        end_pos_array = vis_pos_array + 0.5 * rel_size_array
        for index in range(array.shape[0]):
            self.canvas.create_oval(start_pos_array[index, 0], start_pos_array[index, 1],
                                    end_pos_array[index, 0], end_pos_array[index, 1], fill='blue')

    def step(self):
        self.iterator += 1
        self.canvas.delete('all')
        self.draw_obstacles()
        array = self.open_arrays()
        self.display_arrays(array)
        self.window.after(self.delay, self.step)


parser = argparse.ArgumentParser(description="Prototype Crowd Dynamics Simulation")
parser.add_argument('-r', '--results', type=str, help='JSON file containing descriptions')
args = parser.parse_args()
display = Displayer(args.results)
