#!/usr/bin/env python
import tkinter
from functions import *
from pedestrian import Pedestrian
from planner import Planner

__author__ = 'omar'


class Scene:
    def __init__(self, size, pedNumber=10, dt=0.05):
        self.size = size
        self.ped_number = pedNumber
        self.dt = dt
        self.obs_list = []
        for boundary in self.__create_boundaries():
            self.obs_list.append(boundary)
            boundary.in_interior = False
        self.obs_list.append(Obstacle(self.size * [0.5,0.1], self.size * 0.3, "fontein"))
        self.obs_list.append(Obstacle(self.size * 0.7, self.size * 0.1, "hotdogstand"))
        self.obs_list.append(Obstacle(self.size * np.array([0.1,0.65]), self.size * 0.3, "hotdogstand"))
        self.ped_list = [Pedestrian(self, i, self.exit_obs, color=random.choice(VisualScene.color_list)) for i in
                         range(pedNumber)]

    def __create_boundaries(self):
        left_wall = Obstacle(Point([0., 0.]), Size([1., self.size.height]), 'left wall')
        right_wall = Obstacle(Point([self.size.width - 1., 0.]), Size([1., self.size.height]), 'right wall')
        top_wall_1 = Obstacle(Point([0., self.size.height - 1]), Size([self.size.width * 0.3 - 1, 1.]), 'top wall_left')
        top_wall_2 = Obstacle(Point([self.size.width * 0.7, self.size.height - 1]), Size([self.size.width * 0.3, 1.]),
                              'top wall_right')
        self.exit_obs = Exit(Point([self.size.width * 0.3, self.size.height - 1]),
                             Size([self.size.width * 0.4 - 1, 1.]), 'exit obstacle')
        bottom_wall = Obstacle(Point([0., 0.]), Size([self.size.width, 1.]), 'bottom wall')
        return [left_wall, right_wall, top_wall_1, top_wall_2, self.exit_obs, bottom_wall]

    def is_accessible(self, coord, at_start=False):
        if at_start:
            return all([coord not in obstacle for obstacle in self.obs_list])
        else:
            return all([coord not in obstacle or obstacle.permeable for obstacle in self.obs_list])

    def evaluate_pedestrians(self):
        for pedestrian in self.ped_list:
            pedestrian.update_position(self.dt)


class Obstacle:
    def __init__(self, begin, size, name, permeable=False):
        self.begin = begin
        self.size = size
        self.end = self.begin + self.size
        self.name = name
        self.permeable = permeable
        self.color = 'black'
        self.corner_info_list = [(Point(self.begin + Size([x, y]) * self.size), [x, y]) for x in range(2) for y in
                                 range(2)]
        # todo: revert corner_info_list to corner_list
        self.in_interior = True
        self.center = self.begin + self.size * 0.5

    def __contains__(self, coord):
        return all([self.begin[dim] <= coord[dim] <= self.begin[dim] + self.size[dim] for dim in range(2)])

    def __getitem__(self, item):
        return [self.begin, self.end][item]

    def __repr__(self):
        return "Instance: %s '%s'" % (self.__class__.__name__, self.name)

    def __str__(self):
        return "Obstacle %s. Bottom left: %s, Top right: %s" % (self.name, self.begin, self.end)

        # Pause on wrapping in this method


class Entrance(Obstacle):
    def __init__(self, begin, size, name, spawn_rate=0):
        Obstacle.__init__(self, begin, size, name)
        self.spawn_rate = spawn_rate
        self.color = 'blue'


class Exit(Obstacle):
    def __init__(self, begin, size, name):
        Obstacle.__init__(self, begin, size, name, permeable=True)
        self.color = 'red'



class VisualScene:
    color_list = ["yellow", "green", "cyan", "magenta"]
    directed_polygon = np.array([[0, -1], [1, 0], [1, 1], [-1, 1], [-1, 0]])

    def __init__(self, scene, width, height, step,loop):
        self.step = step
        self.scene = scene
        self.autoloop = loop
        self.window = tkinter.Tk()
        self.window.title("Prototype Crowd Simulation")
        self.window.geometry("%dx%d" % (width, height))
        self.window.bind("<Button-3>", self.provide_information)
        if not self.autoloop:
            self.window.bind("<Button-1>", self.advance_simulation)
        self.canvas = tkinter.Canvas(self.window)
        self.canvas.pack(fill=tkinter.BOTH, expand=1)

    @property
    def size(self):
        return Size([self.canvas.winfo_width(), self.canvas.winfo_height()])

    @size.setter
    def width(self, value):
        self.window.geometry("%dx%d" % tuple(value))

    def advance_simulation(self, event):
        self.scene.evaluate_pedestrians()
        self.step()  # Some functions which should be called on every time step.
        self.draw_scene()

    def loop(self):
        self.advance_simulation(None)
        if self.autoloop:
            self.window.after(20, self.loop)

    def provide_information(self,event):
        for ped in self.scene.ped_list:
            print(ped)
            print(ped.origin)

    def draw_scene(self):
        self.canvas.delete('all')

        for obstacle in self.scene.obs_list:
            self.draw_obstacle(obstacle)
        for ped in self.scene.ped_list:
            self.draw_directed_pedestrian(ped)

    def draw_pedestrian(self, ped):
        rel_pos = ped.position / self.scene.size
        rel_size = ped.size / self.scene.size

        x_0 = self.convert_relative_coordinate(rel_pos - rel_size * 0.5)
        x_1 = self.convert_relative_coordinate(rel_pos + rel_size * 0.5)
        self.canvas.create_oval(x_0 + x_1, fill=ped.color)

    def draw_directed_pedestrian(self, ped):
        position = self.convert_relative_coordinate(ped.position / self.scene.size)
        size = ped.size / self.scene.size * self.size
        angle = ped.velocity.angle
        coords = np.dot(VisualScene.directed_polygon * np.array(size), rot_mat(angle)) + np.array(position)
        self.canvas.create_polygon([tuple(array) for array in coords], fill=ped.color)

    def draw_obstacle(self, obstacle):
        x_0 = self.convert_relative_coordinate(obstacle.begin / self.scene.size)
        x_1 = self.convert_relative_coordinate((obstacle.begin + obstacle.size) / self.scene.size)
        self.canvas.create_rectangle(tuple(x_0) + tuple(x_1), fill=obstacle.color)

    def draw_line_segment(self, line_segment):
        x_0 = self.convert_relative_coordinate(line_segment.begin / self.scene.size)
        x_1 = self.convert_relative_coordinate(line_segment.end / self.scene.size)
        self.canvas.create_line(tuple(x_0) + tuple(x_1), fill=line_segment.color)

    def draw_path(self, path):
        for line_segment in path:
            self.draw_line_segment(line_segment)

    def convert_relative_coordinate(self, coord):
        return Size([coord[0], 1 - coord[1]]) * self.size
