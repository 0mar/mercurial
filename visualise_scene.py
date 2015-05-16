__author__ = 'omar'

import tkinter

from functions import *
from geometry import Point, Size, LineSegment, Path
from pedestrian import Pedestrian
from scene import Obstacle, Scene


class VisualScene:
    color_list = ["yellow", "green", "cyan", "magenta"]
    directed_polygon = np.array([[0, -1], [1, 0], [1, 1], [-1, 1], [-1, 0]])

    def __init__(self, scene: Scene, width, height, step, loop):
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
        # self.scene.evaluate_pedestrians()
        self.step()  # Some functions which should be called on every time step.
        self.draw_scene()

    def loop(self):
        self.advance_simulation(None)
        if self.autoloop:
            self.window.after(1, self.loop)

    def provide_information(self, event):
        x, y = (event.x / self.size[0], 1 - event.y / self.size[1])
        fyi("Mouse location: %s" % Point([x * self.scene.size[0], y * self.scene.size[1]]))
        for ped in self.scene.ped_list:
            fyi(str(ped))
            fyi("Origin: %s" % ped.origin)

    def draw_scene(self):
        self.canvas.delete('all')
        for obstacle in self.scene.obs_list:
            self.draw_obstacle(obstacle)
        for ped in self.scene.ped_list:
            self.draw_pedestrian(ped)

    def draw_pedestrian(self, ped: Pedestrian):
        position = self.convert_relative_coordinate(ped.position / self.scene.size)
        size = ped.size / self.scene.size * self.size
        x_0 = position - size * 0.5
        x_1 = position + size * 0.5
        self.canvas.create_oval(x_0[0], x_0[1], x_1[0], x_1[1], fill=ped.color)

    def draw_directed_pedestrian(self, ped: Pedestrian):
        position = self.convert_relative_coordinate(ped.position / self.scene.size)
        size = ped.size / self.scene.size * self.size
        angle = ped.velocity.angle
        coords = np.dot(VisualScene.directed_polygon * np.array(size), rot_mat(angle)) + np.array(position)
        self.canvas.create_polygon([tuple(array) for array in coords], fill=ped.color)

    def draw_obstacle(self, obstacle: Obstacle):
        x_0 = self.convert_relative_coordinate(obstacle.begin / self.scene.size)
        x_1 = self.convert_relative_coordinate((obstacle.begin + obstacle.size) / self.scene.size)
        self.canvas.create_rectangle(tuple(x_0) + tuple(x_1), fill=obstacle.color)

    def draw_line_segment(self, line_segment: LineSegment):
        x_0 = self.convert_relative_coordinate(line_segment.begin / self.scene.size)
        x_1 = self.convert_relative_coordinate(line_segment.end / self.scene.size)
        self.canvas.create_line(tuple(x_0) + tuple(x_1), fill=line_segment.color)

    def draw_path(self, path: Path):
        for line_segment in path:
            self.draw_line_segment(line_segment)

    def convert_relative_coordinate(self, coord):
        return Size([coord[0], 1 - coord[1]]) * self.size
