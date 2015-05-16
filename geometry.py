#!/usr/bin/env python
import tkinter
import time
from functions import *
from pedestrian import Pedestrian


__author__ = 'omar'


class Scene:
    def __init__(self, size: Size, pedNumber=10, dt=0.05):
        self.size = size
        self.ped_number = pedNumber
        self.dt = dt
        self.obs_list = []
        # Todo: Load different scene files in
        self.obs_list.append(Obstacle(self.size * [0.5, 0.1], self.size * 0.3, "fontein"))
        self.obs_list.append(Obstacle(self.size * 0.7, self.size * 0.1, "hotdogstand"))
        self.obs_list.append(Obstacle(self.size * np.array([0.1, 0.65]), self.size * 0.3, "hotdogstand"))
        self.exit_obs = Exit(Point([self.size.width * 0.3, self.size.height - 1]),
                             Size([self.size.width * 0.4 - 1, 1.]), 'exit obstacle')
        self.obs_list.append(self.exit_obs)
        self.ped_list = [Pedestrian(self, i, self.exit_obs, color=random.choice(VisualScene.color_list)) for i in
                         range(pedNumber)]

    def is_accessible(self, coord: Point, at_start=False) -> bool:
        '''
        Checking whether the coordinate present is an accessible coordinate on the scene.
        When evaluated at the start, the exit is not an accessible object. That would be weird. We can eliminate this later though.
        :param coord: Coordinates to be checked
        :param at_start: Whether to be evaluated at the start
        :return: True if accessible, False otherwise.
        '''
        within_boundaries = all(np.array([0,0]) < coord.array) and all(coord.array < self.size.array)
        if not within_boundaries:
            return False
        if at_start:
            return all([coord not in obstacle for obstacle in self.obs_list])
        else:
            return all([coord not in obstacle or obstacle.permeable for obstacle in self.obs_list])

    def evaluate_pedestrians(self):
        for pedestrian in self.ped_list:
            pedestrian.update_position(self.dt)


class Obstacle:
    def __init__(self, begin: Point, size:Size, name:str, permeable=False):
        self.begin = begin
        self.size = size
        self.end = self.begin + self.size
        self.name = name
        self.permeable = permeable
        self.color = 'black'
        self.corner_info_list = [(Point(self.begin + Size([x, y]) * self.size), [x, y]) for x in range(2) for y in
                                 range(2)]
        self.corner_list = [Point(self.begin + Size([x, y]) * self.size) for x in range(2) for y in range(2)]
        # Safety margin for around the obstacle corners.
        # Todo: Prove that adding a safety margin provides no problem in aggregate objects.
        self.margin_list = [Point(np.sign([x-0.5, y-0.5])) for x in range(2) for y in range(2)]
        self.in_interior = True
        self.center = self.begin + self.size * 0.5

    def __contains__(self, coord:Point):
        return all([self.begin[dim] <= coord[dim] <= self.begin[dim] + self.size[dim] for dim in range(2)])

    def __getitem__(self, item):
        return [self.begin, self.end][item]

    def __repr__(self):
        return "Instance: %s '%s'" % (self.__class__.__name__, self.name)

    def __str__(self):
        return "Obstacle %s. Bottom left: %s, Top right: %s" % (self.name, self.begin, self.end)

        # Pause on wrapping in this method


class Entrance(Obstacle):
    def __init__(self, begin:Point, size:Size, name:str, spawn_rate=0):
        Obstacle.__init__(self, begin, size, name)
        self.spawn_rate = spawn_rate
        self.color = 'blue'


class Exit(Obstacle):
    def __init__(self, begin, size, name):
        Obstacle.__init__(self, begin, size, name, permeable=True)
        self.color = 'red'
        self.margin_list = [Point(np.zeros(2)) for _ in range(4)]
        print(self.margin_list)


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
        self.scene.evaluate_pedestrians()
        self.step()  # Some functions which should be called on every time step.
        self.draw_scene()

    def loop(self):
        self.advance_simulation(None)
        if self.autoloop:
            self.window.after(20, self.loop)

    def provide_information(self, event):
        for ped in self.scene.ped_list:
            print(ped)
            print(ped.origin)

    def draw_scene(self):
        # timing results: Deleting 1000 objects: 0.0003 sec
        #                 Drawing 1000 objects: 0.188 sec
        # Using a bitmap will not be any faster...
        self.canvas.delete('all')
        for obstacle in self.scene.obs_list:
            self.draw_obstacle(obstacle)
        for ped in self.scene.ped_list:
            self.draw_pedestrian(ped)
            self.draw_path(ped.path)

    def draw_pedestrian(self, ped: Pedestrian):
        position = self.convert_relative_coordinate(ped.position / self.scene.size)
        size = ped.size / self.scene.size * self.size
        x_0 = position - size*0.5
        x_1 = position + size*0.5
        self.canvas.create_oval(x_0[0],x_0[1],x_1[0],x_1[1], fill=ped.color)

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
