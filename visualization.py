__author__ = 'omar'

import tkinter

from functions import *
from geometry import Point, Size, LineSegment, Path
from pedestrian import Pedestrian
import scene


class VisualScene:
    color_list = ["yellow", "green", "cyan", "magenta"]
    directed_polygon = np.array([[0, -1], [1, 1], [-1, 1]])

    def __init__(self, scene, width, height, step, loop):
        """
        Initializes a visual interface for the simulation. Updates every fixed amount of seconds.
        Represents the scene on a canvas
        :param scene: Scene to be drawn. The size of the scene is independent of the size of the visualization
        :param width: width of the window
        :param height: height of the window
        :param step: Function that needs to be executed every time step
        :param loop: boolean for determining whether to update automatically
        :return:
        """
        self.step = step
        self.scene = scene
        self.autoloop = loop
        self.window = tkinter.Tk()
        self.window.title("Prototype Crowd Simulation")
        self.window.geometry("%dx%d" % (width, height))
        self.window.bind("<Button-3>", self._provide_information)
        if not self.autoloop:
            self.window.bind("<Button-1>", self._advance_simulation)
        self.canvas = tkinter.Canvas(self.window)
        self.canvas.pack(fill=tkinter.BOTH, expand=1)

    @property
    def size(self):
        return Size([self.canvas.winfo_width(), self.canvas.winfo_height()])

    @size.setter
    def size(self, value):
        self.window.geometry("%dx%d" % tuple(value))

    def _advance_simulation(self, event):
        """
        Method required for moving the simulation forward one time unit
        :param event: Event originating from the tkinter update function
        :return: None
        """
        self.step()  # Some functions which should be called on every time step.
        self.draw_scene()

    def loop(self):
        """
        Public interface for visual scene loop. If requested, had a callback reference to itself to keep the simulation going.
        :return: None
        """
        self._advance_simulation(None)
        if self.autoloop:
            self.window.after(1, self.loop)

    def _provide_information(self, event):
        """
        When assigned to a button by tkinter, prints a live summary of the present pedestrians to the screen.
        :param event: Event instance passed by tkinter
        :return:
        """
        x, y = (event.x / self.size[0], 1 - event.y / self.size[1])
        fyi("Mouse location: %s" % Point([x * self.scene.size[0], y * self.scene.size[1]]))
        for ped in self.scene.ped_list:
            fyi(str(ped))
            fyi("Origin: %s" % ped.origin)

    def draw_scene(self):
        """
        Method that orders the draw commands of all objects within the scene.
        All objects are removed prior to the drawing step. Including a background bitmap will show little improvement
        :return: None
        """
        self.canvas.delete('all')
        for obstacle in self.scene.obs_list:
            self.draw_obstacle(obstacle)
        for ped in self.scene.ped_list:
            self.draw_pedestrian(ped)

    def draw_pedestrian(self, ped: Pedestrian):
        """
        Draws a pedestrian on its relative location in the window as a circle.
        :param ped: pedestrian which is represented on screen
        """
        position = self.convert_relative_coordinate(ped.position / self.scene.size)
        size = ped.size / self.scene.size * self.size
        x_0 = position - size * 0.5
        x_1 = position + size * 0.5
        self.canvas.create_oval(x_0[0], x_0[1], x_1[0], x_1[1], fill=ped.color)

    def draw_directed_pedestrian(self, ped: Pedestrian):
        """
        Draws a pedestrian on its relative location in the window as a triangle.
        :param ped: pedestrian which is represented on the screen
        :return:
        """
        position = self.convert_relative_coordinate(ped.position / self.scene.size)
        size = ped.size / self.scene.size * self.size
        angle = ped.velocity.angle
        coords = np.dot(VisualScene.directed_polygon * np.array(size), rot_mat(angle)) + np.array(position)
        self.canvas.create_polygon([tuple(array) for array in coords], fill=ped.color)

    def draw_obstacle(self, obstacle):
        """
        Draws a rectangular obstacle on its relative location in the screen
        :param obstacle: obstacle to be drawn. Obstacles are black by default, exits are red, entrances are blue.
        :return: None
        """
        x_0 = self.convert_relative_coordinate(obstacle.begin / self.scene.size)
        x_1 = self.convert_relative_coordinate((obstacle.begin + obstacle.size) / self.scene.size)
        self.canvas.create_rectangle(tuple(x_0) + tuple(x_1), fill=obstacle.color)

    def draw_line_segment(self, line_segment: LineSegment):
        """
        Draws a line segment in the scene on its relative location. Can be used for (debugging) paths
        :param line_segment: line segment to be drawn.
        :return: None
        """
        x_0 = self.convert_relative_coordinate(line_segment.begin / self.scene.size)
        x_1 = self.convert_relative_coordinate(line_segment.end / self.scene.size)
        self.canvas.create_line(tuple(x_0) + tuple(x_1), fill=line_segment.color)

    def draw_path(self, path: Path):
        """
        Draws a path in the scene on its relative location. Wrapper function for drawing multiple line segments
        :param path:
        :return:
        """
        for line_segment in path:
            self.draw_line_segment(line_segment)

    def convert_relative_coordinate(self, coord):
        """
        Converts relative coordinates (from [0,1]x[0,1]) to screen size coordinates.
        Should raise an error when coordinates fall from scene, but method is so frequently used I'd rather not
        Also changes the orientation to a Carthesian coordinate system
        :param coord: coordinates (fractions to be converted)
        :return: a Size with the coordinates of screen
        """
        return Size([coord[0], 1 - coord[1]]) * self.size