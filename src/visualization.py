__author__ = 'omar'
import tkinter

import numpy as np

import functions as ft
from geometry import Point, Size, LineSegment, Path



class VisualScene:
    color_list = ["yellow", "green", "cyan", "magenta"]
    directed_polygon = np.array([[0, -1], [1, 1], [-1, 1]])

    def __init__(self, scene, config):
        """
        Initializes a visual interface for the simulation. Updates every fixed amount of seconds.
        Represents the scene on a canvas
        :param scene: Scene to be drawn. The size of the scene is independent of the size of the visualization
        :return: None
        """
        self.config = config
        init_size = Size([config['visual'].getfloat('screen_size_x'), config['visual'].getfloat('screen_size_y')])
        self.scene = scene
        self.autoloop = True
        self.draws_cells = False
        self.delay = config['visual'].getint('time_delay')
        self.window = tkinter.Tk()
        self.window.title("Prototype implementation of a Hybrid Crowd Dynamics model for dense crowds")
        self.window.geometry("%dx%d" % (init_size.width, init_size.height))
        self.window.bind("<Button-3>", self._provide_information)
        self.canvas = tkinter.Canvas(self.window)
        self.canvas.pack(fill=tkinter.BOTH, expand=1)
        self.step_callback = None  # set in manager


    @property
    def size(self):
        return Size([self.canvas.winfo_width(), self.canvas.winfo_height()])

    @size.setter
    def size(self, value):
        self.window.geometry("%dx%d" % tuple(value))

    def start(self):
        self.loop()
        self.window.mainloop()

    def disable_loop(self):
        self.autoloop = False
        self.window.bind("<Button-1>", self.loop)
        self.window.bind("<space>", self.loop)

    def loop(self, _=None):
        """
        Public interface for visual scene loop. If required, has a callback reference to itself to keep the simulation going.
        :param _: Event object from tkinter
        :return: None
        """
        self.draw_scene()
        if self.scene.status == 'DONE':
            self.window.destroy()
            self.autoloop = False
        if self.autoloop:
            self.window.after(self.delay, self.step_callback)
        else:
            self.step_callback()

    def _provide_information(self, event):
        """
        When assigned to a button by tkinter, prints a live summary of the present pedestrians to the screen.
        :param event: Event instance passed by tkinter
        :return: None
        """
        x, y = (event.x / self.size[0], 1 - event.y / self.size[1])
        scene_point = Point([x * self.scene.size[0], y * self.scene.size[1]])
        ft.log("Mouse location: %s" % scene_point)

    def draw_scene(self):
        """
        Method that orders the draw commands of all objects within the scene.
        All objects are removed prior to the drawing step.
        :return: None
        """
        self.canvas.delete('all')
        if self.draws_cells:
            for cell in self.scene.cell_dict.values():
                self.draw_cell(cell)
        for obstacle in self.scene.obstacle_list:
            self.draw_obstacle(obstacle)
        self.draw_pedestrians()

    def store_scene(self, name=None):
        directory = 'images'
        if not name:
            import time

            name = "scene#%d" % time.time()
        filename = "%s/%s-%d.eps" % (directory, name, self.scene.time)
        self.canvas.postscript(file=filename)

    def draw_pedestrians(self):
        """
        Draws all the pedestrians in the scene using the visual_pedestrian coordinates.
        :return: None
        """
        start_pos_array, end_pos_array = self.get_visual_pedestrian_coordinates()
        for pedestrian in self.scene.pedestrian_list:
            index = pedestrian.index
            self.canvas.create_oval(start_pos_array[index, 0], start_pos_array[index, 1],
                                    end_pos_array[index, 0], end_pos_array[index, 1],
                                    fill=pedestrian.color)

    def get_visual_pedestrian_coordinates(self):
        """
        Computes the coordinates of all pedestrian relative to the visualization.
        Uses vectorized operations for speed increments
        :return: relative start coordinates, relative end coordinates.
        """
        rel_pos_array = self.scene.position_array / self.scene.size.array
        rel_size_array = np.ones(
            self.scene.position_array.shape) * self.scene.pedestrian_size.array / self.scene.size.array * self.size.array
        vis_pos_array = np.hstack((rel_pos_array[:, 0][:,None], 1 - rel_pos_array[:, 1][:,None]))* self.size.array
        start_pos_array = vis_pos_array - 0.5 * rel_size_array
        end_pos_array = vis_pos_array + 0.5 * rel_size_array
        return start_pos_array, end_pos_array

    def draw_cell(self, cell):
        """
        Draws a cell as a rectangle within the window.
        Only use for debugging purposes, since drawing is very inefficient.
        :param cell: Cell object to be drawn
        :return: None
        """
        x_0 = self.convert_relative_coordinate(cell.begin / self.scene.size)
        x_1 = self.convert_relative_coordinate((cell.begin + cell.size) / self.scene.size)
        self.canvas.create_rectangle(tuple(x_0) + tuple(x_1), outline='blue')

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
        self.canvas.create_line(tuple(x_0) + tuple(x_1), fill=line_segment.color, width=2)

    def draw_path(self, path: Path):
        """
        Draws a path in the scene on its relative location. Wrapper function for drawing multiple line segments
        :param path:
        :return: None
        """
        for line_segment in path:
            self.draw_line_segment(line_segment)

    def convert_relative_coordinate(self, coord):
        """
        Converts relative coordinates (from [0,1]x[0,1]) to screen size coordinates.
        Should raise an error when coordinates fall from scene,
        but method is so frequently used I'd rather not make the computation
        Also changes the orientation to a Carthesian coordinate system
        :param coord: coordinates (fractions to be converted)
        :return: a Size with the coordinates of screen
        """
        return Size([coord[0], 1 - coord[1]]) * self.size


class NoVisualScene(VisualScene):
    def __init__(self, scene):
        self.scene = scene

    def start(self):
        while not self.scene.status == 'DONE':
            self.step_callback()
            ft.log("Time step %d" % self.scene.counter)
