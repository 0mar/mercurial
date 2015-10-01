__author__ = 'omar'

import tkinter
import sys

from functions import *
from geometry import Point, Size, LineSegment, Path
from pedestrian import Pedestrian


class VisualScene:
    color_list = ["yellow", "green", "cyan", "magenta"]
    directed_polygon = np.array([[0, -1], [1, 1], [-1, 1]])

    def __init__(self, scene, width, height, step, loop, delay):
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
        self.delay = delay
        self.window = tkinter.Tk()
        self.window.title("Prototype Crowd Simulation")
        self.window.geometry("%dx%d" % (width, height))
        self.window.bind("<Button-3>", self._provide_information)
        if not self.autoloop:
            self.window.bind("<Button-1>", self._advance_simulation)
            self.window.bind("<space>", self._advance_simulation)
        self.canvas = tkinter.Canvas(self.window)
        self.canvas.pack(fill=tkinter.BOTH, expand=1)
        self.draws_cells = True

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
        self.step()  # All functions which should be called on every time step.
        self.draw_scene()
        if self.scene.status == 'DONE':
            fyi("Simulation is finished. Exiting")
            sys.exit(0)

    def loop(self):
        """
        Public interface for visual scene loop. If requested, had a callback reference to itself to keep the simulation going.
        :return: None
        """
        self._advance_simulation(None)
        if self.autoloop:
            self.window.after(self.delay, self.loop)

    def _provide_information(self, event):
        """
        When assigned to a button by tkinter, prints a live summary of the present pedestrians to the screen.
        :param event: Event instance passed by tkinter
        :return: None
        """
        x, y = (event.x / self.size[0], 1 - event.y / self.size[1])
        scene_point = Point([x * self.scene.size[0], y * self.scene.size[1]])
        fyi("Mouse location: %s" % scene_point)
        clicked_cell = self.scene.get_cell_from_position(scene_point)
        print(str(clicked_cell))
        # for ped in self.scene.pedestrian_list:p
        # fyi(str(ped))
        #     fyi("Origin: %s" % ped.origin)

    def draw_scene(self):
        """
        Method that orders the draw commands of all objects within the scene.
        All objects are removed prior to the drawing step. Including a background bitmap will show little improvement
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
        for counter in range(self.scene.pedestrian_number):
            if self.scene.alive_array[counter]:
                self.canvas.create_oval(start_pos_array[counter, 0], start_pos_array[counter, 1],
                                        end_pos_array[counter, 0], end_pos_array[counter, 1],
                                        fill=self.scene.pedestrian_list[counter].color)

    def get_visual_pedestrian_coordinates(self):
        """
        Computes the coordinates of all pedestrian relative to the visualization.
        Uses vectorized operations for speed increments
        :return: relative start coordinates, relative end coordinates.
        """
        rel_pos_array = self.scene.position_array / self.scene.size.array
        rel_size_array = np.ones([self.scene.pedestrian_number,
                                  2]) * self.scene.pedestrian_size.array / self.scene.size.array * self.size.array
        # Todo: Replace collective size by pedestrian size
        vis_pos_array = np.hstack((rel_pos_array[:, 0][:, None], 1 - rel_pos_array[:, 1][:, None])) * self.size.array
        start_pos_array = vis_pos_array - 0.5 * rel_size_array
        end_pos_array = vis_pos_array + 0.5 * rel_size_array
        return start_pos_array, end_pos_array

    def draw_pedestrian(self, ped: Pedestrian):
        """
        Draws a single pedestrian on its relative location in the window as a circle.
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
        :return:
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
