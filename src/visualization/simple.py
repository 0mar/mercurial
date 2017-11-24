import time
import tkinter
from PIL import Image,ImageTk
import numpy as np

from math_objects import functions as ft
from math_objects.geometry import Point, Size, LineSegment, Path


class VisualScene:
    """
    Simple visual interface based on TKinter. Please replace with awesome visual interface.
    """
    color_list = ["yellow", "green", "cyan", "magenta"]
    directed_polygon = np.array([[0, -1], [1, 1], [-1, 1]])

    def __init__(self, scene):
        """
        Initializes a visual interface for the simulation. Updates every fixed amount of seconds.
        Represents the scene on a canvas.
        Important: this class progresses the simulation. After each drawing and potential delay,
        the visualisation calls for the progression to the next time step.
        Although it might be cleaner to move that to the simulation manager.
        :param scene: Scene to be drawn. The size of the scene is independent of the size of the visualization
        :return: None
        """
        self.config = scene.config
        init_size = Size(
            [self.config['visual'].getfloat('screen_size_x'), self.config['visual'].getfloat('screen_size_y')])
        self.scene = scene
        self.autoloop = True
        self.draws_cells = self.config['visual'].getboolean('draw_grid')
        self.delay = self.config['visual'].getint('time_delay')
        self.window = tkinter.Tk()
        self.window.title("Mercurial: Hybrid simulation for dense crowds")
        self.window.geometry("%dx%d" % (init_size[0], init_size[1]))
        self.window.bind("<Button-3>", self.store_scene)
        self.window.grid()
        env_image = self.config['general']['scene']
        self.original_env = Image.open(env_image)
        self.env = ImageTk.PhotoImage(self.original_env)
        self.display = tkinter.Canvas(self, bd=0, highlightthickness=0)
        self.display.create_image(0, 0, image=self.env, anchor=tkinter.NW, tags="IMG")
        self.display.grid(row=0, sticky=tkinter.W + tkinter.E + tkinter.N + tkinter.S)
        # self.canvas.pack(fill=tkinter.BOTH, expand=1)
        self.window.pack(fill=tkinter.BOTH,expand=1)
        self.window.bind("<Configure>",self.resize)
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

    def resize(self,event):
        size = (event.width, event.height)
        resized = self.original_env.resize(size, Image.ANTIALIAS)
        self.env = ImageTk.PhotoImage(resized)
        self.display.delete("IMG")
        self.display.create_image(0, 0, image=self.env, anchor=tkinter.NW, tags="IMG")

    def disable_loop(self):
        """
        Stop automatically redrawing.
        Enables space and Left-mouse click as progressing simulation.
        :return: None
        """
        self.autoloop = False
        self.window.bind("<Button-1>", self.loop)
        self.window.bind("<space>", self.loop)

    def loop(self, _=None):
        """
        Public interface for visual scene loop. If required, has a callback reference to itself to keep the simulation going.
        :param _: Event object from tkinter
        :return: None
        """
        if self.scene.status == 'DONE':
            self.window.destroy()
            self.autoloop = False
        else:
            self.draw_scene()
            if self.scene.counter == 630:
                self.store_scene(None)
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
        for pedestrian in self.scene.pedestrian_list:
            print(pedestrian)
        for obstacle in self.scene.obstacle_list:
            print(obstacle)

    def _give_relative_position(self, event):
        x, y = (event.x / self.size[0], 1 - event.y / self.size[1])
        ft.log("Mouse location: (%.2f,%.2f)" % (x, y))

    def draw_scene(self):
        """
        Method that orders the draw commands of all objects within the scene.
        All objects are removed prior to the drawing step.
        :return: None
        """
        # self.canvas.delete('all')
        self.draw_pedestrians()

    def store_scene(self, _, filename=None):

        directory = 'images'
        if not filename:
            import time

            name = "scene#%d" % time.time()
            filename = "%s/%s-%.2f.eps" % (directory, name, self.scene.time)
        print("Snapshot at %.2f. Storing in %s" % (self.scene.time,filename))
        self.canvas.postscript(file=filename, pageheight=self.size[1], pagewidth=self.size[0])

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
        vis_pos_array = np.hstack((rel_pos_array[:, 0][:, None], 1 - rel_pos_array[:, 1][:, None])) * self.size.array
        start_pos_array = vis_pos_array - 0.5 * rel_size_array
        end_pos_array = vis_pos_array + 0.5 * rel_size_array
        return start_pos_array, end_pos_array

    def draw_grid(self, nx, ny):
        """
        Draws a grid
        :param nx: number of cells in x direction
        :param ny: number of cells in y direction
        :return: None
        """

        screen_dx = self.size[0] / nx
        screen_dy = self.size[1] / ny
        for i in range(nx - 1):
            x_coord = (i + 1) * screen_dx
            self.canvas.create_line(x_coord, 0, x_coord, self.size[1])
        for j in range(ny - 1):
            y_coord = (j + 1) * screen_dy
            self.canvas.create_line(0, y_coord, self.size[0], y_coord)

    # def draw_cell(self, cell):
    #     """
    #     Draws a cell as a rectangle within the window.
    #     Only use for debugging purposes, since drawing is very inefficient.
    #     :param cell: Cell object to be drawn
    #     :return: None
    #     """
    #     x_0 = self.convert_relative_coordinate(cell.begin / self.scene.size)
    #     x_1 = self.convert_relative_coordinate((cell.begin + cell.size) / self.scene.size)
    #     self.canvas.create_rectangle(tuple(x_0) + tuple(x_1), outline='blue')

    # def draw_rect_obstacle(self, obstacle):
    #     """
    #     Draws a rectangular obstacle on its relative location in the screen
    #     :param obstacle: obstacle to be drawn. Obstacles are black by default, exits are red, entrances are blue.
    #     :return: None
    #     """
    #     x_0 = self.convert_relative_coordinate(obstacle.begin / self.scene.size)
    #     x_1 = self.convert_relative_coordinate((obstacle.begin + obstacle.size) / self.scene.size)
    #     self.canvas.create_rectangle(tuple(x_0) + tuple(x_1), fill=obstacle.color)

    def draw_circ_obstacle(self, circ_object):
        """
        Draw a circular object in the middle of the scene
        :param circ_object: object object to be drawn. Needs a center, radius and color
        :return: None
        """
        rel_pos_array = circ_object.center.array / self.scene.size.array
        rel_size_array = circ_object.radius / self.scene.size.array * self.size.array
        vis_pos_array = np.array([rel_pos_array[0], 1 - rel_pos_array[1]]) * self.size.array
        start_pos_array = vis_pos_array - 0.5 * rel_size_array
        end_pos_array = vis_pos_array + 0.5 * rel_size_array
        self.canvas.create_oval(start_pos_array[0], start_pos_array[1], end_pos_array[0], end_pos_array[1],
                                fill=circ_object.color)

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
        Draws a path in the scene on its relative location. Wrapper function for drawing multiple line segments.
        Not really used except for demonstrations
        :param path:
        :return: None
        """
        for line_segment in path:
            self.draw_line_segment(line_segment)

    def convert_relative_coordinate(self, coord):
        """
        Converts relative coordinates (from [0,1]x[0,1]) to screen size coordinates.
        Should raise an error when coordinates fall from scene,
        but this method is used so frequently I'd rather not make the computation
        Also changes the orientation to a Carthesian coordinate system
        :param coord: coordinates (fractions to be converted)
        :return: a Size with the coordinates of screen
        """
        return Size([coord[0], 1 - coord[1]]) * self.size
