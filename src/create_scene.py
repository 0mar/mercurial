__author__ = 'omar'
import tkinter
from enum import Enum
import json
import tkinter.simpledialog

import functions as ft


class SceneCreator:
    class DrawStatus(Enum):
        idle = 0
        drawing = 1

    class ObjectStatus(Enum):
        obstacle = 0
        exit = 1

    def __init__(self):
        self.window = tkinter.Tk()
        self.window.title("Scene creator")
        self.size = (1000, 1000)
        self.window.geometry("1000x1000")
        self.window.bind('<Button-1>', self.start_drawing)

        self.canvas = tkinter.Canvas(self.window)
        self.canvas.pack(fill=tkinter.BOTH, expand=1)
        self.draw_status = SceneCreator.DrawStatus.idle
        self.object_status = SceneCreator.ObjectStatus.obstacle
        self.menu = tkinter.Menu(self.window)
        self.menu.add_command(label='Save scene', command=self.ask_save)
        self.menu.add_command(label='Quit', command=self.window.destroy)
        self.object_status = tkinter.IntVar()
        self.menu.add_radiobutton(label='Obstacles', variable=self.object_status, value=0)
        self.menu.add_radiobutton(label='Exits', variable=self.object_status, value=1)
        self.window.config(menu=self.menu)

        ft.debug("Started creating scenes.")

        self.obstacle_list = []
        self.new_object_holder = 0
        self.draw_start = self.draw_end = None
        self.exit_list = []

    def start_drawing(self, event):
        ft.debug(self.object_status.get())
        self.draw_status = SceneCreator.DrawStatus.drawing
        self.draw_start = (event.x, event.y)
        self.window.bind('<Motion>', self.motion)
        self.window.bind('<Button-1>', self.stop_drawing)
        ft.debug("Started drawing. Start location of obstacle: %s" % str(self.draw_start))

    def stop_drawing(self, event):
        ft.debug("stopped drawing")
        self.window.unbind('<Motion>')
        self.draw_status = SceneCreator.DrawStatus.idle
        self.draw_end = (event.x, event.y)
        ft.debug("End of obstacle: %s" % str(self.draw_end))
        new_obstacle = self.create_new_obstacle()
        self.draw_start = self.draw_end = None
        self.window.bind('<Button-1>', self.start_drawing)
        self.draw_obstacle(new_obstacle)

    def draw_scene(self):
        if self.draw_status == SceneCreator.DrawStatus.drawing:
            self.redraw_new_obstacle(self.draw_start, self.draw_end)

    def draw_obstacle(self, obstacle):
        draw_coords = self.convert_relative_coordinate(obstacle.start) + self.convert_relative_coordinate(obstacle.end)
        ft.debug("Drawing obstacle on %s" % str(draw_coords))
        if SceneCreator.ObjectStatus(self.object_status.get()) == SceneCreator.ObjectStatus.obstacle:
            color = 'gray'
        else:
            color = 'red'
        self.canvas.create_rectangle(draw_coords, fill=color)

    def redraw_new_obstacle(self, start, end):
        if self.new_object_holder:
            self.canvas.delete(self.new_object_holder)
        self.new_object_holder = self.canvas.create_rectangle(start, end, fill=None)

    def motion(self, event):
        assert self.draw_status == SceneCreator.DrawStatus.drawing
        cur_pos = (event.x, event.y)
        self.redraw_new_obstacle(self.draw_start, cur_pos)

    def create_new_obstacle(self):
        begin_x = min(self.draw_start[0], self.draw_end[0]) / self.size[0]
        begin_y = 1 - max(self.draw_start[1], self.draw_end[1]) / self.size[1]
        end_x = max(self.draw_start[0], self.draw_end[0]) / self.size[0]
        end_y = 1 - min(self.draw_start[1], self.draw_end[1]) / self.size[1]
        if SceneCreator.ObjectStatus(self.object_status.get()) == SceneCreator.ObjectStatus.obstacle:
            new_obstacle = Obstacle((begin_x, begin_y), (end_x - begin_x, end_y - begin_y))
            self.obstacle_list.append(new_obstacle)
            return new_obstacle
        else:
            new_exit = Exit((begin_x, begin_y), (end_x - begin_x, end_y - begin_y))
            self.exit_list.append(new_exit)
            return new_exit

    def ask_save(self):
        save_name = tkinter.simpledialog.askstring("Save file", "Obstacle file name")
        if not save_name:
            save_name = 'latest_scene'
        save_name = save_name.replace('/', '_')
        self.save(save_name)

    def save(self, save_name):
        obstacle_data = {"obstacles": [], "exits": []}
        for number, obstacle in enumerate(self.obstacle_list):
            name = "Obstacle#%d" % number
            obstacle_data["obstacles"].append({"name": name, "begin": obstacle.start, "size": obstacle.size})
        for number, exit_obstacle in enumerate(self.exit_list):
            name = "Exit#%d" % number
            obstacle_data["exits"].append({"name": name, "begin": exit_obstacle.start, "size": exit_obstacle.size})
        with open('../scenes/%s.json' % save_name, 'w') as object_file:
            object_file.write(json.dumps(obstacle_data, indent=4, sort_keys=True))

    def convert_relative_coordinate(self, coord):
        return coord[0] * self.size[0], (1 - coord[1]) * self.size[1]


class Obstacle:
    def __init__(self, begin, size):
        ft.debug("Creating new obstacle with start %s and size %s" % (str(begin), str(size)))
        self.start = begin
        self.size = size
        self.end = (begin[0] + size[0], begin[1] + size[1])

    def __repr__(self):
        return "obstacle#(%d,%d)-(%d,%d)" % (self.start[0], self.start[1], self.end[0], self.end[1])


class Exit(Obstacle):
    def __repr__(self):
        return "Exit#(%d,%d)-(%d,%d)" % (self.start[0], self.start[1], self.end[0], self.end[1])


sc = SceneCreator()
sc.window.mainloop()
