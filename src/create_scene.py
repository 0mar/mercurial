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
        self.obs_start = self.obs_end = None
        self.exit_list = []

    def start_drawing(self, event):
        ft.debug(self.object_status.get())
        self.draw_status = SceneCreator.DrawStatus.drawing
        self.obs_start = (event.x, event.y)
        self.window.bind('<Motion>', self.motion)
        self.window.bind('<Button-1>', self.stop_drawing)
        ft.debug("Started drawing. Start location of obstacle: %s" % str(self.obs_start))

    def stop_drawing(self, event):
        ft.debug("stopped drawing")
        self.window.unbind('<Motion>')
        self.draw_status = SceneCreator.DrawStatus.idle
        self.obs_end = (event.x, event.y)
        ft.debug("End of obstacle: %s" % str(self.obs_end))
        new_obstacle = self.create_new_obstacle()
        self.obs_start = self.obs_end = None
        self.window.bind('<Button-1>', self.start_drawing)
        self.draw_obstacle(new_obstacle)

    def draw_scene(self):
        if self.draw_status == SceneCreator.DrawStatus.drawing:
            self.redraw_new_obstacle(self.obs_start, self.obs_end)

    def draw_obstacle(self, obstacle):
        real_coords = tuple([i * self.size[0] for i in obstacle.start + obstacle.end])
        ft.debug("Drawing obstacle on %s" % str(real_coords))
        if SceneCreator.ObjectStatus(self.object_status.get()) == SceneCreator.ObjectStatus.obstacle:
            color = 'gray'
        else:
            color = 'red'
        self.canvas.create_rectangle(real_coords, fill=color)

    def redraw_new_obstacle(self, start, end):
        if self.new_object_holder:
            self.canvas.delete(self.new_object_holder)
        self.new_object_holder = self.canvas.create_rectangle(start, end, fill=None)

    def motion(self, event):
        assert self.draw_status == SceneCreator.DrawStatus.drawing
        cur_pos = (event.x, event.y)
        self.redraw_new_obstacle(self.obs_start, cur_pos)

    def create_new_obstacle(self):
        begin_x = min(self.obs_start[0], self.obs_end[0]) / self.size[0]
        begin_y = min(self.obs_start[1], self.obs_end[1]) / self.size[0]
        end_x = max(self.obs_start[0], self.obs_end[0]) / self.size[0]
        end_y = max(self.obs_start[1], self.obs_end[1]) / self.size[0]
        if SceneCreator.ObjectStatus(self.object_status.get()) == SceneCreator.ObjectStatus.obstacle:
            new_obstacle = Obstacle((begin_x, begin_y), (end_x, end_y))
            self.obstacle_list.append(new_obstacle)
            return new_obstacle
        else:
            new_exit = Exit((begin_x, begin_y), (end_x, end_y))
            self.exit_list.append(new_exit)
            return new_exit

    def ask_save(self):
        save_name = tkinter.simpledialog.askstring("Save file", "Obstacle file name")
        self.save(save_name)

    def save(self, save_name):
        obstacle_data = {"obstacles": [], "exits": []}
        for number, obstacle in enumerate(self.obstacle_list):
            name = "Obstacle#%d" % number
            obstacle_data["obstacles"].append({"name": name, "begin": obstacle.start, "size": obstacle.size})

        for number, obstacle in enumerate(self.exit_list):
            name = "Exit#%d" % number
            obstacle_data["exits"].append({"name": name, "begin": obstacle.start, "size": obstacle.size})
        with open('../scenes/%s.json' % save_name, 'w') as object_file:
            object_file.write(json.dumps(obstacle_data, indent=4, sort_keys=True))


class Obstacle:
    def __init__(self, start, end):
        ft.debug("Creating new obstacle with start %s and end %s" % (str(start), str(end)))
        self.start = start
        self.end = end
        self.size = (end[0] - start[0], end[1] - start[1])

    def __repr__(self):
        return "obstacle#(%d,%d)-(%d,%d)" % (self.start[0], self.start[1], self.end[0], self.end[1])


class Exit(Obstacle):
    def __repr__(self):
        return "Exit#(%d,%d)-(%d,%d)" % (self.start[0], self.start[1], self.end[0], self.end[1])


sc = SceneCreator()
sc.window.mainloop()
