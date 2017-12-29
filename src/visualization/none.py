import time


from math_objects import functions as ft

from visualization.simple import VisualScene


class NoVisualScene(VisualScene):
    def __init__(self, scene):
        self.scene = scene
        self.time = time.time()
        self.is_done = False

    def start(self):
        while not self.is_done:
            try:
                self.step_callback()
                # ft.log("Iteration took %.4f seconds" % (time.time() - self.time))
                # ft.log("Time step %d" % self.scene.counter)
                self.time = time.time()
            except KeyboardInterrupt:
                ft.log("\nUser interrupted simulation")
                self.scene.status = 'DONE'

    def finish(self):
        self.is_done = True