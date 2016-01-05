import warnings
import os
import re

from skimage import io
from skimage.color import rgb2gray
from skimage.morphology import medial_axis

import functions
from visualization import VisualScene

class Skeletonizer:
    file_postfix = '.eps'
    skeleton_folder = 'skeletons'

    @staticmethod
    def get_skeleton(scene, store=True):
        if not os.path.isdir(Skeletonizer.skeleton_folder):
            os.makedirs(Skeletonizer.skeleton_folder)
        obstacle_file = scene.config['general']['obstacle_file']
        skeleton_file = Skeletonizer.skeleton_folder \
                        + re.search('/[^/\.]+', obstacle_file).group(0) + Skeletonizer.file_postfix
        if not os.path.exists(skeleton_file):
            functions.log("No earlier skeleton found, creating from obstacle file")
            dummy_vis = EmptyVisualization(scene, scene.config, filename=skeleton_file)
            dummy_vis.step_callback = dummy_vis.loop
            dummy_vis.start()
            data = io.imread(skeleton_file)
            gray_data = rgb2gray(data)
            medial = 1 - medial_axis(gray_data)
            medial[0, :] = medial[-1, :] = medial[:, 0] = medial[:, -1] = 1  # For some reason we have a black boundary
            if store:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    io.imsave(skeleton_file, medial.astype(float))
        else:
            functions.log("Skeleton found, loading from file")
            data = io.imread(skeleton_file)
            medial = rgb2gray(data)
        return medial


class EmptyVisualization(VisualScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.filename = kwargs['filename']
        self.drawn = 2
        self.delay = 400
        scale = self.scene.config['skeleton'].getfloat('pixel_scale')
        self.size = self.scene.size * scale

    def draw_scene(self):
        if self.drawn:  # Pretty bad, but I blame Tkinter
            self.drawn -= 1
            self.canvas.delete('all')
            for obstacle in self.scene.obstacle_list:
                if not obstacle.accessible:
                    self.draw_obstacle(obstacle)
        else:
            self.store_scene(None, self.filename)
            self.window.destroy()


# output_image = 1-medial.astype(float)
# print(output_image)
# with warnings.catch_warnings():
#     warnings.simplefilter('ignore')
#     io.imsave('output_image.png',output_image)

from scene import Scene
from simulation_manager import SimulationManager

scene = Scene(4, SimulationManager.get_default_config())
Skeletonizer.get_skeleton(scene)
