import sys

sys.path.append('src')
from params import Parameters
import numpy as np
import scipy.io as sio

from math_objects import functions
from objects.scene import Scene
from micro.separate import Separate
from macro.separate import Repel
from processing.log_results import PositionLogger
from visualization.simple import VisualScene
from visualization.none import NoVisualScene
from populations.following import Following
from populations.knowing import Knowing
from extensions.fire import Fire
from extensions.camera import Cameras


class Simulation:
    """
    The simulation class controls all the components of the simulation.
    It converts the input flags to configuration options and augments them to the configuration file.
    It initializes the important objects and makes sure that all the event methods (step, on_exit, on_finish) are run.
    """

    def __init__(self, scene_file=None, params=None):
        """
        Create a new simulation.
        """
        self.on_step_functions = []
        self.on_pedestrian_exit_functions = []
        self.on_pedestrian_init_functions = []
        self.finish_functions = []
        # All the effects added in the simulation. keys are strings of the effect name, values are the effect objects
        self.effects = {}
        self.populations = []
        if not params:
            self.params = Parameters()
        else:
            self.params = params
        if scene_file:
            self.scene_file = scene_file
        self.inflow = False
        self.store_positions = False
        self.logger = None
        self.collect_data = None
        self.visual_backend = True
        functions.EPS = self.params.tolerance # TODO: Remove
        self.scene = Scene()

    def _prepare(self):
        if not self.visual_backend and not self.store_positions:
            functions.warn("No results are logged. Ensure you want a headless simulation.")
        # The order in which the following effects are added is important.
        for population in self.populations:
            self.on_step_functions.append(population.step)
        if 'repulsion' in self.effects:
            self.on_step_functions.append(self.effects['repulsion'].step)
        if 'fire' in self.effects:
            self.on_step_functions.append(self.effects['fire'].step)
        self.on_step_functions.append(self.scene.move)
        if 'separation' in self.effects:
            self.on_step_functions.append(self.effects['separation'].step)
        self.on_step_functions.append(self.scene.correct_for_geometry)
        self.on_step_functions.append(self.scene.find_finished)
        if self.store_positions:
            self.on_step_functions.append(self.logger.step)
        if self.visual_backend:
            self.vis = VisualScene(self.scene)
            self.on_step_functions.append(self.vis.loop)
        else:
            self.vis = NoVisualScene(self.scene)
        if self.inflow:
            self.on_step_functions.append(self._add_new_pedestrian_sometimes)
        self.vis.step_callback = self.step
        self.vis.finish_callback = self.finish
        # Todo: Make sure that all pedestrian exit functions are executed
        self.scene.on_pedestrian_exit_functions.append(self._check_percentage)
        if self.params.max_time > 0:
            self.scene.on_pedestrian_exit_functions.append(self._check_max_time)

    def start(self):
        """
        Start the simulation. When the simulation is finished, self.vis.start() returns,
        and cleanup is handled through self.finish()
        :return: None
        """
        if self.params.scene_file:
            self.scene_file = self.params.scene_file
        if not self.scene_file:
            raise AttributeError("No environment provided")
        self.params.scene_file = self.scene_file
        if self.store_positions:
            self.logger = PositionLogger(self)
        self._prepare()
        self.scene.prepare(self.params)
        for effect in self.effects:
            self.effects[effect].prepare(self.params)
        for population in self.populations:
            population.prepare(self.params)
        if self.store_positions:
            self.logger.prepare(self.params)
        self.vis.prepare(self.params)
        self.vis.start()
        self.finish()

    def step(self):
        """
        Increase time and
        run all the event listener methods that run on each time step
        :return:
        """
        self.scene.time += self.params.dt
        self.scene.counter += 1
        [step() for step in self.on_step_functions]

    def add_local(self, effect):
        effect_name = effect.lower()
        if effect_name == 'separation':
            separation = Separate(self.scene)
            self.effects[effect_name] = separation
        else:
            raise NotImplementedError("Local effect %s not found" % effect)

    def add_global(self, effect):
        effect_name = effect.lower()
        if effect_name == 'repulsion':
            repulsion = Repel(self.scene)
            self.effects[effect_name] = repulsion

    def add_pedestrians(self, num, behaviour='knowing'):
        if type(num) != int or num < 1:
            raise ValueError("Provide a positive integer as a population number, not %s" % num)
        if behaviour.lower() == 'following':
            Population = Following
        elif behaviour.lower() == 'knowing':
            Population = Knowing
        else:
            raise NotImplementedError("Behaviour %s not implemented" % behaviour)
        population = Population(self.scene, num)
        self.populations.append(population)

    def add_fire(self, center, radius):
        effect = Fire(center, radius, self.scene)
        self.params.fire = effect
        self.params.smoke = True
        # If we want more fires, key needs to be unique (changing the fire effect)
        self.effects['fire'] = effect

    def add_cameras(self, positions, angles):
        """
        -- Beta functionality --
        Not implemented yet. Used to pass data about future camera's to post-visualisation.
        
        :param positions:
        :return:
        """
        effect = Cameras(np.array(positions), np.array(angles))
        self.effects['cameras'] = effect

    def allow_new_pedestrians(self, probability):
        """
        -- Beta functionality --
        Add a probability of entering new pedestrians on each time step.

        :param probability: number between 0 and 1
        :return: None
        """
        self.inflow = probability

    def set_visualisation(self, on):
        self.visual_backend = bool(on)

    def set_store_positions(self, on):
        self.store_positions = bool(on)

    def set_data_collector(self, on):
        """
        Not implemented yet
        :param on:
        :return:
        """
        self.collect_data = bool(on)
        #
        # results = Result(self.scene, None, None)  # Todo: Make this work
        # self.on_step_functions.append(results.on_step)
        # self.on_pedestrian_exit_functions.append(results.on_pedestrian_exit)
        # self.on_pedestrian_init_functions.append(results.on_pedestrian_entrance)
        # self.finish_functions.append(results.on_finish)

    def set_params(self, params):
        """
        Set the parameters with the given object.

        :param params: Parameter object
        :return: None
        """
        self.params = params

    def print_settings(self):
        """
        Prints the parameters used in this simulation.
        :return:
        """
        print([item for item in dir(self.params) if not item.startswith("__")])

    def finish(self):
        """
        Finish the simulation and run all registered methods that should be called on finish.
        :return: None
        """
        functions.log("Finishing simulation. %d iterations of %.2f seconds" % (self.scene.counter, self.scene.time))
        [finish() for finish in self.finish_functions]

    def on_pedestrian_exit(self, pedestrian):
        """
        Run all methods required when a pedestrian exits.
        :param pedestrian: Pedestrian that exited the scene.
        :return:
        """
        [ped_exit(pedestrian) for ped_exit in self.on_pedestrian_exit_functions]

    def store_config(self, file_name):
        """
        Store the configuration in memory to file(containing the options from the command line and from file).
        Not yet implemented, I guess pickling the Parameters object would be the best option.
        :param file_name: File name for new config file
        :return:
        """
        with open(file_name, 'w') as config_file:
            pass
            # self.params.write(config_file)

    def _check_max_time(self):
        if self.scene.time > self.params.max_time:
            self.vis.finish()

    def _check_percentage(self, _=None):
        """
        Check whether the required percentage of evacs has been reached
        :param _: Placeholder parameter; ignore
        :return:
        """
        if 1 - np.sum(self.scene.active_entries) / len(self.scene.active_entries) >= self.params.max_percentage:
            self.vis.finish()

    def _add_new_pedestrian_sometimes(self):
        """
        -- Beta functionality --
        Prototype for adding new pedestrians to the scene.
        Not suitable for research purposes yet.
        :return:
        """
        if np.random.random() < self.inflow and len(self.scene.pedestrian_list) < self.scene.total_pedestrians:
            self.populations[0].create_new_pedestrian()
