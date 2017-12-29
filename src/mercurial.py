import sys
sys.path.append('src')
import params

import numpy as np
import scipy.io as sio

from math_objects import functions
from objects.scene import Scene
from micro.separate import Separate
from macro.separate import Repel
from processing.results import Result
from visualization.simple import VisualScene
from visualization.none import NoVisualScene
from populations.base import Population
from behaviours.following import Following
from behaviours.knowing import Knowing


class Simulation:
    """
    The simulation class controls all the components of the simulation.
    It converts the input flags to configuration options and augments them to the configuration file.
    It initializes the important objects and makes sure that all the event methods (step,on_exit, on_finish) are run.
    """

    def __init__(self, scene_file=None):
        """
        Create a new simulation.
        """
        self.status = 'PREPARE'
        self.pre_move_functions = []
        self.post_move_functions = []
        self.step_functions = []
        self.on_pedestrian_exit_functions = []
        self.on_pedestrian_init_functions = []
        self.finish_functions = []

        self.effects = []
        self.populations = []

        if scene_file:
            params.scene_file = scene_file
        else:
            self.scene_file = params.scene_file
        self.store_results = False
        self.visual_backend = 'tkinter'
        functions.EPS = params.tolerance
        self.scene = Scene()

    def _prepare(self):
        if not self.store_positions and not self.store_results:
            functions.warn("No results are logged. Ensure you want a headless simulation.")
        self.scene.on_step_functions.append(self.scene.move)
        self.post_move_functions.append(self.scene.correct_for_geometry)
        self.step_functions = self.pre_move_functions + self.step_functions + self.post_move_functions
        if self.visual_backend.lower() == 'tkinter':
            self.vis = VisualScene(self.scene)
        elif self.visual_backend.lower() == 'none' or not self.visual_backend:
            self.vis = NoVisualScene(self.scene)
        else:
            print("Backend '%s' is not available"%self.visual_backend)
        self.vis.step_callback = self.step
        self.vis.finish_callback = self.finish
        self.scene.on_pedestrian_exit_functions.append(self._check_percentage)
        if params.max_time > 0:
            self.scene.on_pedestrian_exit_functions.append(self._check_max_time)

    def start(self):
        """
        Start the simulation. When the simulation is finished, self.vis.start() returns,
        and cleanup is handled through self.finish()
        :return: None
        """
        self._prepare()
        self.scene.prepare()
        for effect in self.effects:
            effect.prepare()
        for population in self.populations:
            population.prepare()
        self.status = 'RUNNING'
        print("started")
        self.vis.start()
        self.finish()

    def step(self):
        """
        Increase time and
        run all the event listener methods that run on each time step
        :return:
        """
        self.scene.time += params.dt
        self.scene.counter += 1
        print("hoi")
        [step() for step in self.step_functions]
        print("Done")

    def add_local(self,effect):
        if effect.lower() == 'separation':
            separation = Separate(self.scene)
            self.effects.append(separation)
            self.post_move_functions.append(separation.step)
        else:
            raise NotImplementedError("Local effect %s not found"%effect)

    def add_global(self,effect):
        if effect.lower() == 'smoke':
            raise NotImplementedError("Smoke is not yet implemented")
        elif effect.lower() == 'fire':
            raise NotImplementedError("Also fire not yet implemented")
        elif effect.lower() == 'repulsion':
            repulsion = Repel(self.scene)
            self.post_move_functions.append(repulsion.step)

    def add_pedestrians(self,num,behaviour):
        population = Population(self.scene,num)
        if behaviour.lower() == 'following':
            Behaviour = Following
        elif behaviour.lower() == 'knowing':
            Behaviour = Knowing
        else:
            raise NotImplementedError("Behaviour %s not implemented"%behaviour)
        self.populations.append(Behaviour(population))

    def store_positions(self): # Todo: Make this work
        # filename = input('Specify storage file\n')
        import re
        filename = re.search('/([^/]+)\.(png|jpe?g)', params.scene_file).group(1)
        functions.log("Storing positions results in '%s%s'" % (params.result_dir, filename))
        self.step_functions.append(lambda: self.store_positions_to_file(filename))
        # self.finish_functions.append(lambda: self.store_position_usage(filename))
        self.finish_functions.append(self.store_exit_logs)

    def log_results(self,true):
        results = Result(self.scene,None,None)  # Todo: Make this work
        self.step_functions.append(results.on_step)
        self.on_pedestrian_exit_functions.append(results.on_pedestrian_exit)
        self.on_pedestrian_init_functions.append(results.on_pedestrian_entrance)
        self.finish_functions.append(results.on_finish)

    def print_settings(self):
        """
        Prints the parameters used in this simulation.
        :return:
        """
        print([item for item in dir(params) if not item.startswith("__")])

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
        Store the configuration in memory to file(containing the options from the command line and from file)
        :param file_name: File name for new config file
        :return:
        """
        with open(file_name, 'w') as config_file:
            pass
            # params.write(config_file)

    def store_exit_logs(self, file_name=None):
        """
        Stores the exit logs of the pedestrians.
        These can be reused as entry data for a different simulation (although it requires manual tweaking ATM).
        :param file_name: File name for exit logs. If left empty, stores in file indicated by configuration.
        :return:
        """
        log_dir = params.result_dir
        if not file_name:
            file_name = params.log_file
        log_dict = {exit_object.name: np.array(exit_object.log_list) for exit_object in self.scene.exit_list}
        sio.savemat(file_name=log_dir + file_name, mdict=log_dict)

    def store_positions_to_file(self, file_name):
        """
        Store the positions of each pedestrian to a file.
        This allows for the creation of a heatmap and a path display in postprocessing.
        Currently not implemented very efficiently.
        :param file_name: Base file name to store the positions, appended with simulation time.
        :return: None
        """
        # Todo: Alter this, we need the positions stored in one file.
        # This way we clean up the files and we are able to extract a 2+1 dimensional density field from one file.
        log_dir = params.result_dir
        if self.scene.counter % 10 == 0:
            with open("%s%s-%d" % (log_dir, file_name, int(self.scene.counter / 10)), 'wb') as f:
                np.save(f, self.scene.position_array[self.scene.active_entries])

    def _check_max_time(self):
        if self.scene.time > params.max_time:
            self.vis.finish()

    def _check_percentage(self):
        if 1 - np.sum(self.scene.active_entries)/len(self.scene.active_entries) >= params.max_percentage:
            self.vis.finish()