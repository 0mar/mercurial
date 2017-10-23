import configparser
import json

import numpy as np
import scipy.io as sio

from math_objects import functions
from objects import scene as scene_module
from objects.initial_distributions import ImpulseScene, TwoImpulseScene, TopScene
from planner import Planner
from processing.results import Result
from visualization.simple import VisualScene
from visualization.none import NoVisualScene


class SimulationManager:
    """
    The simulation manager controls all the components of the simulation.
    It converts the input flags to configuration options and augments them to the configuration file.
    It initializes the important objects and makes sure that all the event methods (step,on_exit, on_finish) are run.
    """
    def __init__(self, args):
        """
        Create a new simulation manager. Object of which only one per simulation should exist.
        :param args: argument object containing the command line flags.
        """
        np.seterr(all='raise')
        functions.VERBOSE = args.verbose
        self.scene = None
        self.step_functions = []
        self.on_pedestrian_exit_functions = []
        self.on_pedestrian_init_functions = []
        self.finish_functions = []
        config = configparser.ConfigParser()
        has_config = config.read(args.config_file)
        if not has_config:
            raise FileNotFoundError("Configuration file %s not found" % args.config_file)
        functions.EPS = config['general'].getfloat('epsilon')
        self.config = config

        if args.obstacle_file:
            config['general']['obstacle_file'] = args.obstacle_file

        # Initialization scene
        if args.number >= 0:
            config['general']['number_of_pedestrians'] = str(args.number)
        if args.aware >= 0:
            config['aware']['percentage'] = str("%.2f" % (args.aware / 100))
            print("Performing simulation with %d%% familiar pedestrians" % args.aware)
        if args.configuration == 'uniform':
            self.scene = scene_module.Scene(config=config)
        elif args.configuration == 'top':
            self.scene = TopScene(barrier=0.8, config=config)
        elif args.configuration == 'center':
            self.scene = ImpulseScene(impulse_location=(0.5, 0.6), impulse_size=8, config=config)
        elif args.configuration == 'bottom':
            self.scene = TwoImpulseScene(impulse_locations=[(0.5, 0.4), (0.4, 0.2)], impulse_size=8, config=config)
        if not self.scene:
            raise ValueError("No scene has been initialized")

        # Initialization planner
        planner = Planner(self.scene)
        self.step_functions.append(planner.step)

        if args.store_positions:
            # filename = input('Specify storage file\n')
            import re
            filename = re.search('/([^/]+)\.json', config['general']['obstacle_file']).group(1)
            functions.log("Storing positions results in '%s%s'" % (config['general']['result_dir'], filename))
            self.step_functions.append(lambda: self.store_positions_to_file(filename))
            self.finish_functions.append(lambda: self.store_position_usage(filename))
            self.finish_functions.append(self.store_exit_logs)

        if args.results:
            results = Result(self.scene, planner, str(args.aware))  # Parameter, subject to change.
            self.step_functions.append(results.on_step)
            self.on_pedestrian_exit_functions.append(results.on_pedestrian_exit)
            self.on_pedestrian_init_functions.append(results.on_pedestrian_entrance)
            self.finish_functions.append(results.on_finish)
        if not args.kernel:
            self.vis = VisualScene(self.scene)
            if args.step:
                self.vis.disable_loop()
            else:
                self.step_functions.append(self.vis.loop)
        else:
            if not (args.store_positions or args.results):
                functions.warn("No results are logged. Ensure you want a headless simulation.")
            self.vis = NoVisualScene(self.scene)
        self.scene.on_pedestrian_exit_functions += self.on_pedestrian_exit_functions
        self.scene.on_pedestrian_init_functions += self.on_pedestrian_init_functions
        self.vis.step_callback = self.step
        self.vis.finish_callback = self.finish

    def start(self):
        """
        Start the simulation. When the simulation is finished, self.vis.start() returns, and cleanup is handled
        :return: None
        """
        self.vis.start()
        self.finish()

    def step(self):
        """
        Run all the event listener methods that run on each time step
        :return:
        """
        [step() for step in self.step_functions]

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
            self.config.write(config_file)

    def store_exit_logs(self, file_name=None):
        """
        Stores the exit logs of the pedestrians.
        These can be reused as entry data for a different simulation (although it requires manual tweaking ATM).
        :param file_name: File name for exit logs. If left empty, stores in file indicated by configuration.
        :return:
        """
        log_dir = self.config['general']['result_dir']
        if not file_name:
            file_name = self.config['general']['log_file']
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
        # This way we clean up the files and we are able to extract a 4 dimension density field from one file.
        log_dir = self.config['general']['result_dir']
        if self.scene.counter % 10 == 0:
            with open("%s%s-%d" % (log_dir, file_name, int(self.scene.counter / 10)), 'wb') as f:
                np.save(f, self.scene.position_array[self.scene.active_entries])

    def store_position_usage(self, file_name):
        """
        Store metadata of the simulation to file, required to combine with the exit logs if you couple the simulation.
        Can be improved if one improves the exit log storage.
        :param file_name: File name for metadata storage.
        :return:
        """
        log_dir = self.config['general']['result_dir']
        store_data = {"number": self.scene.counter, 'name': file_name,
                      'obstacle_file': self.config['general']['obstacle_file'], 'size': self.scene.size.array.tolist()}
        with open('%s%s.json' % (log_dir, file_name), 'w') as f:
            f.write(json.dumps(store_data))

    @staticmethod
    def get_default_config():
        """
        Open default config file. Convenient for testing purposes.
        :return: Default config.
        """
        config_file_name = 'config.ini'
        config = configparser.ConfigParser()
        config.read(config_file_name)
        return config
