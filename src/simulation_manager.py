__author__ = 'omar'
import json
import configparser

import numpy as np
import scipy.io as sio

import scene as scene_module
import functions
from visualization import VisualScene, NoVisualScene
from dynamic_planner import DynamicPlanner
from grid_computer import GridComputer
from results import Result
from static_planner import GraphPlanner
from scene_cases import ImpulseScene, TwoImpulseScene, TopScene
from skeleton_planner import SkeletonPlanner


class SimulationManager:
    def __init__(self, args):
        functions.VERBOSE = args.verbose
        self.scene = None
        self.step_functions = []
        self.on_pedestrian_exit_functions = []
        self.finish_functions = []
        config = configparser.ConfigParser()
        has_config = config.read(args.config_file)
        if not has_config:
            raise FileNotFoundError("Configuration file %s not found" % args.config_file)
        functions.EPS = config['general'].getfloat('epsilon')
        self.config = config

        if args.obstacle_file:
            config['general']['obstacle_file'] = args.obstacle_file

        if args.configuration == 'uniform':
            self.scene = scene_module.Scene(
                initial_pedestrian_number=args.number, config=config)
        elif args.configuration == 'top':
            self.scene = TopScene(barrier=0.8,
                                  initial_pedestrian_number=args.number, config=config)
        elif args.configuration == 'center':
            self.scene = ImpulseScene(initial_pedestrian_number=args.number, impulse_location=(0.5, 0.6),
                                      impulse_size=8, config=config)
        elif args.configuration == 'bottom':
            self.scene = TwoImpulseScene(
                initial_pedestrian_number=args.number, impulse_locations=[(0.5, 0.4), (0.4, 0.2)], impulse_size=8,
                config=config)
        if not self.scene:
            raise ValueError("No scene has been initialized")
        # Initialization planner
        self.step_functions.append(self.scene.step)
        if args.dynamic:
            planner = DynamicPlanner(self.scene, config=config, show_plot=args.graph)
            self.step_functions += [planner.step]
        elif False:  # Todo: Fix with argparser
            planner = GraphPlanner(self.scene, config)
            grid = GridComputer(self.scene, show_plot=args.graph, apply_interpolation=args.apply_interpolation,
                                apply_pressure=args.apply_pressure, config=config)
            self.step_functions += [planner.step, grid.step]
        else:
            planner = SkeletonPlanner(self.scene, config)
            self.step_functions += [planner.step]

        if args.store_positions:
            # filename = input('Specify storage file\n')
            import re
            filename = re.search('/([^/]+)\.json', config['general']['obstacle_file']).group(1)
            functions.log("Storing positions results in '%s%s'" % (config['general']['result_dir'], filename))
            self.step_functions.append(lambda: self.store_positions_to_file(filename))
            self.finish_functions.append(lambda: self.store_position_usage(filename))
            self.finish_functions.append(self.store_exit_logs)

        if args.results:
            results = Result(self.scene, config)
            self.step_functions.append(results.on_step)
            self.on_pedestrian_exit_functions.append(results.on_pedestrian_exit)
            self.finish_functions.append(results.on_finish)
        if not args.kernel:
            self.vis = VisualScene(self.scene, config)
            if args.step:
                self.vis.disable_loop()
            else:
                self.step_functions.append(self.vis.loop)
        else:
            if not args.store_positions or args.results:
                functions.warn("No results are logged. Ensure you want a headless simulation.")
            self.vis = NoVisualScene(self.scene)
        self.vis.step_callback = self.step
        self.vis.finish_callback = self.finish

    def start(self):
        self.vis.start()
        self.finish()

    def step(self):
        [step() for step in self.step_functions]

    def finish(self):
        functions.log("Finishing simulation")
        [finish() for finish in self.finish_functions]

    def on_pedestrian_exit(self, pedestrian):
        [ped_exit(pedestrian) for ped_exit in self.on_pedestrian_exit_functions]

    def store_config(self, file_name):
        with open(file_name, 'w') as config_file:
            self.config.write(config_file)

    def store_exit_logs(self, file_name=None):
        log_dir = self.config['general']['result_dir']
        if not file_name:
            file_name = self.config['general']['log_file']
        log_dict = {exit_object.name: np.array(exit_object.log_list) for exit_object in self.scene.exit_list}
        sio.savemat(file_name=log_dir + file_name, mdict=log_dict)

    def store_positions_to_file(self, file_name):
        log_dir = self.config['general']['result_dir']
        if self.scene.counter % 10 == 0:
            with open("%s%s-%d" % (log_dir, file_name, int(self.scene.counter / 10)), 'wb') as f:
                np.save(f, self.scene.position_array[self.scene.active_entries.astype(bool)])

    def store_position_usage(self, file_name):
        log_dir = self.config['general']['result_dir']
        store_data = {"number": self.scene.counter, 'name': file_name,
                      'obstacle_file': self.config['general']['obstacle_file'], 'size': self.scene.size.array.tolist()}
        with open('%s%s.json' % (log_dir, file_name), 'w') as f:
            f.write(json.dumps(store_data))

    @staticmethod
    def get_default_config():
        config_file_name = 'config.ini'
        config = configparser.ConfigParser()
        config.read(config_file_name)
        return config
