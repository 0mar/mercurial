__author__ = 'omar'

import argparse

from geometry import Size
import scene
from visualise_scene import VisualScene
from planner import GraphPlanner


# Default parameters
number_of_pedestrians = 100
domain_width = 250
domain_height = 150
obstacle_file = 'demo_obstacle_list.json'

# Command line parameters
parser = argparse.ArgumentParser(description="Prototype Crowd Dynamics Simulation")
parser.add_argument('-n', '--number', type=int, help='Number of pedestrians in simulation',
                    default=number_of_pedestrians)
parser.add_argument('-s', '--step', action='store_true', help='Let simulation progress on mouse click only')
parser.add_argument('-x', '--width', type=int, help='Width of the simulation domain', default=domain_width)
parser.add_argument('-y', '--height', type=int, help='Height of the simulation domain', default=domain_height)
parser.add_argument('-o', '--obstacle-file', type=str, help='JSON file containing obstacle descriptions', default=obstacle_file)
args = parser.parse_args()

# Initialization
scene = scene.Scene(size=Size([args.width, args.height]), obstacle_file=args.obstacle_file,pedNumber=args.number)
planner = GraphPlanner(scene)
vis = VisualScene(scene, 1500, 1000, step=planner.update, loop=not args.step)

# Running
vis.loop()
vis.window.mainloop()