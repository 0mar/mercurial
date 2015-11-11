__author__ = 'omar'

import sys

sys.path.insert(1, 'src')
import argparse

from simulation_manager import SimulationManager

# Default parameters
number_of_pedestrians = 100
# Command line parameters
parser = argparse.ArgumentParser(description="Prototype Crowd Dynamics Simulation")
parser.add_argument('-n', '--number', type=int, help='Number of pedestrians in simulation',
                    default=number_of_pedestrians)
parser.add_argument('-s', '--step', action='store_true', help='Let simulation progress on mouse click only')
parser.add_argument('-g', '--graph', action='store_true', help='Let simulation graph grid values on each time step')
parser.add_argument('-i', '--apply-interpolation', action='store_true',
                    help='Let simulation impose swarm behaviour to pedestrians')
parser.add_argument('-p', '--apply-pressure', action='store_true',
                    help='Let simulation impose UIC (pressure term) to the pedestrians (-c implied)')
parser.add_argument('-t', '--time-delay', type=int, help='Delay between time steps (in milliseconds)', default=1)
parser.add_argument('-o', '--obstacle-file', type=str, help='JSON file containing obstacle descriptions',
                    default='')
parser.add_argument('-c', '--configuration', type=str, choices=['uniform', 'top', 'center', 'bottom'],
                    default='uniform',
                    help='Specify configuration of pedestrian initialization')
parser.add_argument('--draw-cells', action='store_true', help='Draw the boundaries and cell centers of the cells. '
                                                              'Slows the visualization significantly, only use for debugging')
parser.add_argument('-f', '--config-file', type=str, default='config.ini', help='Specify configuration file to be used')
parser.add_argument('-r', '--results', action='store_true', help='Log results of simulation to disk')
parser.add_argument('--store-positions', action='store_true', help='Store positions at some time steps')

parser.add_argument('-v', '--verbose', action='store_true', help='Print debugging information to console')
parser.add_argument('-d', '--dynamic', action='store_true', help='Apply dynamic planner to pedestrians')
parser.add_argument('--log-exits', action='store_true', help='Store exit data so simulation results can be reused')
parser.add_argument('-k', '--kernel', action='store_true',
                    help='Don\'t run visualization')

args = parser.parse_args()

manager = SimulationManager(args)
manager.start()
