__author__ = 'omar'

import argparse

import numpy as np

from geometry import Size, Point, LineSegment
from src import scene
from visualization import VisualScene
from grid_computer import GridComputer
from static_planner import GraphPlanner



# Default parameters
number_of_pedestrians = 1
domain_width = 70
domain_height = 70
obstacle_file = 'path_planning_obstacle_list.json'

# Command line parameters
parser = argparse.ArgumentParser(description="Prototype Crowd Dynamics Simulation")
parser.add_argument('-n', '--number', type=int, help='Number of pedestrians in simulation',
                    default=number_of_pedestrians)
parser.add_argument('-s', '--step', action='store_true', help='Let simulation progress on mouse click only')
parser.add_argument('-p', '--plot', action='store_true', help='Let simulation plot global values on each time step')
parser.add_argument('-a', '--apply', action='store_true', help='Let simulation apply UIC to the pedestrians')
parser.add_argument('-x', '--width', type=int, help='Width of the simulation domain', default=domain_width)
parser.add_argument('-y', '--height', type=int, help='Height of the simulation domain', default=domain_height)
parser.add_argument('-d', '--delay', type=int, help='Delay between time steps (in milliseconds)', default=1)
parser.add_argument('-o', '--obstacle-file', type=str, help='JSON file containing obstacle descriptions',
                    default=obstacle_file)
args = parser.parse_args()

option = 'Pedestrian_path'
if option=='Pedestrian_path':
    # Draw pedestrian path
    scene_obj = scene.Scene(size=Size([args.width, args.height]), obstacle_file=args.obstacle_file,
                        pedestrian_number=args.number)
    for obs in scene_obj.obstacle_list:
        obs.margin_list = [Point(np.sign([x - 0.5, y - 0.5])) * 1 for x in range(2) for y in range(2)]
    ped = scene_obj.pedestrian_list[0]
    ped.manual_move(Point([60,10]))
    planner = GraphPlanner(scene_obj)
    grid = GridComputer(scene_obj, show_plot=args.plot, apply_interpolation=True,apply_pressure=args.apply)

    # Methods inserted on every update
    def step():
        planner.collective_update()
        grid.step()

    vis = VisualScene(scene_obj, 500, 300, step=step, loop=False, delay=args.delay)

    def vis_with_line(self):
        self.canvas.delete('all')
        for obstacle in self.scene.obstacle_list:
            self.draw_obstacle(obstacle)
        self.draw_pedestrians()
        self.draw_line_segment(LineSegment([ped.position,ped.line.end]))
        self.draw_path(ped.path)
        self.store_scene()
    # Running
    vis.draw_scene = lambda:vis_with_line(vis)
    vis.loop()
    vis.window.mainloop()
elif option == 'Draw_graph':
    import matplotlib.pyplot as plt
    scene_obj = scene.Scene(size=Size([70, 70]), obstacle_file=args.obstacle_file,
                    pedestrian_number=1)
    ped = scene_obj.pedestrian_list[0]
    ped.manual_move(Point([60,10]))
    planner = GraphPlanner(scene_obj)
    planner.draw_graph(planner.graph,ped)
    plt.axis([-5,scene_obj.size.width+5,-5,scene_obj.size.height+5])
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Obstacle graph')
    plt.savefig('images/obstacle_graph.pdf')
    plt.show()
