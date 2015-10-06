__author__ = 'omar'

import sys

sys.path.insert(1, 'src')
from mpi4py import MPI
from geometry import Size
import scene as scene_module
import functions
from results import Result
from visualization import VisualScene
from grid_computer import GridComputer
from static_planner import GraphPlanner

sim_num = 8
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

domain_width = 70
domain_height = 70
obstacle_file = 'scenes/demo_obstacle_list.json'
result_file_name = 'results/density_dependent.txt'


def get_sim_partition():
    l = []
    sim_index = rank
    while sim_index < sim_num:
        l.append(sim_index)
        sim_index += size
    return l


sim_list = get_sim_partition()
if rank == 0:
    # Reset the results file
    with open(result_file_name, 'w') as f:
        f.write('')

for i in sim_list:
    print("Core %d starts on sim %d" % (rank, i))
    # Default parameters
    number_of_pedestrians = 100 * (i + 1)



    # Initialization
    functions.VERBOSE = False
    scene = scene_module.Scene(size=Size([domain_width, domain_height]), obstacle_file=obstacle_file,
                               pedestrian_number=number_of_pedestrians)
    planner = GraphPlanner(scene)
    grid = GridComputer(scene, show_plot=False, apply_interpolation=False,
                        apply_pressure=False)
    step_functions = [planner.collective_update, grid.step]
    result = Result(scene)

    vis = VisualScene(scene, 1500, 1000, step_functions=step_functions, loop=True, delay=1)

    # Running
    vis.loop()
    vis.window.mainloop()
    scene.finish()
    result.write_density_results(result_file_name)
