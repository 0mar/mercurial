__author__ = 'omar'

import sys

sys.path.insert(1, 'src')
from mpi4py import MPI
from geometry import Size
from scene_cases import TopScene
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
obstacle_file = 'scenes/narrowing_scene.json'
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
    sys.stdout.write("Core %d starts on sim %d" % (rank, i))
    sys.stdout.flush()
    # Default parameters
    number_of_pedestrians = 500 * (i + 1)



    # Initialization
    functions.VERBOSE = False
    scene = TopScene(size=Size([domain_width, domain_height]), obstacle_file=obstacle_file,
                     initial_pedestrian_number=number_of_pedestrians, barrier=0.8)
    planner = GraphPlanner(scene)
    grid = GridComputer(scene, show_plot=False, apply_interpolation=False,
                        apply_pressure=False)
    step_functions = [planner.collective_update, grid.step]
    result = Result(scene)

    vis = VisualScene(scene, 300, 200, step_functions=step_functions, loop=True, delay=1)

    # Running
    vis.loop()
    vis.window.mainloop()
    scene.finish()
    result.write_density_results(result_file_name)
