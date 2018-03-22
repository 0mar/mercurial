from src.mercurial import Simulation
from math import pi
from src.params import Parameters

simulation = Simulation('scenes/office.png')
params = Parameters()
params.dt = 0.3
params.scene_size_x, params.scene_size_y = 100, 200
params.pressure_dx, params.pressure_dy = 1, 1
params.packing_factor = 0.4
params.min_density = 0.2
params.max_density = 0.5
simulation.set_params(params)
simulation.add_pedestrians(3, 'following')
simulation.add_pedestrians(300, 'knowing')
simulation.add_global('repulsion')
simulation.add_local('separation')
cam_positions = [[65, 150], [55, 150]]
cam_angles = [0.5 * pi, -2 / 5 * pi]
simulation.add_cameras(cam_positions, cam_angles)
simulation.add_fire([60, 80], 5)
simulation.visual_backend = True
simulation.store_positions = True
simulation.start()
