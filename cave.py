from src.mercurial import Simulation
from src.params import Parameters
from math import pi

simulation = Simulation('scenes/cave.png')
params = Parameters()
params.pressure_dx = 1
params.pressure_dy = 1
params.packing_factor = 0.6
params.min_density = 0.2
params.max_density = 0.7
params.fire_intensity = 0
params.follow_radius = 50
simulation.add_pedestrians(300, 'knowing')
simulation.add_pedestrians(3, 'following')
simulation.add_global('repulsion')
simulation.add_local('separation')
cam_positions = [[400 / 8, 500 / 6], [600 / 8, 300 / 6], [200 / 8, 250 / 6]]
cam_angles = [0.9 * pi, 0.5 * pi, 0.25 * pi]
simulation.add_cameras(cam_positions, cam_angles)
# simulation.add_fire([30, 40], 5)
simulation.visual_backend = True
simulation.store_positions = True
simulation.inflow = 0.1
simulation.start()
