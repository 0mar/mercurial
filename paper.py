from src.mercurial import Simulation
from src.params import Parameters
# Run simulation on scenario 'test.png'
simulation = Simulation('scenes/fire.png')
# Add 100 pedestrians that know the environment
simulation.add_pedestrians(20, 'knowing')
# Add 100 pedestrians that follow the rest
simulation.add_pedestrians(180, 'following')
# Add a global repulsion of dense zones to the pedestrians
simulation.add_global('repulsion')
# Add a local separation between pedestrians which are too close
simulation.add_local('separation')
params = Parameters()
params.scene_size_x, params.scene_size_y = 100, 100
params.obstacle_clearance = 4
params.random_force = 0.005
params.swarm_force = 100
params.boundary_pressure = 3
simulation.set_params(params)
# Add a fire
simulation.add_fire([25, 50], 2)
# Start the simulation
simulation.start()
