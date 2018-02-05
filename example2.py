from src.mercurial import Simulation
from src.params import Parameters

# Run simulation on scenario 'test.png'
simulation = Simulation('scenes/test.png')
# Add 500 pedestrians that know the environment
simulation.add_pedestrians(100, 'knowing')
# Add 100 pedestrians that follow the rest
simulation.add_pedestrians(100, 'following')
# Add a global repulsion of dense zones to the pedestrians
simulation.add_global('repulsion')
# Add a fire at location (30,40) with intensity level 5
simulation.add_fire([30, 40], 5)
# Don't include a visual backend
simulation.visual_backend = True  # False
# Store the results for future use
simulation.store_positions = True

params = Parameters()
# Set the size of the scene to 300m x 500m
params.scene_size_x, params.scene_size_y = 300, 500
# Set the time step to 0.5 seconds
params.dt = 0.5
# Set the minimal distance to 3m
params.min_dist = 3
# Add the custom parameters to the simulation
simulation.set_params(params)
# Start the simulation
simulation.start()
