from src.mercurial import Simulation

# Run simulation on scenario 'test.png'
simulation = Simulation('scenes/merc1.png')
# Add 100 pedestrians that know the environment
simulation.add_pedestrians(200, 'knowing')
# Add 100 pedestrians that follow the rest
simulation.add_pedestrians(50, 'following')
# Add a global repulsion of dense zones to the pedestrians
simulation.add_global('repulsion')
# Add a local separation between pedestrians which are too close
simulation.add_local('separation')

# Start the simulation
simulation.start()
