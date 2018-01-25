from src.mercurial import Simulation

simulation = Simulation('scenes/test.png')
simulation.add_pedestrians(100, 'knowing')
simulation.add_pedestrians(100, 'following')
simulation.add_global('repulsion')
simulation.add_local('separation')
# simulation.add_fire([30, 40], 5)
simulation.visual_backend = True
simulation.store_positions = False
simulation.start()
