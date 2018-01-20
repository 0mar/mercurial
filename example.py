from src.mercurial import Simulation

sim = Simulation('scenes/test.png')
sim.add_pedestrians(10, 'knowing')
sim.add_pedestrians(100, 'following')
sim.add_global('repulsion')
sim.add_local('separation')
sim.add_fire([30, 40], 5)
sim.visual_backend = 'tkinter'
sim.start()
