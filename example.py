from src.mercurial import Simulation

sim = Simulation('scenes/test.png')
sim.add_pedestrians(100,'knowing')
sim.add_pedestrians(500,'following')
sim.add_global('repulsion')
sim.add_local('separation')
sim.visual_backend = 'tkinter'
sim.start()