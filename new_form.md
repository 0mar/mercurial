# New form for Mercurial

Idea: A simulation framework that allows for the simple scripting of crowd simulations.

 - Standard and default behaviour through running the example code
 - Easy extending by overriding classes/adding own Python magic


Examples: FeNiCs,...

How? Simulation is built using calls to our API.

Requirements on simulation files: They must be a superset of config files. A simulation file must completely determine the run.

- Add and modify all config functions
- Set a random seed generator
- Add events in the simulation
- No command line arguments necessary

```python
sim = Simulation('env.png')
people = Population(100,'following')
sim.add(people)
sim.set_global('repulsion')
sim.set_local('separation')
sim.run()
sim.visual = 'off'
```

Looks better, nice separation between interface and implementation.
Nice extensions, like:
 - Store results of populations in `people`, process like `Results(sim)` or `Results(people)`
 - `sim.max_time=500`, `sim.dt=0.01`
 - `sim.add_on_step(func,step=0)` which adds a function for each step, or a specific `step`.
 - how to enable position storing, results running?
