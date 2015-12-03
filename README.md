# README #

This document provides an overview for setting up and running the prototype crowd dynamics simulation Mercurial.

### What is this repository for? ###

* Quick summary

This repository is a private repository for a project on Crowd Dynamics simulation, managed by Omar Richardson.
The simulation is written in Python and relies on some Cython modules to increase computation speed.

* Version

This simulation is under active development. 
Any information on simulation techniques and features can be found in the included LaTeX file.

### How do I get set up? ###

* Summary of set up

The repository can be cloned using the link above. Build the library objects and use python 3.4 or higher to run the application.
All source files are collected in folder `src/`.
Create the shared library objects with `python3 setup.py build_ext --build-lib=src/cython_modules`
The simulation can then be run like so: `python3 main.py`. 
Additional command line arguments can be inspected by appending `-h`.

* Configuration

All free parameters not provided on the command line can be set in a configuration file.
An example file with a set of regular parameters is provided in `config.ini`.

* Dependencies

This project depends on the following external libraries:

- `numpy`
- `scipy`
- `matplotlib`
- `networkx`
- `cvxopt`
- `cython`

Each of these libraries can be installed using the `pip3` command.

* How to run tests

Unit tests are stored in folder `tests/` and are set up to run using the unittest framework `nosetests`. 
The coverage of the tests highly varies per module. Feel free to contribute to unittests of any module.

### Changelog ###

- Increased pedestrian capacity
- Increased computational speed
- Dynamic planner module
- Alternative LCP solver, based on Projected Gauss Seidel iterations
- Multiple exits
- Entrances
- Result processing
- Bug fixes


### Known issues ###

This simulation has quite some points for improvement. Known issues include:

#### Dynamic planner speed directions ####

For large crowds, the speed computation of the dynamic planner becomes unstable and pedestrians are observed congregating in waves. This is especially noticed around entrances with a high pedestrian output
#### Path planning checkpoint congestion ####

Since the indicative path planner computes intermediate locations for the pedestrians to follow until they reach the goal
 and many pedestrians share the same indicative path, congestion and brawls occur when too many pedestrians approach the same obstacle.
  
#### Pressure driving pedestrians up walls ####

 While the simulation deals with stuck pedestrians, sometimes the pressure pushes pedestrians into walls or obstacles.
  Sometimes this means pedestrians are stationary until the density reduces to normal proportions, but in some configurations a 
  dead lock can occur, in which a group of pedestrians is unable to because of a pressure caused by the group itself.
#### Slowness of QP solver for larger grids ####

The default cell size for the grid computer is 20 times 20, quite coarse. However, larger grids have their impact on the 
simulation speed.

### Contribution guidelines ###

Any contribution, be it better performing modules, more tests, code review, or new features, are welcome.

### Who do I talk to? ###

* Repo owner or admin

The owner of this repo is 0mar, student at University of Technology Eindhoven,
 reachable on o.m.richardson@student.tue.nl.
