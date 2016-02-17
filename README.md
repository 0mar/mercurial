# README #

This document provides an overview for setting up and running the prototype crowd dynamics simulation Mercurial.

### What is this repository for? ###

* Quick summary

This repository is a private repository for a project on Crowd Dynamics simulation, managed by Omar Richardson.
The simulation is written in Python and relies on some Cython and FORTRAN90 modules to increase computation speed.

* Version

This simulation is under active development. 
Any information on simulation techniques and features is present in the (currently not disclosed) graduation report.

### How do I get set up? ###

* Summary of setup

The repository can be cloned using the link above, or with 

`git clone https://github.com/0mar/crowd_simulation.git`

Move into the source directory and build the library objects with 

`python3 setup.py install`

After that, the simulation is ready to be run with the command

`python3 main.py`. 

Additional command line arguments can be inspected by appending `-h`.

* Configuration

All free parameters not provided on the command line can be set in a configuration file.
An example file with a set of regular parameters is provided in `configs/default.ini`.

* Structure

The simulation source files are located in `src`. Simulation results are stored in `results` and can be processed by running `process_results.py`.

Example scenes files are stored in `scenes`. Scenes can be created manually or by using the (simple) tool `create_scene.py`, located in the `src` folder.

Other preset configuration files (corresponding to several test cases in the project) are present in the `configs` folder.

Finally, profiling the code is possible with `get_profile.sh` and `view_profile.sh` provided these scripts are run in a UNIX-environment with `gprof2dot`,`dot`,and `profile_eye` are installed.
* Dependencies

This project depends on the following external libraries:

- `numpy`
- `scipy`
- `matplotlib`
- `networkx`
- `cvxopt`
- `cython`

Each of these libraries can be installed using python's package installer `pip3` command.

* How to run tests

Unit tests are stored in folder `tests/` and are set up to run using the unittest framework `nosetests`. 
The coverage of the tests highly varies per module. Feel free to contribute to unittests of any module.

### Changelog ###

- Increased pedestrian capacity
- Increased computational speed
- Introduced fortran90 modules
- Result processing
- Bug fixes


### Known issues ###

This simulation has quite some points for improvement. Known issues include:

#### Path planning checkpoint congestion ####

Since the indicative path planner computes intermediate locations for the pedestrians to follow until they reach the goal
 and many pedestrians share the same indicative path, congestion and brawls occur when too many pedestrians approach the same obstacle.

#### Slowness of QP solver for larger grids ####

The default cell size for the grid computer is 50 times 50, quite coarse. A sparse matrix implementation should solve this.

#### Memory allocation error/Segfault ###

When using the minimal distance enforcement, for large and dense crowds sometimes the FORTRAN module crashes. This is being looked at. For now, if this poses a problem, the minimal distance enforcement can be disabled in the configuration file.

### Contribution guidelines ###

Any contribution, be it better performing modules, more tests, code review, or new features, are welcome.

### Who do I talk to? ###

* Repo owner or admin

The owner of this repo is 0mar, student at University of Technology Eindhoven,
 reachable on o.m.richardson@student.tue.nl.
