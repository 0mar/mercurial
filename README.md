# README #

Multiscale crowd dynamics simulation *Mercurial*

### What is this repository for? ###

* Quick summary

This repository contains simulation framework on particle interaction and crowd dynamics, managed by Omar Richardson.
The simulation is written in Python 3 and calls some custom Fortran libraries.

* Version

Current version: 3.3.
This simulation is under active development. 
Information on the simulation can be found [here](https://symbols.hotell.kau.se/2016/11/30/mercurial/) and a summary of the mathematical implementations can be found [here](https://symbols.hotell.kau.se/2016/11/20/graduation-project/).
Details and a detailed analysis and features is present in the graduation report.

### How do I get set up? ###

* Summary of setup

Obtain the source code and create the custom modules with

```bash 
git clone https://github.com/0mar/mercurial.git
cd mercurial
python3 setup.py install
```

After that, the simulation is ready to be run with the command

`python3 main.py`. 

For help, append with `-h`.

* Configuration

All free parameters not provided on the command line can be set in a configuration file.
An example file with a set of default parameters is provided in `configs/default.ini`.

* Structure

The simulation source files are located in `src`. Simulation results are stored in `results` and can be processed by running `process_results.py`.

Example scenes files are stored in `scenes`. Scenes can be created manually or by using the simple tool `create_scene.py`.

Other preset configuration files (corresponding to several test cases in the project) are present in the `configs` folder.

* Dependencies

To build the simulation, a Fortran compiler is needed (only once). In addition, the code depends on the following Python libraries:

- `numpy`
- `scipy`
- `matplotlib`

Each of these libraries can be installed using Python 3's package installer `pip3`.
However, it might be more convenient to use OS-specific repository versions of above packages, or bundled Python version.

Note that in case of using `pip3`, basic requirements may include `liblapack-dev`, `libblas-dev` (for `numpy` and `scipy`) and `libfreetype6-dev`, `libpng-dev` (for `matplotlib`).

* How to run tests

Unit tests are stored in folder `tests/` and are set up to run using the unittest framework `nosetests`. 
The coverage of the tests highly varies per module. Feel free to contribute to unittests of any module.

### Known issues ###

This simulation has points for improvement. The biggest issue is the lack of tests and maintaining thereof.

### Contribution guidelines ###

Any contribution, be it better performing modules, more tests, code review, or new features, are welcome.

### Who do I talk to? ###

* Repo owner or admin

The owner of this repo is 0mar, PhD-student at Karlstad University Sweden,
 reachable on omar.richardson@kau.se.
