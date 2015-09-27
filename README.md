# README #

README for the hybrid Crowd Dynamics simulation

### What is this repository for? ###

* Quick summary

This repository is a private repository for a project on Crowd Dynamics simulation, managed by Omar Richardson.

* Version

This simulation is under active development. 
Any information on simulation techniques and features can be found in the included LaTeX file.

### How do I get set up? ###

* Summary of set up

The repository can be cloned using the link above. Use python 3.2 or higher to run the application.
All source files are collected in folder `src/`.
The simulation can be run like so: `python3 main.py`. 
Additional command line arguments can be inspected by appending `-h`.

* Configuration

All free parameters not provided on the command line can be provided in a JSON file. 
An example file with a set of regular parameters is provided in `params.json` .

* Dependencies

This project depends on the following external libraries:

- `numpy`
- `scipy`
- `matplotlib`
- `networkx`
- `cvxopt`

Each of these libraries can be installed using the `pip3` command.

* How to run tests

Unit tests are stored in folder `tests/` and are set up to run using the unittest framework `nosetests`. 
The coverage of the tests highly varies per module. Feel free to contribute to unittests of any module.

### Contribution guidelines ###

Any contribution, be it better performing modules, more tests, code review, or new features, are welcome.

### Who do I talk to? ###

* Repo owner or admin

The owner of this repo is Omar Richardson, student at University of Technology Eindhoven,
 reachable on o.m.richardson@student.tue.nl
