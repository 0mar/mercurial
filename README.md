# README #

Multiscale crowd dynamics simulation *Mercurial*

## Visualisation notes:

This branch has two additional simulation files called `office.py` and `cave.py`. Running these files creates data sets that can be used in visualisation tools. For more information/bugs regarding the simulation/datasets, contact me.


### What is this repository for? ###

* Quick summary

This repository contains a framework to simulate the motion of human crowds, managed by Omar Richardson.
The simulation is written in Python 3 and calls some custom Fortran libraries.

* Version

Current version: 0.3.

This simulation is under active development.
Information on the simulation can be found [here](https://symbols.hotell.kau.se/2016/11/30/mercurial/), a summary of the mathematical implementations can be found [here](https://symbols.hotell.kau.se/2016/11/20/graduation-project/), and more information of the structure of the code is present [here](https://symbols.hotell.kau.se/2018/02/05/mercurial-2/).
Details and a detailed analysis and features is present in the report `thesis.pdf`.

## Installation ##

Obtain the source code and create the custom modules with

```bash
git clone https://github.com/0mar/mercurial.git
cd mercurial
python3 setup.py install
```

## Usage: creating a simulation ##

After installation, the simulation can be imported using module `mercurial`.

Starting a simulation can be as simple as this

```python
from src.mercurial import Simulation

simulation = Simulation('scenes/test.png')
simulation.add_pedestrians(100)
simulation.start()
```

Some example files (called `example*.py`) are included that feature some more options.
You can run them with: `python3 example.py`.

## Environment ##

New environments can be created using Windows' Paint, Linux's Pinta, Mac's preview, or whatever (simple) image editing program you have. Store the images in PNG format and load them in the simulation. Mercurial interprets them as follows:

* White: normal accessible space
* Black: inaccessible zones (obstacles)
* Grey: less accessible zones
* Green: exits

For example:
![Example image](/scenes/cave.png?raw=true "Example of simulation environment")

Caveats:
* If you create your own scenarios in an image editor, remember to turn antialiasing off. Mercurial can have difficulties with interpreting smoothed obstacle edges in images.
* Creating large images (larger than 1024x1024 for instance) delays pre-processing time, and is (probably) not required for the simulation. If you want large environments, you can set the `scene_size_x/scene_size_y` parameters.
The resolution of the images only affects the level of detail in route planning and the obstacles.

## Custom parameters ##

All parameters can be set in the main file by creating a parameter object and feeding it to the simulation object. For instance, add the following lines before `simulation.start()` to change the time step of the simulation:

```python
from src.params import Parameters

params = Parameters()
params.dt = 0.4
simulation.set_params(params)
```

You can change default parameters in `src/params.py`.

## Simulation structure ##

The simulation source files are located in `src`. Simulation results are stored in `results`.
Postprocessing tool `process_results.py` is currently under refactoring.

For more information on the structure of the source code, check [this](https://symbols.hotell.kau.se/2018/02/05/mercurial-2/) blog post.

## Dependencies ##

To build the simulation, a Fortran compiler is needed (only on install). In addition, the code depends on the following Python libraries:

- `numpy`
- `scipy`
- `matplotlib`
- `PIL/Pillow`

Each of these libraries can be installed using Python 3's package installer `pip3`.
However, it might be more convenient to use OS-specific repository versions of above packages, or bundled Python version (Conda/Enthought).

Note that in case of using `pip3`, basic requirements on some older Linux distributions may include `liblapack-dev`, `libblas-dev` (for `numpy` and `scipy`) and `libfreetype6-dev`, `libpng-dev` (for `matplotlib`).

## How to run tests ##

Unit tests are stored in folder `tests/` and are set up to run using the unittest framework `nosetests`.
The coverage of the tests highly varies per module and most have not been maintained. Feel free to contribute to unittests of any module.

### Known issues ###

When running Mercurial on (my) Mac, exiting the simulation sometimes causes it to hang unless the process is aborted manually.
On several (again, of my) Linux distributions, this is not the case.
I suspect there to be a mismatch in how Conda interacts with the tkinter module on macOS.

Other than that, this simulation has points for improvement. The biggest issue is the lack of tests and maintaining thereof.

### Contribution guidelines ###

Any contribution, be it better performing modules, more tests, code review, or new features, are welcome.

### Who do I talk to? ###

The owner of this repo is 0mar, PhD-student at Karlstad University Sweden,
 reachable on omar.richardson@kau.se.
