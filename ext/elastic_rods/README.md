# elastic_rods
A simulation framework for discrete elastic rods and X-Shells.

# Getting Started
## C++ Code Dependencies

The C++ code relies on `boost` and `cholmod/umfpack`. Some parts of the code
(actuator sparsification, design optimization) also depend on the commercial
optimization package [`knitro`](https://www.artelys.com/solvers/knitro/); these
will be omitted from the build if `knitro` is not found.

### macOS
You can install all the necessary dependencies on macOS with [MacPorts](https://www.macports.org):

```bash
# Build/version control tools, C++ code dependencies
sudo port install cmake boost suitesparse ninja
# Dependencies for jupyterlab/notebooks
sudo port install py37-pip
sudo port select --set python python37
sudo port select --set pip3 pip37
sudo port install npm6
```

### Ubuntu 19.04
A few more packages need to be installed on a fresh Ubuntu 19.04 install:
```bash
# Build/version control tools
sudo apt install git cmake ninja-build
# Dependencies for C++ code
sudo apt install libboost-filesystem-dev libboost-system-dev libboost-program-options-dev libsuitesparse-dev
# LibIGL/GLFW dependencies
sudo apt install libgl1-mesa-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev
# Dependencies (pybind11, jupyterlab/notebooks)
sudo apt install python3-pip npm
# Ubuntu 19.04 packages an older version of npm that is incompatible with its nodejs version...
sudo npm install npm@latest -g
```

## Obtaining and Building

Clone this repository *recursively* so that its submodules are also downloaded:

```bash
git clone --recursive https://github.com/EPFL-LGG/elastic_rods
```

Build the C++ code and its Python bindings using `cmake` and your favorite
build system. For example, with [`ninja`](https://ninja-build.org):

```bash
cd elastic_rods
mkdir build && cd build
cmake .. -GNinja
ninja
```


## Running the Jupyter Notebooks
The preferred way to interact with the inflatables code is in a Jupyter notebook,
using the Python bindings.
We recommend that you install the Python dependencies and JupyterLab itself in a
virtual environment (e.g., with [venv](https://docs.python.org/3/library/venv.html)).

```bash
pip3 install wheel # Needed if installing in a virtual environment
pip3 install jupyterlab ipykernel==5.5.5 # Use a slightly older version of ipykernel to avoid cluttering notebook with stdout content.
# If necessary, follow the instructions in the warnings to add the Python user
# bin directory (containing the 'jupyter' binary) to your PATH...

git clone https://github.com/jpanetta/pythreejs
cd pythreejs
pip3 install -e .
cd js
jupyter labextension install .

pip3 install matplotlib scipy
```

Launch Jupyter lab from the root python directory:
```bash
cd python
jupyter lab
```

Now try opening and running an example notebook, e.g.,
`Demos/MantaRayDemo.ipynb`.
