<!-- PROJECT LOGO -->
<p align="center">

  <h1 align="center"><a href="https://go.epfl.ch/c-shells">C-shells: Deployable Gridshells with Curved Beams</a></h1>

![Teaser](./release/zoo.jpg)

  <p align="center">
    ACM Transactions on Graphics (Proceedings of SIGGRAPH Asia), December 2023.
    <br />
    <a href="https://people.epfl.ch/quentin.becker?lang=en"><strong>Quentin Becker</strong></a>
    ·
    <a href="https://people.epfl.ch/seiichi.suzuki?lang=en"><strong>Seiichi Suzuki</strong></a> 
    ·
    <a href="https://people.epfl.ch/yingying.ren?lang=en"><strong>Yingying Ren</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=JnocFM4AAAAJ&hl=en"><strong>Davide Pellis</strong></a>
    ·
    <a href="http://julianpanetta.com/"><strong>Julian Panetta</strong></a> 
    ·
    <a href="https://people.epfl.ch/mark.pauly?lang=en"><strong>Mark Pauly</strong></a>
    <br />
  </p>

  <p align="center">
    <a href='https://infoscience.epfl.ch/record/305959?ln=en'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat-square' alt='Paper PDF'>
    </a>
    <a href='https://go.epfl.ch/c-shells' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat-square' alt='Project Page'>
    </a>
  </p>
</p>

## About

This repository contains the source code and data for the paper C-shells: Deployable Gridshells with Curved Beams, published at SIGGRAPH Asia 2023. 

A C-shell is composed of curved flexible beams connected at rotational joints that can be assembled in a stress-free planar configuration. When actuated, the elastic beams deform and the assembly deploys towards the target 3D shape.

## Getting started

### Setup for elastic_rods

Before cloning the repository, make sure you have installed all the dependencies required by [elastic rods](https://github.com/EPFL-LGG/Cshells/tree/main/ext/elastic_rods#c-code-dependencies). The design optimization and planarization parts of the code also depend on the commercial optimization package [`knitro`](https://www.artelys.com/solvers/knitro/); these will be omitted from the build if `knitro` is not found.

### Cloning the repository

Start by cloning the repository:

```bash
git clone --recursive git@github.com:EPFL-LGG/Cshells.git
cd Cshells
```

### Python environment

The dependencies can be installed as follow

```bash
conda create -n cshell_env python=3.8
conda activate cshell_env
conda install -y pytorch=2.0 -c pytorch
conda install -y scipy matplotlib
conda install -y 'jupyterlab>=3.2'
pip install geomdl==5.3
```

To deactivate the environment, simply execute:

```
conda deactivate
```

### Build instructions
You can build the project as follow (**you need to have the environment activated**):

```
mkdir build && cd build
cmake .. -GNinja
ninja
```

### Setting up jupyter lab

Install `pythreejs` using

```bash
conda install -y pythreejs -c conda-forge
jupyter lab build
```

If this fails, you can try to install the source. Go to the `ext/` folder and clone:

```
git clone https://github.com/jpanetta/pythreejs
cd pythreejs
pip3 install -e .
cd js
jupyter labextension install .
```

### Running a jupyter notebook

To run a notebook, first launch jupyter lab in the one of the parent folders:

```
jupyter lab
```

Make sure you have activated the environment first.

## Running code

Three main actions can be performed on C-shells: deployment, design optimization, and planarization.

### Deployments

A `CShell` object (defined in `python/CShell.py`) can be instanciated and deployed as shown in the notebook `notebooks/deployments/deployments.ipynb`. A plurality of designs to deploy can be found under `data/models`. A linkage can be deployed either by specifying an attraction surface (see `src/AverageAngleSurfaceAttractedLinkage.hh`) or by fixing some degrees of freedom (see `src/AverageAngleLinkage.hh`). We show different deployment strategies in `notebooks/deployments/deployments_torus_symmetric.ipynb`

### Design Optimization

Once deployed, a `CShell` can be optimized towards a prescribed target surface while keeping the elastic energy in the deformed state low. Some examples can be found under `notebooks/design_optimizations`. In particular `optim_torus_symmetric.ipynb` shows an example of radially symmetric design. The notebook `optim_hexagon_xshell.ipynb` optimizes a design using the [X-shell](https://julianpanetta.com/publication/xshells/) method in the current code framework.

### Planarization

If a target surface in the form of a B-spline surface is known in advance, one may apply the planarization algorithm as shown in `notebooks/planarizations`. The joints are laid out in a plane and optimized so that the C-shell formed by connecting the joints is a good initial guess for further design optimization towards the user defined target surface.