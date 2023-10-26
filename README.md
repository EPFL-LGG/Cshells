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

## Getting started

### Setup for elastic_rods

Before cloning the repository, make sure you have installed all the dependencies required by [elastic rods](https://github.com/EPFL-LGG/elastic_rods/tree/14a55673b7c65baf2cd410583203abc606965d1b). The design optimization and planarization parts of the code also depend on the commercial optimization package [`knitro`](https://www.artelys.com/solvers/knitro/); these will be omitted from the build if `knitro` is not found.

### Cloning the repository

Start by cloning the repository:

```bash
git clone --recursive git@github.com:EPFL-LGG/Cshells.git
cd Cshells
```

### Python environement

The dependencies can be installed as follow

```bash
conda create -n CShellEnv python=3.8
conda activate CShellEnv
conda install pytorch=2.0 -c pytorch
conda install scipy matplotlib
pip install opencv-python
conda install 'jupyterlab>=3.2'
pip install specklepy
pip install open3d==0.16 geomdl==5.3
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
conda install -c conda-forge pythreejs
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
