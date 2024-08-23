# Getting started
This page provides instructions for installing CADDEE.

## Installation
```{note}
   As CADDEE is currently still under active development, we recommend following the installation instruction for developers. In addition, we recommend installing CADDEE and all subsequent packages within a [conda](https://docs.anaconda.com/miniconda/) environment.
```

### Installation instructions for developers
First, create a conda environment with Python 3.9 or 3.10
```sh
conda create -n caddee_env python=3.9
conda activate caddee_env
```

To install `CADDEE_alpha`, first clone the repository and install using pip. On the terminal or command line, run
```sh
git clone https://github.com/LSDOlab/CADDEE_alpha.git
cd CADDEE_alpha
pip install -e ./CADDEE_alpha
```

### Installation instructions for users (for future, stable versions)
For direct installation with core dependencies, run on the terminal or command line
```sh
pip install git+https://github.com/LSDOlab/CADDEE_alpha.git
```

For installation with solver toolsuite, run
```sh
pip install git+https://github.com/LSDOlab/CADDEE_alpha.git[toolsuite]
```


### Additional packages
CADDEE is a toolsuite for integrating physics-based solvers to build complex multi-disciplinary models of engineering systems. Therefore, a number of external packages are available and need to be installed. to take full advantage of CADDEE's capabilities. In future versions of CADDEE, these packages will be automatically installed. The following table provides an overview of all the packages that can be interfaced with CADDEE. The installation of these packages is identical to installing CADDEE, with the exception that if the working branch is not `main`, the corresponding branch needs to be "checked out" via `git checkout branch_name`, where `branch_name` is the one listed in the right column of the table below.


| Package                                                                 | Classification | Description                                                                                                                                                                                                                           | Working branch (for CADDEE scripts) | Notes                                                                                             |
|-------------------------------------------------------------------------|----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------|---------------------------------------------------------------------------------------------------|
| [CSDL]( https://github.com/LSDOlab/CSDL_alpha )                         | Core           | Computational System Design Language; Python-based algebraic modeling language for building multidisciplinary <br> computational models of engineering systems with automated  adjoint-based sensitivity analysis                         | main                                | Requires python>=3.9                                                                              |
| [modopt](https://github.com/LSDOlab/modopt)                             | Toolsuite      | A MODular development environment and library for OPTimization algorithms                                                                                                                                                             | main                                |                                                                                                   |
| [lsdo_geo](https://github.com/LSDOlab/lsdo_geo)                         | Core           | Geometry engine for efficient manipulation of geometries via free-form deformation techniques                                                                                                                                         | main                |                                                                                                   |
| [lsdo_function_spaces](https://github.com/LSDOlab/lsdo_function_spaces) | Core           | Package that enables the solver-independent field representation of field quantities (e.g., pressure) via a host of functions <br> that can be fit to solver data (note: mind the installation instructions regarding Cython)              | main                                | special instructions for installing Cython (*see below)                                            |
| [Aframe](https://github.com/LSDOlab/aframe)                             | Toolsuite      | Linear (Euler--Bernoulli) beam solver                                                                                                                                                                                                 | 2                                   |                                                                                                   |
| [FEMO](https://github.com/LSDOlab/femo_alpha)                           | Additional     | Finite Element in PDE-constrained Multidisciplinary Optimization problems                                                                                                                                                             | main                                | package leverages [FEniCS](https://fenicsproject.org/)  and has special installation instructions |
| [VortexAD](https://github.com/lscotzni/VortexAD_temp)                   | Toolsuite      | Vortex-based (i.e., low-speed) aerodynamic solver with <br> - steady/unsteady vortex lattice method (VLM) <br> - steady/unsteady panel method                                                                                                  | dev_ml (steady VLM only)            |                                                                                                   |
| [BladeAD](https://github.com/LSDOlab/BladeAD)                           | Toolsuite      | Rotor aerodynamic solver with implementations of:<br> - Blade element moment theory<br> - Dynamic inflow (steady)<br>  &nbsp;&nbsp;&nbsp; - Pitt--Peters <br> &nbsp;&nbsp;&nbsp; - Peters--He                                                                       | main                                |                                                                                                   |
| [lsdo_acoustics](https://github.com/LSDOlab/lsdo_acoustics)             | Toolsuite      | Rotor aeroacoustic models for tonal and broadband noise<br> - Tonal: Lowson (w/ Barry--Magliozzi thickness noise component)<br> - Broadband:<br> &nbsp;&nbsp;&nbsp;  - Gill-Lee semi-empirical<br>  &nbsp;&nbsp;&nbsp;  - Brook--Pope--Marcolini (under development)<br> | dev_csdl_alpha                      |                                                                                                   |
| [lsdo_airfoil](https://github.com/LSDOlab/lsdo_airfoil)                 | Additional     | Subsonic machine learning model for airfoil aerodynamics (based on XFOIL training data) for coefficients <br> and boundary layer parameters and airfoil shape parameterization                                                             | main                                | requires [PyTorch](https://pytorch.org/)                                                          |


<!-- #### *Special installation instructions for lsdo_function_spaces
This package uses Cython for better performance of projections. A few extra steps are required for installation on Ubuntu (MacOS not tested but should work the same):
```sh
conda install cython=0.29.28
git clone https://github.com/LSDOlab/lsdo_b_splines_cython.git
pip install -e ./lsdo_b_spline_cython
git clone https://github.com/LSDOlab/lsdo_function_spaces.git
pip install -e ./lsdo_function_spaces
``` -->

<!-- #### The Computational System Design Language ([CSDL](https://csdl-alpha.readthedocs.io/en/latest/))
CSDL is a domain-embedded language for building computational models with automated (adjoint-based) sensitivity analysis. CADDEE relies on solvers being implemented in CSDL in order to perform large-scale design optimziation. 

All packages listed on this page use or interface with CSDL.
```sh
git clone https://github.com/LSDOlab/CSDL_alpha.git
cd CSDL_alpha
pip install -e ./CSDL_alpha
```

#### Geometry engine: lsdo_geo
CADDEE takes a geometry-centric approach to conceptual design of engineering systems. This allows for seamless manipulation of the central geometry and automated updating of meshes and other geometry-related solver inputs.
```sh
git clone https://github.com/LSDOlab/lsdo_geo.git
cd lsdo_geo
pip install -e ./lsdo_geo
``` -->
