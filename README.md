# FEniCS/DOLFINx/MPC & PFHub BM 1: Spinodal Decomposition

This repository presents an implementation of the PFHub
[Spinodal Decomposition Benchmark](
https://pages.nist.gov/pfhub/benchmarks/benchmark1.ipynb/)
using DOLFINx, the experimental version of FEniCS.
The basic code for this implementation is lifted from the
Cahn-Hilliard example provided by the [DOLFINx Docs](
https://docs.fenicsproject.org/dolfinx/main/python/demos/cahn-hilliard/demo_cahn-hilliard.py.html).

> *N.B.:* DOLFINx does not currently support resuming a simulation from the
>         XDMF/HDF5 checkpoint files it writes. This is a severe deficiency.

## DOLFINx & Singularity

I use Singularity to run the nightly DOLFINx build from [DockerHub]
(https://hub.docker.com/r/dolfinx/dolfinx/tags?page=1&ordering=last_updated)
inside a Conda environment. The [`Makefile`](Makefile) reflects this
environment.

* Install Miniconda and Mamba (better dependency resolver)
  ```bash
  $ curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  $ chmod +x Miniconda3-latest-Linux-x86_64.sh
  $ ./Miniconda3-latest-Linux-x86_64.sh
  «agree to EULA, specify install dir, and prepare your shell»
  $ conda install mamba -n base -c conda-forge
  ```
* Create and activate an environment containing Singularity
  ```bash
  $ mamba create -n sing python=3 singularity
  $ conda activate sing
  ```
* Move to the benchmark directory and run the benchmark
  ```bash
  $ cd ~/path/to/dolfinx-bm1-spinodal
  $ make
  ```

## DOLFINx-MPC

DOLFINx MPC provides for periodic boundary conditions. At present, stock
DOLFINx does not.
