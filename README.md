# FEniCS/DOLFINx & PFHub BM 1: Spinodal Decomposition

This repository presents an implementation of the PFHub
[Spinodal Decomposition Benchmark](
https://pages.nist.gov/pfhub/benchmarks/benchmark1.ipynb/)
using DOLFINx, the experimental version of FEniCS.

The easiest way to run DOLFINx is via Docker.

* To get the CLI version:
  ```bash
  docker run -ti dolfinx/dolfinx
  ```
  This will `chroot` your terminal into the container.

* If you prefer the Jupyter Lab interface:
  ```bash
  docker run --init -ti -p 8888:8888 dolfinx/lab
  ```
  Point your browser to <http://localhost:8888> and get hacking.

The basic code for this implementation is lifted from the
Cahn-Hilliard example provided by the [DOLFINx Docs](
https://docs.fenicsproject.org/dolfinx/main/python/demos/cahn-hilliard/demo_cahn-hilliard.py.html).
