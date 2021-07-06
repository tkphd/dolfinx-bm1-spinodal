# FEniCS/DOLFINx/MPC & PFHub BM 1: Spinodal Decomposition

This repository presents an implementation of the PFHub
[Spinodal Decomposition Benchmark](
https://pages.nist.gov/pfhub/benchmarks/benchmark1.ipynb/)
using DOLFINx, the experimental version of FEniCS.
The basic code for this implementation is lifted from the
Cahn-Hilliard example provided by the [DOLFINx Docs](
https://docs.fenicsproject.org/dolfinx/main/python/demos/cahn-hilliard/demo_cahn-hilliard.py.html).

I use the Spack package for ease of installation. Note that
it requires a relatively new version of GCC: I have found
`gcc-10.3.0` works well. To install it, do

```bash
$ spack install gcc@10.3.0
$ spack compiler add $(spack location -i gcc@10.3.0)
$ spack compiler find
```

Then you should be able to follow the [Spack instructions](
https://github.com/FEniCS/dolfinx#spack) without too much
more difficulty.

## DOLFINx-MPC

Visit the [DOLFINx MPC repo](https://github.com/jorgensd/dolfinx_mpc)
for details on installation. Docker is preferred. `cd` to your preferred
working directory, then invoke

```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared dokken92/dolfinx_mpc:0.1.0
```

(DOLFINx MPC provides for periodic boundary conditions. At present, stock
DOLFINx does not.)
