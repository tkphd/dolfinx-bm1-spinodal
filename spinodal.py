# -*- coding: utf-8 -*-
# mpirun -np 4 --mca opal_cuda_support 0 python cahn-hilliard.py

"""The Cahn-Hilliard equation is a fourth-order equation, so casting it in a
weak form would result in the presence of second-order spatial derivatives, and
the problem could not be solved using a standard Lagrange finite element basis.
A solution is to rephrase the problem as two coupled second-order equations.
The unknown fields are c and μ.

* Ω=(0,1)×(0,1) (unit square)
* f=100c²(1−c)²
* λ=1×10⁻²
* M=1
* dt=5×10⁻⁶
* θ=0.5

"""

import numpy as np
import os

from dolfinx import (Form, Function, FunctionSpace, NewtonSolver,
                     UnitSquareMesh, fem, log, plot)
from dolfinx.cpp.mesh import CellType
from dolfinx.fem.assemble import assemble_matrix, assemble_vector
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (FiniteElement, TestFunctions, TrialFunction, derivative, diff,
                 dx, grad, inner, split, variable)

rank = MPI.COMM_WORLD.Get_rank()

# Model parameters
λ = 1.0e-2 # surface parameter
dt = 5.0e-6 # timestep
θ = 0.5    # Crank-Nicolson

# Output
log.set_output_file("dolfinx-spinodal.log")
hdf = XDMFFile(MPI.COMM_WORLD, "dolfinx-spinodal.xdmf", "w")

"""
CahnHilliardEquation: a subclass of NonlinearProblem invoking the Newton solver
"""

class CahnHilliardEquation:
    def __init__(self, a, L):
        self.L, self.a = Form(L), Form(a)

    def form(self, x):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                      mode=PETSc.ScatterMode.FORWARD)

    def F(self, x, b):
        with b.localForm() as f:
            f.set(0.0)
        assemble_vector(b, self.L)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)

    def J(self, x, A):
        A.zeroEntries()
        assemble_matrix(A, self.a)
        A.assemble()

    def matrix(self):
        return fem.create_matrix(self.a)

    def vector(self):
        return fem.create_vector(self.L)

# Create mesh & element basis
Ω = UnitSquareMesh(MPI.COMM_WORLD, 96, 96, CellType.triangle) # mesh
LE = FiniteElement("Lagrange", Ω.ufl_cell(), 1)

# Create the function space from both the mesh and the element
FS = FunctionSpace(Ω, LE * LE)

# Build the solution, trial, and test functions
u  = Function(FS) # current solution
u0 = Function(FS) # previous solution
du = TrialFunction(FS)
q, v = TestFunctions(FS)

# Mixed functions
c, μ = split(u) # references to components of u for clear, direct access
dc, dμ = split(du)
c0, μ0 = split(u0)

# === Initial Conditions ===

with u.vector.localForm() as x:
    x.set(0.0)

noisy = lambda x: 0.63 + 0.02 * (0.5 - np.random.rand(x.shape[1]))
u.sub(0).interpolate(noisy)

# ChemPot
c = variable(c) # declare this as a variable things can be differentiated wrt
f = 100 * c**2 * (1 - c)**2
dfdc = diff(f, c)

# === Weak Form ===

# Half-stepping parameter for Crank-Nicolson
μ_mid = (1.0 - θ) * μ0 + θ * μ

# Discretization in UFL syntax
L0 = inner(c, q) * dx - inner(c0, q) * dx + dt * inner(grad(μ_mid), grad(q)) * dx
L1 = inner(μ, v) * dx - inner(dfdc, v) * dx - λ * inner(grad(c), grad(v)) * dx
L = L0 + L1

# Jacobian of L
J = derivative(L, u, du)

# === Solver ===

problem = CahnHilliardEquation(J, L)
solver = NewtonSolver(MPI.COMM_WORLD)
solver.setF(problem.F, problem.vector())
solver.setJ(problem.J, problem.matrix())
solver.set_form(problem.form)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-6

# Prepare for timestepping

t = 0.0
T = 50 * dt

hdf.write_mesh(Ω)
u.vector.copy(result=u0.vector)
u0.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                      mode=PETSc.ScatterMode.FORWARD)

# === TIMESTEPPING ===

while (t < T):
    t += dt
    r = solver.solve(u.vector)
    if rank == 0:
        print("Step {:6d}: {:6d} iterations".format(int(t / dt), r[0]))
    u.vector.copy(result=u0.vector)
    hdf.write_function(u.sub(0), t)

hdf.close()

# === VISUALIZATION ===

# Open the XDMF file (XML-HDF5) in VisIT, ParaView, or equivalent tool.
