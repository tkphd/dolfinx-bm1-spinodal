# -*- coding: utf-8 -*-

# mpirun -np 4 --mca opal_cuda_support 0 python cahn-hilliard.py

"""
 𝔉 = ∫{𝐹 + 0.5⋅𝜅⋅|∇𝑐|²}⋅d𝛺
 𝐹 = 𝜌⋅(𝑐 - 𝛼)²⋅(𝛽 - 𝑐)²

 ∂𝑐/∂𝑡= ∇⋅{𝑀 ∇(𝑓 - 𝜅∇²𝑐)}

 𝛺 = (0,200)×(0,200) (unit square)
"""

import csv
import gc
import numpy as np
import queue

from dolfinx import Form, Function, FunctionSpace, NewtonSolver, RectangleMesh
from dolfinx import fem, log

from dolfinx.cpp.mesh import CellType
from dolfinx.fem.problem import NonlinearProblem
from dolfinx.fem.assemble import assemble_matrix, assemble_scalar, assemble_vector
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py import PETSc
from ufl import FiniteElement, Measure, TestFunctions, TrialFunction
from ufl import  derivative, diff, dx, grad, inner, split, variable

epoch = MPI.Wtime()

# Model parameters
𝜅 = 2    # gradient energy coefficient
𝜌 = 5    # well height
𝛼 = 0.3  # eqm composition of phase 1
𝛽 = 0.7  # eqm composition of phase 2
𝜁 = 0.5  # system composition
𝑀 = 5    # interface mobility
𝜀 = 0.01 # noise amplitude

# Discretization parameters
𝐿 = 200  # width
𝑁 = 400  # cells
Δ𝑡 = 0.1 # timestep
𝑇 = 1e6  # simulation timeout

p_deg = 2 # element/polynomial degree
q_deg = 4 # quadrature_degree

# Output
log.set_output_file("dolfinx-spinodal.log")
pfhub_log = "dolfinx-bm-1b.csv"
hdf = XDMFFile(MPI.COMM_WORLD, "dolfinx-spinodal.xdmf", "w")


"""
class CahnHilliardEquation:
    def __init__(self, a, L):
        self.L, self.a = Form(L), Form(a)

    def F(self, x, b):
        with b.localForm() as f:
            f.set(0.0)
        assemble_vector(b, self.L)

    def J(self, x, A):
        A.zeroEntries()
        assemble_matrix(A, self.a)
        A.assemble()

    def matrix(self):
        return fem.create_matrix(self.a)

    def vector(self):
        return fem.create_vector(self.L)
"""

# Create mesh & element basis
𝛺 = RectangleMesh(
    MPI.COMM_WORLD,
    [np.array([0, 0, 0]), np.array([𝐿, 𝐿, 0])],
    [𝑁, 𝑁],
    CellType.triangle,
    diagonal="crossed"
)

MPI_COMM_WORLD = 𝛺.mpi_comm()
rank = MPI_COMM_WORLD.Get_rank()

LE = FiniteElement("Lagrange", 𝛺.ufl_cell(), p_deg)

# Create the function space from both the mesh and the element
FS = FunctionSpace(𝛺, LE * LE)

# Build the solution, trial, and test functions
u = Function(FS)  # current solution
u0 = Function(FS)  # previous solution
du = TrialFunction(FS)
q, v = TestFunctions(FS)

# Mixed functions
𝑐, 𝜇 = split(u)  # references to components of u for clear, direct access
d𝑐, d𝜇 = split(du)
# 𝑏, 𝜆 are the previous values for 𝑐, 𝜇
𝑏, 𝜆 = split(u0)

𝑐 = variable(𝑐)
𝐹 = 𝜌 * (𝑐 - 𝛼) ** 2 * (𝛽 - 𝑐) ** 2
dfdc = diff(𝐹, 𝑐)

# === Weak Form ===

# Half-stepping parameter for Crank-Nicolson
𝜃 = 0.5  # Crank-Nicolson parameter
𝜇_mid = (1 - 𝜃) * 𝜆 + 𝜃 * 𝜇

# Time discretization in UFL syntax
# (𝑏 is the previous timestep)
L0 = inner(𝑐, q) * dx - inner(𝑏, q) * dx + Δ𝑡 * inner(grad(𝜇_mid), grad(q)) * dx
L1 = inner(𝜇, v) * dx - inner(dfdc, v) * dx - 𝜅 * inner(grad(𝑐), grad(v)) * dx
L = L0 + L1

# Jacobian of L
# J = derivative(L, u, du)

# === Solver ===

problem = NonlinearProblem(L, u)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
#solver.setF(problem.F, problem.vector())
#solver.setJ(problem.J, problem.matrix())
solver.convergence_criterion = "incremental"
solver.rtol = 1e-6

# PETSc options
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

# === Initial Conditions ===

with u.vector.localForm() as x:
    x.set(0.0)

noisy = lambda x: 𝜁 + 𝜀 * (
    np.cos(0.105 * x[0]) * np.cos(0.11 * x[1])
    + (np.cos(0.13 * x[0]) * np.cos(0.087 * x[1])) ** 2
    + np.cos(0.025 * x[0] - 0.15 * x[1]) * np.cos(0.07 * x[0] - 0.02 * x[1])
)

u.sub(0).interpolate(noisy)

hdf.write_mesh(𝛺)
u.vector.copy(result=u0.vector)
u0.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

# === TIMESTEPPING ===

# Enqueue output timestamps
io_q = queue.Queue()
for t_out in np.arange(0.1, 1, 0.1):
    io_q.put(t_out)
for n in np.arange(0, 7):
    for m in np.arange(1, 10):
        t_out = m * 10.0 ** n
        if (t_out <= 𝑇):
            io_q.put(t_out)

# Endpoint detection based on Δ𝜇 is borrowed from @smondal44,
# <https://github.com/smondal44/spinodal-decomposition>

𝑡 = 0.0
𝑛 = MPI_COMM_WORLD.allreduce(len(𝛺.geometry.x), op=MPI.SUM)

start = MPI.Wtime()

def crunch_the_numbers(𝑡, 𝑐, 𝜇, 𝜆, r, t0):
    𝓕 = assemble_scalar(𝜌 * (𝑐 - 𝛼) ** 2 * (𝛽 - 𝑐) ** 2 * dx \
                        + 0.5 * 𝜅 * inner(grad(𝑐), grad(𝑐)) * dx)

    Δ𝜇= assemble_scalar(np.abs(𝜇 - 𝜆) * dx)

    𝓕 = MPI_COMM_WORLD.allreduce(𝓕, op=MPI.SUM)
    Δ𝜇= MPI_COMM_WORLD.allreduce(Δ𝜇 / 𝑛, op=MPI.SUM)
    r = MPI_COMM_WORLD.allreduce(r, op=MPI.MAX)

    return (𝑡, 𝓕, Δ𝜇, r, MPI.Wtime() - t0)

summary = crunch_the_numbers(𝑡, 𝑐, 𝜇, 𝜆, 0, start)

if rank == 0:
    with open(pfhub_log, mode="w") as nrg_file:
        io = csv.writer(nrg_file)

        header = ["time", "free_energy", "driving_force", "iterations", "runtime"]
        io.writerow(header)
        io.writerow(summary)

Δ𝜇 = 1.0
io_t = io_q.get()

if rank == 0:
    print("[{}] Next summary at 𝑡={}".format(MPI.Wtime() - epoch, io_t))

while (Δ𝜇 > 1e-8) and (𝑡 < 𝑇):
    𝑡 += Δ𝑡
    r = solver.solve(u)[0]
    u.vector.copy(result=u0.vector)

    if 𝑡 >= io_t:
        summary = crunch_the_numbers(𝑡, 𝑐, 𝜇, 𝜆, r, start)
        hdf.write_function(u.sub(0), 𝑡)

        if rank == 0:
            with open(pfhub_log, mode="a") as nrg_file:
                io = csv.writer(nrg_file)
                io.writerow(summary)

        io_t = io_q.get()

        if rank == 0:
            print("[{}] Next summary at 𝑡={}".format(MPI.Wtime() - epoch, io_t))

        gc.collect()


hdf.write_function(u.sub(0), 𝑡)

summary = crunch_the_numbers(𝑡, 𝑐, 𝜇, 𝜆, r, start)
if rank == 0:
    with open(pfhub_log, mode="a") as nrg_file:
        io = csv.writer(nrg_file)
        io.writerow(summary)

hdf.close()

print("Finished simulation after {} s.".format(MPI.Wtime() - epoch))
