# -*- coding: utf-8 -*-

# mpirun -np 4 --mca opal_cuda_support 0 python cahn-hilliard.py

"""
 𝔉 = ∫{𝐹 + 0.5⋅𝜅⋅|∇𝑐|²}⋅d𝛺
 𝐹 = 𝜌⋅(𝑐 - 𝛼)²⋅(𝛽 - 𝑐)²

 ∂𝑐/∂𝑡= ∇⋅{𝑀 ∇(𝑓 - 𝜅∇²𝑐)}

 𝛺 = (0,200)×(0,200) (unit square)
"""

from numpy import array, cos

from dolfinx import (Form, Function, FunctionSpace, NewtonSolver,
                     RectangleMesh, fem, log, plot)
from dolfinx.cpp.mesh import CellType
from dolfinx.fem.assemble import assemble_matrix, assemble_vector
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (FiniteElement, Measure, TestFunctions, TrialFunction,
                 derivative, diff, grad, inner, split, variable)

rank = MPI.COMM_WORLD.Get_rank()

# Model parameters
𝜅 = 2    # gradient energy coefficient
𝜌 = 5    # well height
𝛼 = 0.3  # eqm composition of phase 1
𝛽 = 0.7  # eqm composition of phase 2
𝜁 = 0.5  # system composition
𝑀 = 5    # interface mobility
𝜀 = 0.01 # noise amplitude

# Discretization parameters
𝐿 = 200 # width
𝑁 = 128 # cells
Δ𝑡= 1   # timestep

p_deg = 2 # element/polynomial degree
q_deg = 4 # quadrature_degree

# Output
log.set_output_file("dolfinx-spinodal.log")
hdf = XDMFFile(MPI.COMM_WORLD, "dolfinx-spinodal.xdmf", "w")

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
𝛺 = RectangleMesh(MPI.COMM_WORLD,
                  [array([0,0,0]), array([𝐿,𝐿,0])],
                  [𝑁,𝑁], CellType.triangle)
LE = FiniteElement("Lagrange", 𝛺.ufl_cell(), p_deg)

dx = Measure("dx", metadata={"quadrature_degree": q_deg})

# Create the function space from both the mesh and the element
FS = FunctionSpace(𝛺, LE * LE)

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

noisy = lambda x: 𝜁 + 𝜀 * ( cos(0.105*x[0]) * cos(0.11*x[1])
                          + (cos(0.13*x[0]) * cos(0.087*x[1]))**2
                          + cos(0.025*x[0] - 0.15*x[1])
                            * cos(0.07*x[0] - 0.02*x[1])
)

u.sub(0).interpolate(noisy)

c = variable(c)
𝐹 = 𝜌 * (c - 𝛼)**2 * (𝛽 - c)**2
dfdc = diff(𝐹, c)

# === Weak Form ===

# Half-stepping parameter for Crank-Nicolson
𝜃 = 0.5  # Crank-Nicolson parameter
μ_mid = (1 - 𝜃) * μ0 + 𝜃 * μ

# Time discretization in UFL syntax
# (c0 is the previous timestep)
L0 = inner(c, q) * dx - inner(c0, q) * dx \
   + Δ𝑡 * inner(grad(μ_mid), grad(q)) * dx
L1 = inner(μ, v) * dx - inner(dfdc, v) * dx \
   - 𝜅 * inner(grad(c), grad(v)) * dx
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
T = 100

hdf.write_mesh(𝛺)
u.vector.copy(result=u0.vector)
u0.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                      mode=PETSc.ScatterMode.FORWARD)

# === TIMESTEPPING ===

while (t < T):
    t += Δ𝑡
    r = solver.solve(u.vector)
    if rank == 0:
        print("Step {:6d}: {:6d} iterations".format(int(t / Δ𝑡), r[0]))
    u.vector.copy(result=u0.vector)
    hdf.write_function(u.sub(0), t)

hdf.close()

# === VISUALIZATION ===

# Open the XDMF file (XML-HDF5) in VisIT, ParaView, or equivalent tool.
