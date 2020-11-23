using BenchmarkTools, ForwardDiff, Gridap, JSOSolvers, Krylov, LinearAlgebra, NLPModels, SparseArrays, Test

using PDENLPModels
using PDENLPModels: FEFunctionType, _split_vector, _split_FEFunction,
                         _obj_integral, _obj_cell_integral, _compute_gradient_k!

include("unit-test.jl")

#I. Test on GridapPDENLPModel:
#Elementary tests on an unconstrained problem
include("test-unconstrained.jl")
#Elementary tests on a PDE problem (no objective fct and no other constraints)
include("pde-only-incompressible-NS.jl")
#Unconstrained optimization <=> Laplacian equation
include("test-unconstrained-2.jl")

#Optimization problem with PDE constraint:
#Laplacian with Dirichlet boundary conditions
#Mixed boundary conditions, and a source term.
#include("poisson-with-Neumann-and-Dirichlet.jl") #Peut être décommenter
#include("1d-Burger-example.jl") #Peut être décommenter
