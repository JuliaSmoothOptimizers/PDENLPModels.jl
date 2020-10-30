using ForwardDiff, Gridap, JSOSolvers, LinearAlgebra, NLPModels, Test
using Main.PDENLPModels

#I. Test on GridapPDENLPModel:
#Elementary tests on an unconstrained problem
include("test-unconstrained.jl")
#Elementary tests on a PDE problem (no objective fct and no other constraints)
include("pde-only-incompressible-NS.jl")

#Optimization problem with PDE constraint:
#include("1d-Burger-example.jl")
#include("poisson-with-Neumann-and-Dirichlet.jl")

#include("distributed-Poisson-control-with-Dirichlet-bdry.jl")
