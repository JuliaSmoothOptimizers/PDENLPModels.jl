using ForwardDiff, Gridap, JSOSolvers, LinearAlgebra, NLPModels, Test
using Main.PDENLPModels

#I. Test on GridapPDENLPModel:
#Elementary tests on an unconstrained problem
include("test-unconstrained.jl")
#Elementary tests on a PDE problem (no objective fct and no other constraints)
include("pde-only-incompressible-NS.jl")

#Optimization problem with PDE constraint:
#include("1d-Burger-example.jl")

#include("distributed-Poisson-control-with-Dirichlet-bdry.jl")

#Test the solvers:
#On a toy rosenbrock variation.
include("test/test-solvers/test-0.jl")
#On a problem from the package OptimizationProblems
include("test/test-solvers/test-1.jl")
