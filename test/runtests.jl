using ForwardDiff, Gridap, JSOSolvers, LinearAlgebra, NLPModels, Test
using Main.PDENLPModels

#I. Test on GridapPDENLPModel:
#Elementary tests on an unconstrained problem
include("test-unconstrained.jl")
#Elementary tests on a PDE problem (no objective fct and no other constraints)
include("pde-only-incompressible-NS.jl") #TODO check hprod! here.
#Unconstrained optimization <=> Laplacian equation
include("test-unconstrained-2.jl")

#Optimization with a simple equality constraint - smallest eigenvalue of the Laplacian
include("smallest-laplacian-eigenvalue.jl") #TODO

#Optimization problem with PDE constraint:

#Laplacian with Dirichlet boundary conditions
include("control-elastic-membrane.jl") #TODO
#Mixed boundary conditions, and a source term.
include("poisson-with-Neumann-and-Dirichlet.jl") #TODO check hprod! here.

#The three examples in the paper IMPLEMENTING A SMOOTH EXACT PENALTY FUNCTION FOR EQUALITY-CONSTRAINED NONLINEAR OPTIMIZATION
include("1d-Burger-example.jl") #TODO check hprod! here + solver
include("distributed-Poisson-control-with-Dirichlet-bdry.jl") #TODO
include("2d-poisson-Boltzman-problem.jl") #TODO
