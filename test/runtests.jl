using BenchmarkTools, ForwardDiff, Gridap, LinearAlgebra, Printf, SparseArrays, Test

#JSO
using JSOSolvers, Krylov, NLPModels, NLPModelsIpopt

#PDENLPModels
using PDENLPModels
using PDENLPModels: FEFunctionType, _split_vector, _split_FEFunction,
                    _obj_integral, _obj_cell_integral, _compute_gradient_k, _compute_gradient!, _compute_hess_coo

use_derivative_check = false #set true to derivative_check (this is slow)

#Test constructors, util_functions.jl and additional_obj_terms.jl
include("unit-test.jl")

#I. Test on unconstrained problems:
#Elementary tests on an unconstrained problem
include("test-unconstrained.jl")
#Unconstrained optimization <=> Laplacian equation
include("test-unconstrained-2.jl")

#II. Elementary tests on a PDE problem (no objective fct and no other constraints)
#Nonlinear with mutli-field
include("pde-only-incompressible-NS.jl")

#III. Optimization problem with PDE constraints:
#Mixed boundary conditions, and a source term.
#include("poisson-with-Neumann-and-Dirichlet.jl") #Peut être décommenter
#include("1d-Burger-example.jl") #Peut être décommenter

#IV. Mixed optimization problem with PDE-constraints
#Objective only on the parameter
include("poisson-with-parameter-optim.jl")
#Mixed objectives with no intertwined terms
include("poisson-with-mixed-optim.jl")
#Mixed objectives with intertwined terms
include("poisson-with-true-mixed-optim.jl") #TODO check the hessian computation
