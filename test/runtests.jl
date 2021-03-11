using BenchmarkTools, ForwardDiff, Gridap, LinearAlgebra, Printf, SparseArrays, Test
using LineSearches: BackTracking
#JSO
using JSOSolvers, Krylov, NLPModels, NLPModelsIpopt

#PDENLPModels
using PDENLPModels
using PDENLPModels: FEFunctionType, _split_vector, _split_FEFunction,
                    _obj_integral, _obj_cell_integral, _compute_gradient_k, _compute_gradient!, _compute_hess_coo

use_derivative_check = false #set true to derivative_check (this is slow)
include("check-dimensions.jl")

#Test constructors, util_functions.jl and additional_obj_terms.jl
include("unit-test.jl")

#I. Test on unconstrained problems:
#Elementary tests on an unconstrained problem
@info "Unconstrained problem I"
include("test-unconstrained.jl")
#Unconstrained optimization <=> Laplacian equation
@info "Unconstrained problem II"
include("test-unconstrained-2.jl")

#II. Elementary tests on a PDE problem (no objective fct and no other constraints)
#Nonlinear with mutli-field
@info "PDE-only incompressible Navier-Stokes"
include("pde-only-incompressible-NS.jl")

#III. Optimization problem with PDE constraints:
#Mixed boundary conditions, and a source term.
@info "Poisson-equation with mixed boundary conditions [add model - work locally]"
#include("poisson-with-Neumann-and-Dirichlet.jl") #Peut être décommenter
@info "1d Burger's equation"
include("1d-Burger-example.jl") #Peut être décommenter
if false
  include("code_issue.jl")
end

#IV. Mixed optimization problem with PDE-constraints
#Objective only on the parameter
@info "Parameter optimization with Poisson-equation [broken] l.77"
#include("poisson-with-parameter-optim.jl")
#Mixed objectives with no intertwined terms
@info "Separable parameter/function optimization with Poisson-equation"
include("poisson-with-mixed-optim.jl")
#Mixed objectives with intertwined terms
@info "Intertwined parameter/function optimization with Poisson-equation"
include("poisson-with-true-mixed-optim.jl") #TODO check the hessian computation
