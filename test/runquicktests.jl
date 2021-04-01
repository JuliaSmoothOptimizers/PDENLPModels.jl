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

#III. Optimization problem with PDE constraints:
@info "1d Burger's equation"
include("1d-Burger-example.jl") #Peut être décommenter

#II. Elementary tests on a PDE problem (no objective fct and no other constraints)
#Nonlinear with mutli-field
@info "PDE-only incompressible Navier-Stokes"
include("pde-only-incompressible-NS.jl")

nlp = Burgernlp()

hessian_lagrangian_test_functions(nlp)