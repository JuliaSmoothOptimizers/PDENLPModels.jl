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

nlp = Burgernlp()

using ForwardDiff
function hessian_lagrangian_test_functions(nlp :: GridapPDENLPModel; udc = false, tol = 1e-15)

  n, m, p = nlp.meta.nvar, nlp.meta.ncon, nlp.nparam
  nnzh = nlp.meta.nnzh #Lagrangian function
  #Why not an nnzh for the Hessian of the objective function?
  x0, y0 = nlp.meta.x0, nlp.meta.y0
  xr, yr = rand(n), rand(m)

  Hx = hess(nlp, x0)
  Hr = hess(nlp, xr)
  v0 = hess_coord(nlp, x0)
  vr = hess_coord(nlp, xr)
  (rows, cols) = hess_obj_structure(nlp)
  nnzh_obj = length(rows)
  #Si 1 term, et même trian:
  if length(nlp.op.terms) == 1
    @test nnzh == 2 * nnzh_obj 
  end

  #Compare when Lagrangian vanishes:
  Hx0 = hess(nlp, x0, zeros(m))
  Hxr = hess(nlp, xr, zeros(m))

  @test Hx0 == Hx
  @test Hxr == Hr

  #The old dense function (very slow)
  function hess2(nlp :: GridapPDENLPModel,
                 x   :: AbstractVector,
                 λ   :: AbstractVector;
                 obj_weight :: Real = one(eltype(x)))
    @lencheck nlp.meta.nvar x
    @lencheck nlp.meta.ncon λ
    function ℓ(x, λ)
      κ, xyu = x[1 : nlp.nparam], x[nlp.nparam + 1 : nlp.meta.nvar]
      yu  = FEFunction(nlp.Y, xyu)
      int = PDENLPModels._obj_integral(nlp.tnrj, κ, yu)

      c = similar(x, nlp.meta.ncon)
      PDENLPModels._from_terms_to_residual!(nlp.op, x, nlp, c)

      return obj_weight * sum(int) + dot(c, λ)
    end

    #ℓ(x) = obj_weight * obj(nlp, x) + dot(cons(nlp, x), λ)
    Hx = ForwardDiff.hessian(x->ℓ(x, λ), x)
    return tril(Hx)
  end

  Hxy  = hess(nlp, x0, y0)
  @test Hxy == hess2(nlp, x0, y0)
  Hxyr = hess(nlp, xr, yr)
  @test Hxyr == hess2(nlp, xr, yr)

  (rows, cols) = hess_structure(nlp)
  @test (length(rows) == nnzh) && length(cols) == nnzh
  vxy0 = hess_coord(nlp, x0, y0)
  vxyr = hess_coord(nlp, xr, yr)
  @test length(vxy0) == nnzh
  @test length(vxyr) == nnzh

  return true
end

hessian_lagrangian_test_functions(nlp)

#II. Elementary tests on a PDE problem (no objective fct and no other constraints)
#Nonlinear with mutli-field
@info "PDE-only incompressible Navier-Stokes"
include("pde-only-incompressible-NS.jl")