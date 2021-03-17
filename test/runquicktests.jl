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

function Burgernlp()
  n = 512
  domain = (0,1)
  partition = n
  model = CartesianDiscreteModel(domain,partition)

  labels = get_face_labeling(model)
  add_tag_from_tags!(labels,"diri1",[2])
  add_tag_from_tags!(labels,"diri0",[1])

  D = 1
  order = 1
  V = TestFESpace(
      reffe=:Lagrangian, conformity=:H1, valuetype=Float64,
      model=model, labels=labels, order=order, dirichlet_tags=["diri0","diri1"])

  h(x) = 2*(nu + x[1]^3)
  uD0 = VectorValue(0)
  uD1 = VectorValue(-1)
  U = TrialFESpace(V,[uD0,uD1])

  @law conv(u,∇u) = (∇u ⋅one(∇u))⊙u
  @law dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

  function a(u,v)
    ∇(v)⊙∇(u)
  end

  c(u,v) = v⊙conv(u,∇(u))
  nu = 0.08
  function res_pde(u,v)
    z(x) = 0.5
    -nu*(∇(v)⊙∇(u)) + c(u,v) - v * z - v * h
  end

  trian = Triangulation(model)
  @test Gridap.FESpaces.num_cells(trian) == 512
  degree = 1
  quad = CellQuadrature(trian,degree)
  t_Ω = FETerm(res_pde,trian,quad)
  op_pde = FEOperator(U,V,t_Ω)

  #Now we move to the optimization:
  ud(x) = -x[1]^2
  α = 1e-2

  #objective function:
  f(u, z) = 0.5 * (ud - u) * (ud - u) + 0.5 * α * z * z
  function f(yu) #:: Union{Gridap.MultiField.MultiFieldFEFunction, Gridap.CellData.GenericCellField}
    u, z = yu
    f(u,z)
  end

  function res(yu, v) #u is the solution of the PDE and z the control
    u, z = yu
    v

    -nu*(∇(v)⊙∇(u)) + c(u,v) - v * z - v * h
  end
  t_Ω = FETerm(res,trian,quad)
  op = FEOperator(U, V, t_Ω)

  Xcon = TestFESpace(
            reffe=:Lagrangian, order=1, valuetype=Float64,
            conformity=:H1, model=model)
  Ycon = TrialFESpace(Xcon)

  @test Gridap.FESpaces.num_free_dofs(U) < Gridap.FESpaces.num_free_dofs(Ycon)
  #################################################################################

  Y = MultiFieldFESpace([U, Ycon])
  xin = zeros(Gridap.FESpaces.num_free_dofs(Y))
  nlp = GridapPDENLPModel(xin, f, trian, quad, U, Ycon, V, Xcon, op)
  return nlp
end

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
#=
  Hxy  = hess(nlp, x0, y0)
  @test Hxy == hess2(nlp, x0, y0)
  Hxyr = hess(nlp, xr, yr)
  @test Hxyr == hess2(nlp, xr, yr)
=#
  (rows, cols) = hess_structure(nlp)
  @test (length(rows) == nnzh) && length(cols) == nnzh
  vxy0 = hess_coord(nlp, x0, y0)
  vxyr = hess_coord(nlp, xr, yr)
  @test length(vxy0) == nnzh
  @test length(vxyr) == nnzh

  return true
end

hessian_lagrangian_test_functions(nlp)

#III. Optimization problem with PDE constraints:
@info "1d Burger's equation"
include("1d-Burger-example.jl") #Peut être décommenter

#II. Elementary tests on a PDE problem (no objective fct and no other constraints)
#Nonlinear with mutli-field
@info "PDE-only incompressible Navier-Stokes"
include("pde-only-incompressible-NS.jl")