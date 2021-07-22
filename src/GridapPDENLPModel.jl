@doc raw"""
PDENLPModels using Gridap.jl

https://github.com/gridap/Gridap.jl
Cite: Badia, S., & Verdugo, F. (2020). Gridap: An extensible Finite Element toolbox in Julia.
Journal of Open Source Software, 5(52), 2520.

Find functions (y,u): Y -> ℜⁿ x ℜⁿ and κ ∈ ℜⁿ satisfying

min      ∫_Ω​ f(κ,y,u) dΩ​
s.t.     y solution of a PDE(κ,u)=0
         lcon <= c(κ,y,u) <= ucon
         lvar <= (κ,y,u)  <= uvar

         ```math
         \begin{aligned}
         \min_{κ,y,u} \ & ∫_Ω​ f(κ,y,u) dΩ​ \\
         \mbox{ s.t. } & y \mbox{ solution of } PDE(κ,u)=0, \\
         & lcon <= c(κ,y,u) <= ucon, \\
         & lvar <= (κ,y,u)  <= uvar.
         \end{aligned}
         ```

The weak formulation is then:
res((y,u),(v,q)) = ∫ v PDE(κ,y,u) + ∫ q c(κ,y,u)

where the unknown (y,u) is a MultiField see [Tutorials 7](https://gridap.github.io/Tutorials/stable/pages/t007_darcy/)
 and [8](https://gridap.github.io/Tutorials/stable/pages/t008_inc_navier_stokes/) of Gridap.

The set Ω​ is represented here with *trian* and *quad*.

Main constructor:

`GridapPDENLPModel(:: NLPModelMeta, :: Counters, :: AbstractEnergyTerm, :: FESpace, :: Union{FESpace,Nothing}, :: FESpace, :: Union{FESpace,Nothing}, :: FESpace, :: Union{FESpace,Nothing}, :: Union{FEOperator, Nothing}, :: Int, :: Int, :: Int)` 

The following keyword arguments are available to all constructors:
- `name`: The name of the model (default: "Generic")
The following keyword arguments are available to the constructors for
constrained problems:
- `lin`: An array of indexes of the linear constraints
(default: `Int[]` or 1:ncon if c is an AffineFEOperator)

The following keyword arguments are available to the constructors for
constrained problems explictly giving lcon and ucon:
- `y0`: An inital estimate to the Lagrangian multipliers (default: zeros)

Notes:
 - We handle two types of FEOperator: AffineFEOperator, and FEOperatorFromWeakForm
 - If lcon and ucon are not given, they are assumed zeros.
 - If the type can't be deduced from the argument, it is Float64.
"""
mutable struct GridapPDENLPModel{T, S, NRJ <: AbstractEnergyTerm} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}

  counters::Counters

  # For the objective function
  tnrj::NRJ

  #Gridap discretization
  Ypde::FESpace #TrialFESpace for the solution of the PDE
  Ycon::FESpace #TrialFESpace for the parameter
  Xpde::FESpace #TestFESpace for the solution of the PDE
  Xcon::FESpace #TestFESpace for the parameter

  Y::FESpace #concatenated TrialFESpace
  X::FESpace #concatenated TestFESpace

  op::Union{FEOperator, Nothing}

  nvar_pde::Int #number of dofs in the solution functions
  nvar_con::Int #number of dofs in the control functions
  nparam::Int
  nnzh_obj::Int

  # store the structure for hessian and jacobian matrix
  Hrows
  Hcols
  Jrows
  Jcols
end

include("bounds_function.jl")
include("additional_constructors.jl")

NLPModels.show_header(io::IO, nlp::GridapPDENLPModel) = println(io, "GridapPDENLPModel")

function obj(nlp::GridapPDENLPModel, x::AbstractVector)
  @lencheck nlp.meta.nvar x
  increment!(nlp, :neval_obj)

  κ, xyu = x[1:(nlp.nparam)], x[(nlp.nparam + 1):(nlp.meta.nvar)]
  yu = FEFunction(nlp.Y, xyu)
  int = _obj_integral(nlp.tnrj, κ, yu)

  return sum(int)
end

function grad!(nlp::GridapPDENLPModel, x::AbstractVector, g::AbstractVector)
  @lencheck nlp.meta.nvar x g
  increment!(nlp, :neval_grad)

  κ, xyu = x[1:(nlp.nparam)], x[(nlp.nparam + 1):(nlp.meta.nvar)]
  yu = FEFunction(nlp.Y, xyu)

  return _compute_gradient!(g, nlp.tnrj, κ, yu, nlp.Y, nlp.X)
end

function hess_coord!(
  nlp::GridapPDENLPModel,
  x::AbstractVector{T},
  vals::AbstractVector;
  obj_weight::Real = one(T),
) where {T}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnzh vals
  increment!(nlp, :neval_hess)

  κ, xyu = x[1:(nlp.nparam)], x[(nlp.nparam + 1):(nlp.meta.nvar)]
  yu = FEFunction(nlp.Y, xyu)

  nnz_hess_k = get_nnz_hess_k(nlp.tnrj, nlp.meta.nvar, nlp.nparam)
  vals[1:nnz_hess_k] .= _compute_hess_k_vals(nlp, nlp.tnrj, κ, xyu) # do in-place
  nini = nnz_hess_k

  if typeof(nlp.tnrj) != NoFETerm
    if nlp.nparam > 0
      luh = nlp.tnrj.f(κ, yu)
      lag_hess = Gridap.FESpaces._hessian(x -> nlp.tnrj.f(κ, x), yu, luh)
    else
      luh = nlp.tnrj.f(yu)
      lag_hess = Gridap.FESpaces._hessian(nlp.tnrj.f, yu, luh)
    end
    matdata = Gridap.FESpaces.collect_cell_matrix(lag_hess)
    assem = SparseMatrixAssembler(nlp.Y, nlp.X)
    nini = fill_hess_coo_numeric!(vals, assem, matdata, n=nini)
  end

  if nini < nlp.meta.nnzh
    vals[(nini + 1):end] .= zero(T)
  end

  vals .*= obj_weight
  return vals
end

function hprod!(
  nlp::GridapPDENLPModel,
  x::AbstractVector{T},
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight::Real = one(T),
) where {T}
  @lencheck nlp.meta.nvar x v Hv
  increment!(nlp, :neval_hprod)

  if obj_weight == 0
    Hv .= zero(T)
    return Hv
  end

  rows, cols = hess_structure(nlp)
  vals = hess_coord(nlp, x, obj_weight = obj_weight) # store in a workspace in the model
  decrement!(nlp, :neval_hess)
  coo_sym_prod!(cols, rows, vals, v, Hv)

  return Hv
end

function hess_structure(nlp::GridapPDENLPModel)
  return (nlp.Hrows, nlp.Hcols)
end

function hess_structure!(
  nlp::GridapPDENLPModel,
  rows::AbstractVector{T},
  cols::AbstractVector{T},
) where {T <: Integer}
  @lencheck nlp.meta.nnzh rows cols

  rows .= T.(nlp.Hrows)
  cols .= T.(nlp.Hcols)

  (rows, cols)
end

function hess_coord!(
  nlp::GridapPDENLPModel,
  x::AbstractVector{T},
  λ::AbstractVector,
  vals::AbstractVector;
  obj_weight::Real = one(T),
) where {T}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon λ
  @lencheck nlp.meta.nnzh vals
  increment!(nlp, :neval_hess)

  hess_coord!(nlp, x, vals, obj_weight = obj_weight) # @view vals[1:nnzh_obj]
  decrement!(nlp, :neval_hess)
  nini = nlp.nnzh_obj

  p, n = nlp.nparam, nlp.meta.nvar
  κ, xyu = x[1:p], x[(p + 1):n]

  if p > 0
    nnz_hess_k = Int(p * (p + 1) / 2) + (n - p) * p
    function ℓ(x, λ)
      c = similar(x, nlp.meta.ncon)
      _from_terms_to_residual!(nlp.op, x, nlp.nparam, nlp.Y, nlp.Ypde, nlp.Ycon, c)
      return dot(c, λ)
    end
    agrad = @closure k -> ForwardDiff.gradient(x -> ℓ(x, λ), vcat(k, xyu))
    Hk = ForwardDiff.jacobian(agrad, κ)
    vals[(nini + 1):(nini + nnz_hess_k)] .= [Hk[i, j] for i = 1:n, j = 1:p if j ≤ i]
    nini += nnz_hess_k
  end

  if typeof(nlp.op) <: Gridap.FESpaces.FEOperatorFromWeakForm
    # λ = zeros(Gridap.FESpaces.num_free_dofs(nlp.Ypde))
    λf = FEFunction(nlp.Xpde, λ) # or Ypde
    xh = FEFunction(nlp.Y, x)

    function split_res(x, λ)
      if typeof(nlp.Ycon) <: VoidFESpace
        if nlp.nparam > 0
          return nlp.op.res(κ, x, λ)
        else
          return nlp.op.res(x, λ)
        end
      else
        y, u = _split_FEFunction(x, nlp.Ypde, nlp.Ycon)
        if nlp.nparam > 0
          return nlp.op.res(κ, y, u, λ)
        else
          return nlp.op.res(y, u, λ)
        end
      end
    end
    luh = split_res(xh, λf)

    lag_hess = Gridap.FESpaces._hessian(x -> split_res(x, λf), xh, luh)
    matdata = Gridap.FESpaces.collect_cell_matrix(lag_hess)
    assem = SparseMatrixAssembler(nlp.Y, nlp.X)
    fill_hess_coo_numeric!(vals, assem, matdata, n=nini)
  end

  return vals
end

function hprod!(
  nlp::GridapPDENLPModel,
  x::AbstractVector{T},
  λ::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight::Real = one(T),
) where {T}
  @lencheck nlp.meta.nvar x v Hv
  @lencheck nlp.meta.ncon λ
  increment!(nlp, :neval_hprod)

  rows, cols = hess_structure(nlp)
  vals = hess_coord(nlp, x, λ, obj_weight = obj_weight)
  decrement!(nlp, :neval_hess)
  coo_sym_prod!(cols, rows, vals, v, Hv)

  return Hv
end

function cons!(nlp::GridapPDENLPModel, x::AbstractVector, c::AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon c
  increment!(nlp, :neval_cons)
  _from_terms_to_residual!(nlp.op, x, nlp.nparam, nlp.Y, nlp.Ypde, nlp.Ycon, c)
  return c
end

function jprod!(nlp::GridapPDENLPModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck nlp.meta.nvar x v
  @lencheck nlp.meta.ncon Jv
  increment!(nlp, :neval_jprod)
  rows, cols = jac_structure(nlp)
  vals = jac_coord(nlp, x)
  decrement!(nlp, :neval_jac)
  coo_prod!(rows, cols, vals, v, Jv)
  return Jv
end

function jtprod!(nlp::GridapPDENLPModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  @lencheck nlp.meta.nvar x Jtv
  @lencheck nlp.meta.ncon v
  increment!(nlp, :neval_jtprod)
  rows, cols = jac_structure(nlp)
  vals = jac_coord(nlp, x)
  decrement!(nlp, :neval_jac)
  coo_prod!(cols, rows, vals, v, Jtv)
  return Jtv
end

function jac_structure(nlp::GridapPDENLPModel)
  return (nlp.Jrows, nlp.Jcols)
end

function jac_structure!(
  nlp::GridapPDENLPModel,
  rows::AbstractVector{T},
  cols::AbstractVector{T},
) where {T<:Integer}
  @lencheck nlp.meta.nnzj rows cols
  rows .= T.(nlp.Jrows)
  cols .= T.(nlp.Jcols)
  return rows, cols
end

function jac_coord!(nlp::GridapPDENLPModel, x::AbstractVector, vals::AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnzj vals
  increment!(nlp, :neval_jac)
  return _jac_coord!(nlp.op, nlp.nparam, nlp.meta.ncon, nlp.Y, nlp.Ypde, nlp.Xpde, nlp.Ycon, x, vals)
end
