"""
    PDENLPMeta
A composite type that represents the main features of the PDE-constrained optimization problem

---
The following arguments are accepted:
- tnrj : structure representing the objective term
- Ypde : TrialFESpace for the solution of the PDE
- Ycon : TrialFESpace for the parameter
- Xpde : TestFESpace for the solution of the PDE
- Xcon : TestFESpace for the parameter
- Y : concatenated TrialFESpace
- X : concatenated TestFESpace
- op : operator representing the PDE-constraint (nothing if no constraints)
- nvar_pde :number of dofs in the solution functions
- nvar_con : number of dofs in the control functions
- nparam: : number of real unknowns
- nnzh_obj : number of nonzeros elements in the objective hessian
- Hrows : store the structure for the hessian of the lagrangian
- Hcols : store the structure for the hessian of the lagrangian
- Jkrows : store the structure for the hessian of the jacobian
- Jkcols : store the structure for the hessian of the jacobian
- Jyrows : store the structure for the hessian of the jacobian
- Jycols : store the structure for the hessian of the jacobian
- Jurows : store the structure for the hessian of the jacobian
- Jucols : store the structure for the hessian of the jacobian
"""
struct PDENLPMeta{NRJ <: AbstractEnergyTerm, Op <: Union{FEOperator, Nothing}}
  # For the objective function
  tnrj::NRJ

  #Gridap discretization
  Ypde::FESpace #TrialFESpace for the solution of the PDE
  Ycon::FESpace #TrialFESpace for the parameter
  Xpde::FESpace #TestFESpace for the solution of the PDE
  Xcon::FESpace #TestFESpace for the parameter

  Y::FESpace #concatenated TrialFESpace
  X::FESpace #concatenated TestFESpace

  op::Op

  nvar_pde::Int #number of dofs in the solution functions
  nvar_con::Int #number of dofs in the control functions
  nparam::Int
  nnzh_obj::Int

  # store the structure for hessian and jacobian matrix
  Hrows::AbstractVector{Int}
  Hcols::AbstractVector{Int}
  Jkrows::AbstractVector{Int}
  Jkcols::AbstractVector{Int}
  Jyrows::AbstractVector{Int}
  Jycols::AbstractVector{Int}
  Jurows::AbstractVector{Int}
  Jucols::AbstractVector{Int}
end

struct PDEWorkspace{T, S, M}
  Hvals::S # vector of values of the Hessian matrix
  Jvals::S # vector of values of the Jacobian matrix
  g::S
  Hk::M
  c::S
  Jk::M
end

function PDEWorkspace(T, S, nvar, ncon, nparam, nnzh, nnzj)
  Hvals = S(undef, nnzh)
  Jvals = S(undef, nnzj)
  g = S(undef, nvar)
  Hk = Array{T, 2}(undef, nvar, nparam)
  c = S(undef, ncon)
  Jk = Array{T, 2}(undef, ncon, nparam)
  return PDEWorkspace{T, S, Array{T, 2}}(Hvals, Jvals, g, Hk, c, Jk)
end

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
mutable struct GridapPDENLPModel{T, S, NRJ, Op} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
  pdemeta::PDENLPMeta{NRJ, Op}
  workspace::PDEWorkspace
end

for field in fieldnames(PDENLPMeta)
  meth = Symbol("get_", field)
  @eval begin
    @doc """
        $($meth)(nlp)
        $($meth)(meta)
    Return the value $($(QuoteNode(field))) from meta or nlp.meta.
    """
    $meth(meta::PDENLPMeta) = getproperty(meta, $(QuoteNode(field)))
  end
  @eval $meth(nlp::GridapPDENLPModel) = $meth(nlp.pdemeta)
  @eval export $meth
end

include("bounds_function.jl")
include("additional_constructors.jl")

NLPModels.show_header(io::IO, nlp::GridapPDENLPModel) = println(io, "GridapPDENLPModel")

function obj(nlp::GridapPDENLPModel, x::AbstractVector)
  @lencheck nlp.meta.nvar x
  increment!(nlp, :neval_obj)

  κ = @view x[1:(nlp.pdemeta.nparam)]
  xyu = @view x[(nlp.pdemeta.nparam + 1):(nlp.meta.nvar)]
  yu = FEFunction(nlp.pdemeta.Y, xyu)
  int = _obj_integral(nlp.pdemeta.tnrj, κ, yu)

  return sum(int)
end

function grad!(nlp::GridapPDENLPModel, x::AbstractVector, g::AbstractVector)
  @lencheck nlp.meta.nvar x g
  increment!(nlp, :neval_grad)

  κ = @view x[1:(nlp.pdemeta.nparam)]
  xyu = @view x[(nlp.pdemeta.nparam + 1):(nlp.meta.nvar)]
  yu = FEFunction(nlp.pdemeta.Y, xyu)

  return _compute_gradient!(g, nlp.pdemeta.tnrj, κ, yu, nlp.pdemeta.Y, nlp.pdemeta.X)
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

  κ = @view x[1:(nlp.pdemeta.nparam)]
  xyu = @view x[(nlp.pdemeta.nparam + 1):(nlp.meta.nvar)]
  yu = FEFunction(nlp.pdemeta.Y, xyu)

  nnz_hess_k = get_nnz_hess_k(nlp.pdemeta.tnrj, nlp.meta.nvar, nlp.pdemeta.nparam)
  _compute_hess_k_vals!(view(vals, 1:nnz_hess_k), nlp, nlp.pdemeta.tnrj, κ, xyu)
  nini = nnz_hess_k

  if typeof(nlp.pdemeta.tnrj) != NoFETerm
    luh = _obj_integral(nlp.pdemeta.tnrj, κ, yu)
    lag_hess = Gridap.FESpaces._hessian(x -> _obj_integral(nlp.pdemeta.tnrj, κ, x), yu, luh)

    matdata = Gridap.FESpaces.collect_cell_matrix(nlp.pdemeta.Y, nlp.pdemeta.X, lag_hess)
    assem = SparseMatrixAssembler(nlp.pdemeta.Y, nlp.pdemeta.X)
    ###############################################################
    # TO BE IMPROVED
    m1 = Gridap.FESpaces.nz_counter(
      Gridap.FESpaces.get_matrix_builder(assem),
      (Gridap.FESpaces.get_rows(assem), Gridap.FESpaces.get_cols(assem)),
    ) # Gridap.Algebra.CounterCS
    Gridap.FESpaces.symbolic_loop_matrix!(m1, assem, matdata)
    m2 = Gridap.FESpaces.nz_allocation(m1) # Gridap.Algebra.InserterCSC
    Gridap.FESpaces.numeric_loop_matrix!(m2, assem, matdata)
    m3 = sparse(LowerTriangular(Gridap.FESpaces.create_from_nz(m2)))
    _, _, v = findnz(m3)
    vals[(nini + 1):(nini + length(v))] .= v
    nini += length(v)
    # nini = fill_hess_coo_numeric!(vals, assem, matdata, n = nini)
    ##############################################################
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
  vals = hess_coord!(nlp, x, nlp.workspace.Hvals, obj_weight = obj_weight)
  decrement!(nlp, :neval_hess)
  coo_sym_prod!(cols, rows, vals, v, Hv)

  return Hv
end

function hess_structure(nlp::GridapPDENLPModel)
  return (nlp.pdemeta.Hrows, nlp.pdemeta.Hcols)
end

function hess_structure!(
  nlp::GridapPDENLPModel,
  rows::AbstractVector{T},
  cols::AbstractVector{T},
) where {T <: Integer}
  @lencheck nlp.meta.nnzh rows cols
  rows .= T.(nlp.pdemeta.Hrows)
  cols .= T.(nlp.pdemeta.Hcols)
  return (rows, cols)
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

  hess_coord!(nlp, x, vals, obj_weight = obj_weight)
  decrement!(nlp, :neval_hess)
  nini = nlp.pdemeta.nnzh_obj

  p, n = nlp.pdemeta.nparam, nlp.meta.nvar
  κ = @view x[1:p]
  xyu = @view x[(p + 1):n]

  if p > 0
    nnz_hess_k = Int(p * (p + 1) / 2) + (n - p) * p
    function ℓ(x, λ)
      c = similar(x, nlp.meta.ncon)
      _from_terms_to_residual!(
        nlp.pdemeta.op,
        x,
        nlp.pdemeta.nparam,
        nlp.pdemeta.Y,
        nlp.pdemeta.Ypde,
        nlp.pdemeta.Ycon,
        c,
      )
      return dot(c, λ)
    end
    agrad = @closure (g, k) -> ForwardDiff.gradient!(g, x -> ℓ(x, λ), vcat(k, xyu))
    Hk = ForwardDiff.jacobian!(nlp.workspace.Hk, agrad, nlp.workspace.g, κ)
    vals[(nini + 1):(nini + nnz_hess_k)] .= [Hk[i, j] for i = 1:n, j = 1:p if j ≤ i]
    nini += nnz_hess_k
  end

  if typeof(nlp.pdemeta.op) <: Gridap.FESpaces.FEOperatorFromWeakForm
    λf = FEFunction(nlp.pdemeta.Xpde, λ)
    xh = FEFunction(nlp.pdemeta.Y, x)

    function split_res(x, λ)
      if typeof(nlp.pdemeta.Ycon) <: VoidFESpace
        if nlp.pdemeta.nparam > 0
          return nlp.pdemeta.op.res(κ, x, λ)
        else
          return nlp.pdemeta.op.res(x, λ)
        end
      else
        y, u = _split_FEFunction(x, nlp.pdemeta.Ypde, nlp.pdemeta.Ycon)
        if nlp.pdemeta.nparam > 0
          return nlp.pdemeta.op.res(κ, y, u, λ)
        else
          return nlp.pdemeta.op.res(y, u, λ)
        end
      end
    end
    luh = split_res(xh, λf)

    # lag_hess = Gridap.FESpaces.jacobian(Gridap.FESpaces._gradient(x -> split_res(x, λf), xh, luh), xh) # 
    # lag_hess = Gridap.FESpaces._hessian(x -> split_res(x, λf), xh, luh)
    lag_hess = _hessianv1(x -> split_res(x, λf), xh, luh)
    matdata = Gridap.FESpaces.collect_cell_matrix(nlp.pdemeta.Y, nlp.pdemeta.X, lag_hess)
    assem = SparseMatrixAssembler(nlp.pdemeta.Y, nlp.pdemeta.X)
    ##############################################################
    # TO BE IMPROVED
    m1 = Gridap.FESpaces.nz_counter(
      Gridap.FESpaces.get_matrix_builder(assem),
      (Gridap.FESpaces.get_rows(assem), Gridap.FESpaces.get_cols(assem)),
    ) # Gridap.Algebra.CounterCS
    Gridap.FESpaces.symbolic_loop_matrix!(m1, assem, matdata)
    m2 = Gridap.FESpaces.nz_allocation(m1) # Gridap.Algebra.InserterCSC
    Gridap.FESpaces.numeric_loop_matrix!(m2, assem, matdata)
    m3 = sparse(LowerTriangular(Gridap.FESpaces.create_from_nz(m2)))
    _, _, v = findnz(m3)
    vals[(nini + 1):(nini + length(v))] .= v
    nini += length(v)
    # nini = fill_hess_coo_numeric!(vals, assem, matdata, n = nini)
    ##############################################################
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
  vals = hess_coord!(nlp, x, λ, nlp.workspace.Hvals, obj_weight = obj_weight)
  decrement!(nlp, :neval_hess)
  coo_sym_prod!(cols, rows, vals, v, Hv)

  return Hv
end

function cons!(nlp::GridapPDENLPModel, x::AbstractVector, c::AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon c
  increment!(nlp, :neval_cons)
  _from_terms_to_residual!(
    nlp.pdemeta.op,
    x,
    nlp.pdemeta.nparam,
    nlp.pdemeta.Y,
    nlp.pdemeta.Ypde,
    nlp.pdemeta.Ycon,
    c,
  )
  return c
end

function cons!(
  nlp::GridapPDENLPModel{T, S, NRJ, Nothing},
  x::AbstractVector,
  c::AbstractVector,
) where {T, S, NRJ}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon c
  increment!(nlp, :neval_cons)
  return c
end

function jprod!(nlp::GridapPDENLPModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck nlp.meta.nvar x v
  @lencheck nlp.meta.ncon Jv
  increment!(nlp, :neval_jprod)
  rows, cols = jac_structure(nlp)
  vals = jac_coord!(nlp, x, nlp.workspace.Jvals)
  decrement!(nlp, :neval_jac)
  coo_prod!(rows, cols, vals, v, Jv)
  return Jv
end

function jtprod!(nlp::GridapPDENLPModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  @lencheck nlp.meta.nvar x Jtv
  @lencheck nlp.meta.ncon v
  increment!(nlp, :neval_jtprod)
  rows, cols = jac_structure(nlp)
  vals = jac_coord!(nlp, x, nlp.workspace.Jvals)
  decrement!(nlp, :neval_jac)
  coo_prod!(cols, rows, vals, v, Jtv)
  return Jtv
end

function jac_structure(nlp::GridapPDENLPModel)
  nparam = nlp.pdemeta.nparam
  ny = num_free_dofs(nlp.pdemeta.Ypde)
  return (vcat(nlp.pdemeta.Jkrows, nlp.pdemeta.Jyrows, nlp.pdemeta.Jurows), vcat(nlp.pdemeta.Jkcols, nlp.pdemeta.Jycols .+ nparam, nlp.pdemeta.Jucols .+ nparam .+ ny))
end

function jac_structure!(
  nlp::GridapPDENLPModel,
  rows::AbstractVector{T},
  cols::AbstractVector{T},
) where {T <: Integer}
  @lencheck nlp.meta.nnzj rows cols
  nparam = nlp.pdemeta.nparam
  ny = num_free_dofs(nlp.pdemeta.Ypde)
  rows .= T.(vcat(nlp.pdemeta.Jkrows, nlp.pdemeta.Jyrows, nlp.pdemeta.Jurows))
  cols .= T.(vcat(nlp.pdemeta.Jkcols, nlp.pdemeta.Jycols .+ nparam, nlp.pdemeta.Jucols .+ nparam .+ ny))
  return rows, cols
end

function jac_coord!(nlp::GridapPDENLPModel, x::AbstractVector, vals::AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnzj vals
  increment!(nlp, :neval_jac)
  return _jac_coord!(
    nlp.pdemeta.op,
    nlp.pdemeta.nparam,
    nlp.meta.ncon,
    nlp.pdemeta.Y,
    nlp.pdemeta.Ypde,
    nlp.pdemeta.Xpde,
    nlp.pdemeta.Ycon,
    x,
    vals,
    nlp.workspace.c,
    nlp.workspace.Jk,
  )
end

function jac_coord!(
  nlp::GridapPDENLPModel{T, S, NRJ, Nothing},
  x::AbstractVector,
  vals::AbstractVector,
) where {T, S, NRJ}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnzj vals
  increment!(nlp, :neval_jac)
  return vals
end
