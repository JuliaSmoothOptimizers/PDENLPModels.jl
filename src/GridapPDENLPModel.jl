"""
    PDENLPMeta
A composite type that represents the main features of the PDE-constrained optimization problem

---
`PDENLPMeta` contains the following attributes:
- `tnrj` : structure representing the objective term
- `Ypde` : TrialFESpace for the solution of the PDE
- `Ycon` : TrialFESpace for the parameter
- `Xpde` : TestFESpace for the solution of the PDE
- `Xcon` : TestFESpace for the parameter
- `Y` : concatenated TrialFESpace
- `X` : concatenated TestFESpace
- `op` : operator representing the PDE-constraint (nothing if no constraints)
- `nvar_pde` :number of dofs in the solution functions
- `nvar_con` : number of dofs in the control functions
- `nparam` : number of real unknowns
- `nnzh_obj` : number of nonzeros elements in the objective hessian
- `Hrows` : store the structure for the hessian of the lagrangian
- `Hcols` : store the structure for the hessian of the lagrangian
- `Jrows` : store the structure for the hessian of the jacobian
- `Jcols` : store the structure for the hessian of the jacobian
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
  Jrows::AbstractVector{Int}
  Jcols::AbstractVector{Int}
end

"""
    PDEWorkspace

Pre-allocated memory for `GridapPDENLPModel`.
"""
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
`GridapPDENLPModel` returns an instance of an [`AbstractNLPModel`](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) using [Gridap.jl](https://github.com/gridap/Gridap.jl) for the discretization of the domain with finite-elements.
Given a domain `Ω ⊂ ℜᵈ` Find a state function y: `Ω -> Y`, a control function u: `Ω -> U` and an algebraic vector κ ∈ ℜⁿ satisfying
```math
\begin{aligned}
\min_{κ,y,u} \ & ∫_Ω​ f(κ,y,u) dΩ​ \\
\mbox{ s.t. } & y \mbox{ solution of } PDE(κ,u)=0, \\
& lcon <= c(κ,y,u) <= ucon, \\
& lvar <= (κ,y,u)  <= uvar.
\end{aligned}
```

The [weak formulation of the PDE](https://en.wikipedia.org/wiki/Weak_formulation) is then:
`res((y,u),(v,q)) = ∫ v PDE(κ,y,u) + ∫ q c(κ,y,u)`

where the unknown `(y,u)` is a `MultiField` see [Tutorials 7](https://gridap.github.io/Tutorials/stable/pages/t007_darcy/)
 and [8](https://gridap.github.io/Tutorials/stable/pages/t008_inc_navier_stokes/) of Gridap.

## Constructors

Main constructor:

    GridapPDENLPModel(::NLPModelMeta, ::Counters, ::PDENLPMeta, ::PDEWorkspace)

This is the main constructors with the attributes of the `GridapPDENLPModel`:
- `meta::NLPModelMeta`: usual `meta` for NLPModels, see [doc here](https://juliasmoothoptimizers.github.io/NLPModels.jl/dev/models/);
- `counters::Counters`: usual `counters` for NLPModels, see [doc here](https://juliasmoothoptimizers.github.io/NLPModels.jl/dev/tools/);
- `pdemeta::PDENLPMeta`: metadata specific to `GridapPDENLPModel`;
- `workspace::PDEWorkspace`: Pre-allocated memory for `GridapPDENLPModel`.

More practical constructors are also available.

- For unconstrained problems:

    GridapPDENLPModel(x0, ::Function, ::Union{Measure, Triangulation}, Ypde::FESpace, Xpde::FESpace; kwargs...)

    GridapPDENLPModel(x0, ::AbstractEnergyTerm, ::Union{Measure, Triangulation}, Ypde::FESpace, Xpde::FESpace; kwargs...)

- For constrained problems without controls:

    GridapPDENLPModel(x0, ::Function, ::Union{Measure, Triangulation}, Ypde::FESpace, Xpde::FESpace, c::Union{Function, FEOperator}; kwargs...)

    GridapPDENLPModel(x0, ::AbstractEnergyTerm, Ypde::FESpace, Xpde::FESpace, c::Union{Function, FEOperator}; kwargs...)

- For general constrained problems:

    GridapPDENLPModel(x0, ::Function, ::Union{Measure, Triangulation}, Ypde::FESpace, Ycon::FESpace, Xpde::FESpace, Xcon::FESpace, c::Union{Function, FEOperator}; kwargs...)

    GridapPDENLPModel(x0, ::AbstractEnergyTerm, Ypde::FESpace, Ycon::FESpace, Xpde::FESpace, Xcon::FESpace, c::Union{Function, FEOperator}; kwargs...)

where the different arguments are:
- `x0`: initial guess for the system of size `≥ num_free_dofs(Ypde) + num_free_dofs(Ycon)`;
- `f`: objective function, the number of arguments depend on the application `(y)` or `(y,u)` or `(y,u,θ)`;
- `Ypde`: trial space for the state;
- `Ycon`: trial space for the control (`VoidFESpace` if none);
- `Xpde`: test space for the state;
- `Xcon`: test space for the control (`VoidFESpace` if none);
- `c`: operator/function for the PDE-constraint, were we assume by default that the right-hand side is zero (otw. use `lcon` and `ucon` keywords), the number of arguments depend on the application `(y,v)` or `(y,u,v)` or `(y,u,θ,v)`.

If `length(x0) > num_free_dofs(Ypde) + num_free_dofs(Ycon)`, then the additional components are considered algebraic variables.

The function `f` and `c` must return integrals complying with Gridap's functions with a `Measure/Triangulation` given in the arguments of `GridapPDENLPModel`.
Internally, the objective function `f` and the `Measure/Triangulation` are combined to instantiate an `AbstractEnergyTerm`.

The following keyword arguments are available to all constructors:
- `name`: The name of the model (default: "Generic")
The following keyword arguments are available to the constructors for
constrained problems:
- `lin`: An array of indexes of the linear constraints (default: `Int[]`)

The following keyword arguments are available to the constructors for
constrained problems explictly giving lcon and ucon:
- `y0`: An inital estimate to the Lagrangian multipliers (default: zeros)

The bounds on the variables are given as `AbstractVector` via keywords arguments as well:
- either with `lvar` and `uvar`, or,
- `lvary`, `lvaru`, `lvark`, and `uvary`, `uvaru`, `uvark`.

Notes:
 - We handle two types of `FEOperator`: `AffineFEOperator`, and `FEOperatorFromWeakForm`.
 - If `lcon` and `ucon` are not given, they are assumed zeros.
 - If the type can't be deduced from the argument, it is `Float64`.

## Example

```julia
using Gridap, PDENLPModels

  # Definition of the domain
  n = 100
  domain = (-1, 1, -1, 1)
  partition = (n, n)
  model = CartesianDiscreteModel(domain, partition)

  # Definition of the spaces:
  valuetype = Float64
  reffe = ReferenceFE(lagrangian, valuetype, 2)
  Xpde = TestFESpace(model, reffe; conformity = :H1, dirichlet_tags = "boundary")
  y0(x) = 0.0
  Ypde = TrialFESpace(Xpde, y0)

  reffe_con = ReferenceFE(lagrangian, valuetype, 1)
  Xcon = TestFESpace(model, reffe_con; conformity = :H1)
  Ycon = TrialFESpace(Xcon)

  # Integration machinery
  trian = Triangulation(model)
  degree = 1
  dΩ = Measure(trian, degree)

  # Objective function:
  yd(x) = -x[1]^2
  α = 1e-2
  function f(y, u)
    ∫(0.5 * (yd - y) * (yd - y) + 0.5 * α * u * u) * dΩ
  end

  # Definition of the constraint operator
  ω = π - 1 / 8
  h(x) = -sin(ω * x[1]) * sin(ω * x[2])
  function res(y, u, v)
    ∫(∇(v) ⊙ ∇(y) - v * u - v * h) * dΩ
  end

  # initial guess
  npde = num_free_dofs(Ypde)
  ncon = num_free_dofs(Ycon)
  xin = zeros(npde + ncon)

  nlp = GridapPDENLPModel(xin, f, trian, Ypde, Ycon, Xpde, Xcon, res, name = "Control elastic membrane")
```

You can also check the tutorial [Solve a PDE-constrained optimization problem](https://jso-docs.github.io/solve-pdenlpmodels-with-jsosolvers/) on JSO's website, [juliasmoothoptimizers.github.io](https://juliasmoothoptimizers.github.io).

We refer to the folder `test/problems` for more examples of problems of different types:
  - calculus of variations,
  - optimal control problem,
  - PDE-constrained problems,
  - mixed PDE-contrained problems with both function and algebraic unknowns. 
An alternative is to visit the repository [PDEOptimizationProblems](https://github.com/tmigot/PDEOptimizationProblems) that contains a collection of test problems.

Without objective function, the problem reduces to a classical PDE and we refer to [Gridap tutorials](https://github.com/gridap/Tutorials) for examples.

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
    Return the value `$($(QuoteNode(field)))` from meta or nlp.meta.
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

    matdata = Gridap.FESpaces.collect_cell_matrix(lag_hess)
    assem = SparseMatrixAssembler(nlp.pdemeta.Y, nlp.pdemeta.X)
    nini = fill_hess_coo_numeric!(vals, assem, matdata, n = nini)
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

    lag_hess = Gridap.FESpaces._hessian(x -> split_res(x, λf), xh, luh)
    matdata = Gridap.FESpaces.collect_cell_matrix(lag_hess)
    assem = SparseMatrixAssembler(nlp.pdemeta.Y, nlp.pdemeta.X)
    fill_hess_coo_numeric!(vals, assem, matdata, n = nini)
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

function cons_nln!(nlp::GridapPDENLPModel, x::AbstractVector, c::AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnln c
  increment!(nlp, :neval_cons_nln)
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

function cons_nln!(
  nlp::GridapPDENLPModel{T, S, NRJ, Union{Nothing, AffineFEOperator}},
  x::AbstractVector,
  c::AbstractVector,
) where {T, S, NRJ}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnln c
  increment!(nlp, :neval_cons_nln)
  return c
end

function cons_lin!(
  nlp::GridapPDENLPModel{T, S, NRJ, AffineFEOperator},
  x::AbstractVector,
  c::AbstractVector,
) where {T, S, NRJ}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nlin c
  increment!(nlp, :neval_cons_lin)
  op = nlp.pdemeta.op
  mul!(c, get_matrix(op), x)
  axpy!(-one(T), get_vector(op), c)
end

function jprod_nln!(
  nlp::GridapPDENLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck nlp.meta.nvar x v
  @lencheck nlp.meta.nnln Jv
  increment!(nlp, :neval_jprod_nln)
  rows, cols = jac_nln_structure(nlp)
  vals = jac_nln_coord!(nlp, x, nlp.workspace.Jvals)
  decrement!(nlp, :neval_jac_nln)
  coo_prod!(rows, cols, vals, v, Jv)
  return Jv
end

function jprod_lin!(
  nlp::GridapPDENLPModel{T, S, NRJ, AffineFEOperator},
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
) where {T, S, NRJ}
  @lencheck nlp.meta.nvar x v
  @lencheck nlp.meta.nlin Jv
  increment!(nlp, :neval_jprod_lin)
  rows, cols = jac_lin_structure(nlp)
  vals = jac_lin_coord!(nlp, x, nlp.workspace.Jvals)
  decrement!(nlp, :neval_jac_lin)
  coo_prod!(rows, cols, vals, v, Jv)
  return Jv
end

function jtprod_nln!(
  nlp::GridapPDENLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck nlp.meta.nvar x Jtv
  @lencheck nlp.meta.nnln v
  increment!(nlp, :neval_jtprod_nln)
  rows, cols = jac_nln_structure(nlp)
  vals = jac_nln_coord!(nlp, x, nlp.workspace.Jvals)
  decrement!(nlp, :neval_jac_nln)
  coo_prod!(cols, rows, vals, v, Jtv)
  return Jtv
end

function jtprod_lin!(
  nlp::GridapPDENLPModel{T, S, NRJ, AffineFEOperator},
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
) where {T, S, NRJ}
  @lencheck nlp.meta.nvar x Jtv
  @lencheck nlp.meta.nlin v
  increment!(nlp, :neval_jtprod_lin)
  rows, cols = jac_lin_structure(nlp)
  vals = jac_lin_coord!(nlp, x, nlp.workspace.Jvals)
  decrement!(nlp, :neval_jac_lin)
  coo_prod!(cols, rows, vals, v, Jtv)
  return Jtv
end

function jac_nln_structure(nlp::GridapPDENLPModel)
  return (nlp.pdemeta.Jrows, nlp.pdemeta.Jcols)
end

function jac_nln_structure!(
  nlp::GridapPDENLPModel,
  rows::AbstractVector{T},
  cols::AbstractVector{T},
) where {T <: Integer}
  @lencheck nlp.meta.nln_nnzj rows cols
  rows .= T.(nlp.pdemeta.Jrows)
  cols .= T.(nlp.pdemeta.Jcols)
  return rows, cols
end

function jac_lin_structure(nlp::GridapPDENLPModel{T, S, NRJ, AffineFEOperator}) where {T, S, NRJ}
  return (nlp.pdemeta.Jrows, nlp.pdemeta.Jcols)
end

function jac_lin_structure!(
  nlp::GridapPDENLPModel{T, S, NRJ, AffineFEOperator},
  rows::AbstractVector{It},
  cols::AbstractVector{It},
) where {It <: Integer, T, S, NRJ}
  @lencheck nlp.meta.lin_nnzj rows cols
  rows .= It.(nlp.pdemeta.Jrows)
  cols .= It.(nlp.pdemeta.Jcols)
  return rows, cols
end

function jac_nln_coord!(nlp::GridapPDENLPModel, x::AbstractVector, vals::AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nln_nnzj vals
  increment!(nlp, :neval_jac_nln)
  return _jac_coord!(
    nlp.pdemeta.op,
    nlp.pdemeta.nparam,
    nlp.meta.nnln,
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

function jac_nln_coord!(
  nlp::GridapPDENLPModel{T, S, NRJ, Union{Nothing, AffineFEOperator}},
  x::AbstractVector,
  vals::AbstractVector,
) where {T, S, NRJ}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nln_nnzj vals
  increment!(nlp, :neval_jac_nln)
  return vals
end

function jac_lin_coord!(
  nlp::GridapPDENLPModel{T, S, NRJ, AffineFEOperator},
  x::AbstractVector,
  vals::AbstractVector,
) where {T, S, NRJ}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.lin_nnzj vals
  increment!(nlp, :neval_jac_lin)
  _, _, V = findnz(get_matrix(nlp.pdemeta.op))
  vals .= V
  return vals
end

function jac_nln_op!(
  nlp::GridapPDENLPModel,
  x::AbstractVector,
  Jv::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck nlp.meta.nvar x Jtv
  @lencheck nlp.meta.ncon Jv
  rows = nlp.pdemeta.Jrows
  cols = nlp.pdemeta.Jcols
  vals = jac_nln_coord!(nlp, x, nlp.workspace.Jvals)
  return jac_nln_op!(nlp, rows, cols, vals, Jv, Jtv)
end

function jac_op!(nlp::GridapPDENLPModel, x::AbstractVector, Jv::AbstractVector, Jtv::AbstractVector)
  @lencheck nlp.meta.nvar x Jtv
  @lencheck nlp.meta.ncon Jv
  rows = nlp.pdemeta.Jrows
  cols = nlp.pdemeta.Jcols
  if nlp.meta.nnln > 0 && nlp.meta.nlin > 0
    @warn "PDENLPModels do not handle mixed linear and nonlinear constraints (yet)."
  end
  vals = jac_coord!(nlp, x, nlp.workspace.Jvals)
  return jac_op!(nlp, rows, cols, vals, Jv, Jtv)
end

function hess_op!(
  nlp::GridapPDENLPModel,
  x::AbstractVector,
  Hv::AbstractVector;
  obj_weight::Real = one(eltype(x)),
)
  @lencheck nlp.meta.nvar x Hv
  rows = nlp.pdemeta.Hrows
  cols = nlp.pdemeta.Hcols
  vals = hess_coord!(nlp, x, nlp.workspace.Hvals, obj_weight = obj_weight)
  return hess_op!(nlp, rows, cols, vals, Hv)
end

function hess_op!(
  nlp::GridapPDENLPModel,
  x::AbstractVector,
  y::AbstractVector,
  Hv::AbstractVector;
  obj_weight::Real = one(eltype(x)),
)
  @lencheck nlp.meta.nvar x Hv
  rows = nlp.pdemeta.Hrows
  cols = nlp.pdemeta.Hcols
  vals = hess_coord!(nlp, x, y, nlp.workspace.Hvals, obj_weight = obj_weight)
  return hess_op!(nlp, rows, cols, vals, Hv)
end
