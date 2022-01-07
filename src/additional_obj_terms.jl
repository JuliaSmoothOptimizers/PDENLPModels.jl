abstract type AbstractEnergyTerm end

"""
Return the integral of the objective function

`_obj_integral(:: AbstractEnergyTerm, :: FEFunctionType, :: AbstractVector)`

See also: `MixedEnergyFETerm`, `EnergyFETerm`, `NoFETerm`,
`_obj_cell_integral`, `_compute_gradient_k`,
`_compute_hess_k_coo`
"""
function _obj_integral end

"""
Return the derivative of the objective function w.r.t. κ.

`_compute_gradient_k!(::AbstractVector, :: AbstractEnergyTerm, :: FEFunctionType, :: AbstractVector)`

See also: `MixedEnergyFETerm`, `EnergyFETerm`, `NoFETerm`, `_obj_integral`,
`_obj_cell_integral`, `_compute_hess_k_coo`
"""
function _compute_gradient_k! end

"""
Return the gradient of the objective function and set it in place.

`_compute_gradient!(:: AbstractVector, :: EnergyFETerm, :: AbstractVector, :: FEFunctionType, :: FESpace, :: FESpace)`

See also: `MixedEnergyFETerm`, `EnergyFETerm`, `NoFETerm`, `_obj_integral`,
`_obj_cell_integral`, `_compute_hess_k_coo`
"""
function _compute_gradient! end

"""
Return the values of the hessian w.r.t. κ of the objective function.

`_compute_hess_k_vals!(:: AbstractVector, :: AbstractNLPModel, :: AbstractEnergyTerm, :: AbstractVector, :: AbstractVector)`

See also: `MixedEnergyFETerm`, `EnergyFETerm`, `NoFETerm`, `_obj_integral`,
`_obj_cell_integral`, `_compute_gradient_k`
"""
function _compute_hess_k_vals! end

@doc raw"""
FETerm modeling the objective function when there are no integral objective.

```math
\begin{aligned}
 f(\kappa)
\end{aligned}
 ```

Constructors:

  `NoFETerm()`

  `NoFETerm(:: Function)`

See also: `MixedEnergyFETerm`, `EnergyFETerm`, `_obj_cell_integral`, `_obj_integral`, `_compute_gradient_k!`
"""
struct NoFETerm <: AbstractEnergyTerm
  f::Function
end

function NoFETerm()
  return NoFETerm(x -> 0.0)
end

_obj_integral(tnrj::NoFETerm, κ::AbstractVector, x) = tnrj.f(κ)

function _compute_gradient!(
  g::AbstractVector{T},
  tnrj::NoFETerm,
  κ::AbstractVector,
  yu::FEFunctionType,
  Y::FESpace,
  X::FESpace,
) where {T}
  nparam = length(κ)
  nyu = num_free_dofs(Y)
  nvar = nparam + nyu
  @lencheck nvar g

  g[(nparam + 1):nvar] .= zero(T)
  _compute_gradient_k!(view(g, 1:nparam), tnrj, κ, yu)

  return g
end

function _compute_gradient_k!(
  g::AbstractVector,
  tnrj::NoFETerm,
  κ::AbstractVector,
  yu::FEFunctionType,
)
  return ForwardDiff.gradient!(g, tnrj.f, κ)
end

function _compute_hess_k_vals!(
  vals::AbstractVector,
  nlp::AbstractNLPModel,
  tnrj::NoFETerm,
  κ::AbstractVector,
  xyu::AbstractVector,
)
  vals .= LowerTriangular(ForwardDiff.hessian(tnrj.f, κ))[:]
  return vals
end

@doc raw"""
FETerm modeling the objective function of the optimization problem.

```math
\begin{aligned}
\int_{\Omega} f(y,u) d\Omega,
\end{aligned}
```

Constructor:

`EnergyFETerm(:: Function, :: Triangulation, :: Measure)`

See also: MixedEnergyFETerm, NoFETerm, `_obj_cell_integral`, `_obj_integral`,
`_compute_gradient_k!`
"""
struct EnergyFETerm <: AbstractEnergyTerm
  f::Function
  trian::Triangulation
  Ypde::FESpace
  Ycon::FESpace
end

function EnergyFETerm(f::Function, trian::Triangulation, Ypde)
  return EnergyFETerm(f, trian, Ypde, VoidFESpace())
end

function _obj_integral(tnrj::EnergyFETerm, κ::AbstractVector, x)
  @lencheck 0 κ
  if typeof(tnrj.Ycon) <: VoidFESpace
    return tnrj.f(x)
  else
    y, u = _split_FEFunction(x, tnrj.Ypde, tnrj.Ycon)
    return tnrj.f(y, u)
  end
end

function _compute_gradient!(
  g::AbstractVector,
  tnrj::EnergyFETerm,
  κ::AbstractVector,
  yu::FEFunctionType,
  Y::FESpace,
  X::FESpace,
)
  @lencheck 0 κ
  fsplit(x) = if typeof(tnrj.Ycon) <: VoidFESpace
    return tnrj.f(x)
  else
    y, u = _split_FEFunction(x, tnrj.Ypde, tnrj.Ycon)
    return tnrj.f(y, u)
  end
  vec = Gridap.FESpaces.collect_cell_vector(X, gradient(fsplit, yu))
  #Assemble the gradient in the "good" space
  assem = Gridap.FESpaces.SparseMatrixAssembler(Y, X)
  Gridap.FESpaces.assemble_vector!(g, assem, vec)
  return g
end

#=
function _compute_gradient_k!(g::AbstractVector, tnrj::EnergyFETerm, κ::AbstractVector{T}, yu::FEFunctionType) where {T}
  @lencheck 0 κ
  return g
end
=#

function _compute_hess_k_vals!(
  vals::AbstractVector,
  nlp::AbstractNLPModel,
  tnrj::EnergyFETerm,
  κ::AbstractVector{T},
  xyu::AbstractVector{T},
) where {T}
  @lencheck 0 κ
  return vals
end

@doc raw"""
FETerm modeling the objective function of the optimization problem with
functional and discrete unknowns.

```math
\begin{aligned}
\int_{\Omega} f(y,u,\kappa) d\Omega,
\end{aligned}
```

Constructor:

`MixedEnergyFETerm(:: Function, :: Triangulation, :: Int)`

See also: `EnergyFETerm`, `NoFETerm`, `_obj_cell_integral`, `_obj_integral`,
`_compute_gradient_k!`
"""
struct MixedEnergyFETerm <: AbstractEnergyTerm
  f::Function
  trian::Triangulation
  nparam::Integer #number of discrete unkonwns.
  inde::Bool
  Ypde::FESpace
  Ycon::FESpace

  function MixedEnergyFETerm(
    f::Function,
    trian::Triangulation,
    n::Integer,
    inde::Bool,
    Ypde::FESpace,
    Ycon::FESpace,
  )
    @assert n > 0
    return new(f, trian, n, inde, Ypde, Ycon)
  end
end

function MixedEnergyFETerm(f::Function, trian::Triangulation, n::Integer, Ypde, Ycon)
  inde = false
  return MixedEnergyFETerm(f, trian, n, inde, Ypde, Ycon)
end

function MixedEnergyFETerm(f::Function, trian::Triangulation, n::Integer, Ypde)
  return MixedEnergyFETerm(f, trian, n, false, Ypde, VoidFESpace())
end

function MixedEnergyFETerm(f::Function, trian::Triangulation, n::Integer, inde::Bool, Ypde)
  return MixedEnergyFETerm(f, trian, n, inde, Ypde, VoidFESpace())
end

function _obj_integral(tnrj::MixedEnergyFETerm, κ::AbstractVector, x)
  @lencheck tnrj.nparam κ
  if typeof(tnrj.Ycon) <: VoidFESpace
    return tnrj.f(κ, x)
  else
    y, u = _split_FEFunction(x, tnrj.Ypde, tnrj.Ycon)
    return tnrj.f(κ, y, u)
  end
end

function _compute_gradient!(
  g::AbstractVector,
  tnrj::MixedEnergyFETerm,
  κ::AbstractVector,
  yu::FEFunctionType,
  Y::FESpace,
  X::FESpace,
)
  @lencheck tnrj.nparam κ
  nyu = num_free_dofs(Y)
  @lencheck tnrj.nparam + nyu g

  fsplit(x) = if typeof(tnrj.Ycon) <: VoidFESpace
    return tnrj.f(κ, x)
  else
    y, u = _split_FEFunction(x, tnrj.Ypde, tnrj.Ycon)
    return tnrj.f(κ, y, u)
  end
  vec = Gridap.FESpaces.collect_cell_vector(X, gradient(fsplit, yu))
  #Assemble the gradient in the "good" space
  assem = Gridap.FESpaces.SparseMatrixAssembler(Y, X)
  Gridap.FESpaces.assemble_vector!(
    view(g, (tnrj.nparam + 1):(tnrj.nparam + nyu)),
    assem,
    vec,
  )

  _compute_gradient_k!(view(g, 1:(tnrj.nparam)), tnrj, κ, yu)

  return g
end

function _compute_gradient_k!(
  g::AbstractVector,
  tnrj::MixedEnergyFETerm,
  κ::AbstractVector,
  yu::FEFunctionType,
)
  @lencheck tnrj.nparam κ
  intf = @closure k -> sum(_obj_integral(tnrj, k, yu))
  return ForwardDiff.gradient!(g, intf, κ)
end

function _compute_hess_k_vals!(
  vals::AbstractVector,
  nlp::AbstractNLPModel,
  tnrj::MixedEnergyFETerm,
  κ::AbstractVector{T},
  xyu::AbstractVector{T},
) where {T}
  yu = FEFunction(nlp.pdemeta.Y, xyu)
  prows = nlp.pdemeta.tnrj.inde ? nlp.pdemeta.nparam : nlp.meta.nvar
  gk = @view nlp.workspace.g[1:prows]
  Hk = @view nlp.workspace.Hk[1:prows, 1:(nlp.pdemeta.nparam)]

  if nlp.pdemeta.tnrj.inde
    agrad = @closure (g, k) -> _compute_gradient_k!(g, nlp.pdemeta.tnrj, k, yu)
  else
    function _obj(x)
      κ = @view x[1:(nlp.pdemeta.nparam)]
      xyu = @view x[(nlp.pdemeta.nparam + 1):(nlp.meta.nvar)]
      yu = FEFunction(nlp.pdemeta.Y, xyu)
      int = _obj_integral(nlp.pdemeta.tnrj, κ, yu)
      return sum(int)
    end
    agrad = (g, k) -> ForwardDiff.gradient!(g, _obj, vcat(k, xyu))
  end
  ForwardDiff.jacobian!(Hk, agrad, gk, κ)
  vals .= [Hk[i, j] for i = 1:prows, j = 1:(nlp.pdemeta.nparam) if j ≤ i]

  return vals
end

#=
@doc raw"""
FETerm modeling the objective function of the optimization problem with
functional and discrete unknowns, describe as a norm and a regularizer.

```math
\begin{aligned}
\frac{1}{2}\|Fyu(y,u)\|^2_{L^2_\Omega} + \lambda\int_{\Omega} lyu(y,u) d\Omega
 + \frac{1}{2}\|Fk(κ)\|^2 + \mu lk(κ)
\end{aligned}
```

Constructor:

`ResidualEnergyFETerm(:: Function, :: Triangulation, :: Measure, :: Function, :: Int)`

See also: `EnergyFETerm`, `NoFETerm`, `MixedEnergyFETerm`
"""
struct ResidualEnergyFETerm <: AbstractEnergyTerm
  Fyu::Function
  #lyu      :: Function #regularizer
  #λ        :: Real
  trian::Triangulation
  Fk::Function
  #lk       :: Function #regularizer
  #μ        :: Real

  nparam::Integer #number of discrete unkonwns.

  #?counters :: NLSCounters #init at NLSCounters()

  function ResidualEnergyFETerm(
    Fyu::Function,
    trian::Triangulation,
    Fk::Function,
    n::Integer,
  )
    @assert n > 0
    return new(Fyu, trian, Fk, n)
  end
end

#TODO: this is specific to ResidualEnergyFETerm
function _jac_residual_yu end
function _jac_residual_k end
function _jprod_residual_yu end
function _jprod_residual_k end
function _jtprod_residual_yu end
function _jtprod_residual_k end
function hess_residual end

function _obj_cell_integral end

function _obj_integral end

function _compute_gradient_k end

function _compute_gradient! end

function _compute_hess_k_coo end

function _compute_hess_k_vals! end
=#
