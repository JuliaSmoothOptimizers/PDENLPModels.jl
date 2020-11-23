abstract type AbstractEnergyTerm end

"""
Return the integral of the objective function

`_obj_cell_integral(:: AbstractEnergyTerm, :: GenericCellField,  :: AbstractVector)`

x is a vector of GenericCellField, for instance resulting from
`yuh = CellField(nlp.Y, cell_yu)`.

See also: MixteEnergyFETerm, EnergyFETerm, NoFETerm, \\_obj\\_integral,
\\_compute\\_gradient\\_k!
"""
function _obj_cell_integral end

"""
Return the integral of the objective function

`_obj_integral(:: AbstractEnergyTerm, :: FEFunctionType, :: AbstractVector)`

See also: MixteEnergyFETerm, EnergyFETerm, NoFETerm,
\\_obj\\_cell\\_integral, \\_compute\\_gradient\\_k!
"""
function _obj_integral end

"""
Return the derivative of the objective function w.r.t. κ and allocate it in g.

`_compute_gradient_k!(:: AbstractVector, :: AbstractEnergyTerm, :: FEFunctionType, :: AbstractVector)`

See also: MixteEnergyFETerm, EnergyFETerm, NoFETerm, \\_obj\\_integral,
\\_obj\\_cell\\_integral
"""
function _compute_gradient_k! end

"""
Return the derivative (jacobian) of the given function w.r.t. κ in coo format.

`_compute_hessian_k_coo(:: AbstractEnergyTerm, :: Function, :: AbstractVector)`

See also: MixteEnergyFETerm, EnergyFETerm, NoFETerm, \\_obj\\_integral,
\\_obj\\_cell\\_integral, \\_compute\\_gradient\\_k!
"""
function _compute_hessian_k_coo end

@doc raw"""
FETerm modeling the objective function of the optimization problem.

```math
\begin{equation}
\int_{\Omega} f(y,u) d\Omega,
\end{equation}
```
where Ω is described by:
 - trian :: Triangulation
 - quad  :: CellQuadrature

Constructor:

`EnergyFETerm(:: Function, :: Triangulation, :: CellQuadrature)`

See also: MixteEnergyFETerm, NoFETerm, \_obj\_cell\_integral, \_obj\_integral,
_compute\_gradient\_k!
"""
struct EnergyFETerm <: AbstractEnergyTerm
    # For the objective function
    f        :: Function
    trian    :: Triangulation
    quad     :: CellQuadrature
end

function _obj_integral(term :: EnergyFETerm,
                       x    :: FEFunctionType,
                       κ    :: AbstractVector)
  @lencheck 0 κ
  return integrate(term.f(x), term.trian, term.quad)
end

function _obj_cell_integral(term :: EnergyFETerm,
                            yuh  :: CellFieldType,
                            κ    :: AbstractVector)
  @lencheck 0 κ
  _yuh = Gridap.FESpaces.restrict(yuh, term.trian)

  return integrate(term.f(_yuh), term.trian, term.quad)
end

function _compute_gradient_k!(g    :: AbstractVector,
                              term :: EnergyFETerm,
                              yu   :: FEFunctionType,
                              κ    :: AbstractVector)
  @lencheck 0 κ g
  return []
end

function _compute_hessian_k_coo(term  :: EnergyFETerm,
                                agrad :: Function,
                                κ     :: AbstractVector)
 @lencheck 0 κ
 return ([],[],[])
end

@doc raw"""
FETerm modeling the objective function of the optimization problem with
functional and discrete unknowns.

```math
\begin{equation}
\int_{\Omega} f(y,u,\kappa) d\Omega,
\end{equation}
```
where Ω is described by:
 - trian :: Triangulation
 - quad  :: CellQuadrature

Constructor:

`MixteEnergyFETerm(:: Function, :: Triangulation, :: CellQuadrature, :: Int)`

See also: EnergyFETerm, NoFETerm, \_obj\_cell\_integral, \_obj\_integral,
\_compute\_gradient\_k!
"""
struct MixteEnergyFETerm <: AbstractEnergyTerm
    f        :: Function
    trian    :: Triangulation
    quad     :: CellQuadrature

    nparam   :: Int #number of discrete unkonwns.

    function MixteEnergyFETerm(f     :: Function, 
                               trian :: Triangulation,
                               quad  :: CellQuadrature,
                               n     :: Int)
        @assert n > 0
        return new(f, trian, quad, n)
    end
end

function _obj_integral(term :: MixteEnergyFETerm,
                       x    :: FEFunctionType,
                       κ    :: AbstractVector)
  @lencheck term.nparam κ
  return integrate(term.f(x, κ), term.trian, term.quad)
end

function _obj_cell_integral(term :: MixteEnergyFETerm,
                            yuh  :: CellFieldType,
                            κ    :: AbstractVector)
  @lencheck term.nparam κ
  _yuh = Gridap.FESpaces.restrict(yuh, term.trian)

  return integrate(term.f(_yuh, κ), term.trian, term.quad)
end

function _compute_gradient_k!(g    :: AbstractVector,
                              term :: MixteEnergyFETerm,
                              yu   :: FEFunctionType,
                              κ    :: AbstractVector)
  @lencheck term.nparam κ g
  intf = @closure k -> sum(integrate(term.f(yu, k), term.trian, term.quad))
  return ForwardDiff.gradient!(g, intf, κ)
end

function _compute_hessian_k_coo(term  :: MixteEnergyFETerm,
                                agrad :: Function,
                                κ     :: AbstractVector)
 @lencheck term.nparam κ
 #Take J-I as we transpose
 (J,I,V) = findnz(ForwardDiff.jacobian(agrad, κ))
 return (I,J,V)
end

@doc raw"""
FETerm modeling the objective function when there are no intregral objective.

```math
\begin{equation}
 f(\kappa)
\end{equation}
 ```

Constructors:

  `NoFETerm()`

  `NoFETerm(:: Function)`

See also: MixteEnergyFETerm, EnergyFETerm, \_obj\_cell\_integral,
\_obj\_integral, \_compute\_gradient\_k!
"""
struct NoFETerm <: AbstractEnergyTerm
    # For the objective function
    f      :: Function
end

function NoFETerm()
 return NoFETerm(x -> 0.)
end

_obj_integral(term :: NoFETerm, x :: FEFunctionType, κ :: AbstractVector) = term.f(κ)
_obj_cell_integral(term :: NoFETerm, yuh :: CellFieldType, κ :: AbstractVector) = term.f(κ)

function _compute_gradient_k!(g    :: AbstractVector,
                              term :: NoFETerm,
                              yu   :: FEFunctionType,
                              κ    :: AbstractVector)
  @lencheck length(κ) g
  return ForwardDiff.gradient!(g, term.f, κ)
end

function _compute_hessian_k_coo(term  :: NoFETerm,
                                agrad :: Function,
                                κ     :: AbstractVector)
 #Take J-I as we transpose
 (J,I,V) = findnz(ForwardDiff.jacobian(agrad, κ))
 return (I,J,V)
end
