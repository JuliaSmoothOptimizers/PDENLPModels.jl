abstract type AbstractEnergyTerm end

"""
Return the integral of the objective function

`_obj_cell_integral(:: AbstractEnergyTerm, :: GenericCellField,  :: AbstractVector)`

x is a vector of GenericCellField, for instance resulting from
`yuh = CellField(Y, cell_yu)`.

See also: MixedEnergyFETerm, EnergyFETerm, NoFETerm, \\_obj\\_integral,
\\_compute\\_gradient\\_k!
"""
function _obj_cell_integral end

"""
Return the integral of the objective function

`_obj_integral(:: AbstractEnergyTerm, :: FEFunctionType, :: AbstractVector)`

See also: MixedEnergyFETerm, EnergyFETerm, NoFETerm,
\\_obj\\_cell\\_integral, \\_compute\\_gradient\\_k!
"""
function _obj_integral end

"""
Return the derivative of the objective function w.r.t. κ.

`_compute_gradient_k(:: AbstractEnergyTerm, :: FEFunctionType, :: AbstractVector)`

See also: MixedEnergyFETerm, EnergyFETerm, NoFETerm, \\_obj\\_integral,
\\_obj\\_cell\\_integral
"""
function _compute_gradient_k end

"""
Return the derivative (jacobian) of the given function w.r.t. κ in coo format.

`_compute_hessian_k_coo(:: AbstractEnergyTerm, :: Function, :: AbstractVector)`

See also: MixedEnergyFETerm, EnergyFETerm, NoFETerm, \\_obj\\_integral,
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

See also: MixedEnergyFETerm, NoFETerm, \_obj\_cell\_integral, \_obj\_integral,
_compute\_gradient\_k!
"""
struct EnergyFETerm <: AbstractEnergyTerm
    # For the objective function
    f        :: Function
    trian    :: Triangulation
    quad     :: CellQuadrature
end

function _obj_integral(term :: EnergyFETerm,
                       κ    :: AbstractVector,
                       x    :: FEFunctionType)
  @lencheck 0 κ
  return integrate(term.f(x), term.trian, term.quad)
end

function _obj_cell_integral(term :: EnergyFETerm,
                            κ    :: AbstractVector,
                            yuh  :: CellFieldType)
  @lencheck 0 κ
  _yuh = Gridap.FESpaces.restrict(yuh, term.trian)

  return integrate(term.f(_yuh), term.trian, term.quad)
end

function _compute_gradient!(g    :: AbstractVector,
                            tnrj :: EnergyFETerm,
                            κ    :: AbstractVector,
                            yu   :: FEFunctionType,
                            Y    :: FESpace,
                            X    :: FESpace)
    @lencheck 0 κ

    cell_yu    = Gridap.FESpaces.get_cell_values(yu)
    cell_id_yu = Gridap.Arrays.IdentityVector(length(cell_yu))

    function _cell_obj_yu(cell)
         yuh = CellField(Y, cell)
        _obj_cell_integral(tnrj, κ, yuh)
    end

    #Compute the gradient with AD
    cell_r_yu = Gridap.Arrays.autodiff_array_gradient(_cell_obj_yu,
                                                       cell_yu,
                                                       cell_id_yu)
    #Put the result in the format expected by Gridap.FESpaces.assemble_matrix
    vecdata_yu = [[cell_r_yu], [cell_id_yu]]
    #Assemble the gradient in the "good" space
    assem  = Gridap.FESpaces.SparseMatrixAssembler(Y, X)
    g .= Gridap.FESpaces.assemble_vector(assem, vecdata_yu)

 return g
end

function _compute_gradient_k(term :: EnergyFETerm,
                             κ    :: AbstractVector,
                             yu   :: FEFunctionType)
  @lencheck 0 κ
  return []
end

function _compute_hess_coo(tnrj :: EnergyFETerm,
                           κ    :: AbstractVector,
                           yu   :: FEFunctionType,
                           Y    :: FESpace,
                           X    :: FESpace)
    @lencheck 0 κ

    cell_yu    = Gridap.FESpaces.get_cell_values(yu)
    cell_id_yu = Gridap.Arrays.IdentityVector(length(cell_yu))

    function _cell_obj_yu(cell)
         yuh = CellField(Y, cell)
        _obj_cell_integral(tnrj, κ, yuh)
    end

    #Compute the hessian with AD
    cell_r_yu  = Gridap.Arrays.autodiff_array_hessian(_cell_obj_yu, cell_yu, cell_id_yu)
    #Assemble the matrix in the "good" space
    assem      = Gridap.FESpaces.SparseMatrixAssembler(Y, X)
    (I ,J, V) = assemble_hess(assem, cell_r_yu, cell_id_yu)

    return (I ,J, V)
end

function _compute_hessian_k_coo(term  :: EnergyFETerm,
                                κ     :: AbstractVector,
                                agrad :: Function)
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

`MixedEnergyFETerm(:: Function, :: Triangulation, :: CellQuadrature, :: Int)`

See also: EnergyFETerm, NoFETerm, \_obj\_cell\_integral, \_obj\_integral,
\_compute\_gradient\_k!
"""
struct MixedEnergyFETerm <: AbstractEnergyTerm
    f        :: Function
    trian    :: Triangulation
    quad     :: CellQuadrature

    nparam   :: Int #number of discrete unkonwns.

    function MixedEnergyFETerm(f     :: Function,
                               trian :: Triangulation,
                               quad  :: CellQuadrature,
                               n     :: Int)
        @assert n > 0
        return new(f, trian, quad, n)
    end
end

function _obj_integral(term :: MixedEnergyFETerm,
                       κ    :: AbstractVector,
                       x    :: FEFunctionType)
  @lencheck term.nparam κ
  return integrate(term.f(κ, x), term.trian, term.quad)
end

function _obj_cell_integral(term :: MixedEnergyFETerm,
                            κ    :: AbstractVector,
                            yuh  :: CellFieldType)
  @lencheck term.nparam κ
  _yuh = Gridap.FESpaces.restrict(yuh, term.trian)

  return integrate(term.f(κ, _yuh), term.trian, term.quad)
end

function _compute_gradient!(g    :: AbstractVector,
                            tnrj :: MixedEnergyFETerm,
                            κ    :: AbstractVector,
                            yu   :: FEFunctionType,
                            Y    :: FESpace,
                            X    :: FESpace)
    @lencheck term.nparam κ
    nyu = num_free_dofs(Y)
    @lencheck term.nparam+nyu g

    cell_yu    = Gridap.FESpaces.get_cell_values(yu)
    cell_id_yu = Gridap.Arrays.IdentityVector(length(cell_yu))

    function _cell_obj_yu(cell)
         yuh = CellField(Y, cell)
        _obj_cell_integral(tnrj, κ, yuh)
    end

    #Compute the gradient with AD
    cell_r_yu = Gridap.Arrays.autodiff_array_gradient(_cell_obj_yu,
                                                       cell_yu,
                                                       cell_id_yu)
    #Put the result in the format expected by Gridap.FESpaces.assemble_matrix
    vecdata_yu = [[cell_r_yu], [cell_id_yu]]
    #Assemble the gradient in the "good" space
    assem  = Gridap.FESpaces.SparseMatrixAssembler(Y, X)
    g[term.nparam + 1 : term.nparam + nyu] .= Gridap.FESpaces.assemble_vector(assem, vecdata_yu)

    g[1 : term.nparam] .= _compute_gradient_k(tnrj, κ, yu)

 return g
end

function _compute_gradient_k(term :: MixedEnergyFETerm,
                             κ    :: AbstractVector,
                             yu   :: FEFunctionType)
  @lencheck term.nparam κ
  intf = @closure k -> sum(integrate(term.f(k, yu), term.trian, term.quad))
  return ForwardDiff.gradient(intf, κ)
end

function _compute_hess_coo(tnrj :: MixedEnergyFETerm,
                           κ    :: AbstractVector,
                           yu   :: FEFunctionType,
                           Y    :: FESpace,
                           X    :: FESpace)

    cell_yu    = Gridap.FESpaces.get_cell_values(yu)
    cell_id_yu = Gridap.Arrays.IdentityVector(length(cell_yu))

    function _cell_obj_yu(cell)
         yuh = CellField(Y, cell)
        _obj_cell_integral(tnrj, κ, yuh)
    end

    #Compute the hessian with AD
    cell_r_yu  = Gridap.Arrays.autodiff_array_hessian(_cell_obj_yu, cell_yu, cell_id_yu)
    #Assemble the matrix in the "good" space
    assem      = Gridap.FESpaces.SparseMatrixAssembler(Y, X)
    (I, J, V) = assemble_hess(assem, cell_r_yu, cell_id_yu)

    @warn "Not finished"
    #\TODO: Compute the derivative w.r.t. κ
    #agrad = @closure k -> grad(nlp, vcat(k, xyu))
    #(I2, J2, V2) = _compute_hessian_k_coo(nlp.tnrj, agrad, κ) #Not sure we really need that? Almost impossible to test separately
    #We probably need to sort by columns then.
    #Eventually, has a hess_coo_k ??

    return (I .+ tnrj.nparam, J .+ tnrj.nparam, V)
end

function _compute_hessian_k_coo(term  :: MixedEnergyFETerm,
                                agrad :: Function,
                                κ     :: AbstractVector)
 @lencheck term.nparam κ
 #Take J-I as we transpose
 (J,I,V) = findnz(ForwardDiff.jacobian(agrad, κ))
 return (I,J,V)
end

function _compute_hessian_k_coo(term  :: MixedEnergyFETerm,
                                κ     :: AbstractVector,
                                yu    :: FEFunctionType)
 @lencheck term.nparam κ
 fk = @closure k -> term.f(k, yu)
 (I, J, V) = findnz(sparse(LowerTriangular(ForwardDiff.hessian(fk, κ))))
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

See also: MixedEnergyFETerm, EnergyFETerm, \_obj\_cell\_integral,
\_obj\_integral, \_compute\_gradient\_k!
"""
struct NoFETerm <: AbstractEnergyTerm
    # For the objective function
    f      :: Function
end

function NoFETerm()
 return NoFETerm(x -> 0.)
end

_obj_integral(term :: NoFETerm, κ :: AbstractVector, x :: FEFunctionType) = term.f(κ)
_obj_cell_integral(term :: NoFETerm, κ :: AbstractVector, yuh :: CellFieldType) = term.f(κ)

function _compute_gradient!(g    :: AbstractVector,
                            tnrj :: NoFETerm,
                            κ    :: AbstractVector,
                            yu   :: FEFunctionType,
                            Y    :: FESpace,
                            X    :: FESpace)
    nparam = length(κ)
    nyu    = num_free_dofs(Y)
    nvar   = nparam + nyu
    @lencheck nvar g

    #Assemble the gradient in the "good" space
    g[nparam + 1 : nvar] .= zeros(nyu)

    g[1 : nparam] .= _compute_gradient_k(tnrj, κ, yu)

 return g
end


function _compute_gradient_k(term :: NoFETerm,
                             κ    :: AbstractVector,
                             yu   :: FEFunctionType)
  return ForwardDiff.gradient(term.f, κ)
end

function _compute_hess_coo(tnrj :: NoFETerm,
                           κ    :: AbstractVector,
                           yu   :: FEFunctionType,
                           Y    :: FESpace,
                           X    :: FESpace)
    nyu = num_free_dofs(Y)
    #Compute the derivative w.r.t. κ
    (I, J, V) = _compute_hessian_k_coo(tnrj, κ, yu)

    return (I, J, V)
end

function _compute_hessian_k_coo(term  :: NoFETerm,
                                κ     :: AbstractVector,
                                yu    :: FEFunctionType)

 (I, J, V) = findnz(sparse(LowerTriangular(ForwardDiff.hessian(term.f, κ))))
 return (I,J,V)
end
