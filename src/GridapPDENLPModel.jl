"""
PDENLPModels using Gridap.jl

https://github.com/gridap/Gridap.jl
Cite: Badia, S., & Verdugo, F. (2020). Gridap: An extensible Finite Element toolbox in Julia. Journal of Open Source Software, 5(52), 2520.

Find functions (y,u): Y -> ℜⁿ x ℜⁿ and κ ∈ ℜⁿ satisfying

min      ∫_Ω​ f(y,u,κ) dΩ​
s.t.     y solution of a PDE(u,κ)=0
         lcon <= c(y,u,κ) <= ucon
         lvar <= (y,u,κ)  <= uvar

The weak formulation is then:
res((y,u),(v,q)) = ∫ v PDE(u,y,κ) + ∫ q c(y,u,κ)

where the unknown (y,u) is a MultiField see Tutorials 7 and 8 of Gridap.
https://gridap.github.io/Tutorials/stable/pages/t007_darcy/
https://gridap.github.io/Tutorials/stable/pages/t008_inc_navier_stokes/

The set Ω​ is represented here with *trian* and *quad*.

`GridapPDENLPModel(:: Function, :: FESpace, :: FESpace, :: FEOperator, trian, quad; g :: Union{Function, Nothing} = nothing, H :: Union{Function, Nothing} = nothing, c :: Union{Function, Nothing} = nothing, J :: Union{Function, Nothing} = nothing, kwargs...)`

TODO:
- time evolution edp problems.
- Handle the case where op.jac is given
- Handle the case where J is not given (in jac - use ForwardDiff as in hess)
- Handle the case where g and H are given
- Handle several terms in the objective function (via an FEOperator)
- Handle finite dimension variables/unknwon parameters. Should we ask: f(yu, k) all the times or only when necessary (otw. f(yu) )
- Handle the constraints in the cell space by adding a new term in the weak formulation. Right now I handle the constraints at each dofs, which is ***.
- Handle FESource term (as in poisson-with-Neumann-and-Dirichlet example). [Now we model them via FETerm with a minus sign!]
- Test the Right-hand side if op is an AffineFEOperator
- Improve Gridap.FESpaces.assemble_matrix in hess! to get directly the lower triangular?
- l.257, in hess!: sparse(LowerTriangular(hess_yu)) #there must be a better way for this
- Be more explicit on the different types of FETerm in  _from_term_to_terms!
- Right now we handle only AffineFEOperator and FEOperatorFromTerms [to be specified]
- Could we control the Dirichlet boundary condition? (like classical control of heat equations)
- Clean the tests.

Example:
Unconstrained case:
`GridapPDENLPModel(x0, f, Yedp, Ycon, Xedp, Xcon, trian, quad)`
PDE-only case:
`GridapPDENLPModel(x0, x->0.0, Yedp, Ycon, Xedp, Xcon, trian, quad, op = op)`
"""
mutable struct GridapPDENLPModel <: AbstractNLPModel

  meta_func :: NLPModelMeta
  meta      :: NLPModelMeta

  counters :: Counters

  # For the objective function
  f :: Function
  g :: Union{Function, Nothing}
  H :: Union{Function, Nothing}
  # For the constraints
  c :: Union{Function, Nothing}
  J :: Union{Function, Nothing}

  #Gridap discretization
  Yedp  :: FESpace #TrialFESpace for the solution of the PDE
  Ycon  :: Union{FESpace, Nothing} #TrialFESpace for the parameter
  Xedp  :: FESpace #TestFESpace for the solution of the PDE
  Xcon  :: Union{FESpace, Nothing} #TestFESpace for the parameter
  X     :: Union{FESpace, Nothing} #concatenated TestFESpace
  Y     :: Union{FESpace, Nothing} #concatenated TrialFESpace

  op    :: Union{FEOperator, Nothing} #take (u, trian, quad, btrian, bquad, Ug, V0) returns FEOperator (AffineFEOperator or )

  trian
  quad

  n_edp_fields     :: Int #number of solutions' fields
  nvar_edp         :: Int #number of dofs in the solution functions
  n_control_fields :: Int #number of controls' fields
  nvar_control     :: Int #number of dofs in the control functions
  nparam           :: Int #number of unknown parameters

  nvar_per_field   :: Int #number of dofs per field (all equals for algebraic constraints)

  analytic_gradient :: Bool
  analytic_hessian  :: Bool
  analytic_jacobian :: Bool

  function GridapPDENLPModel(x0   :: AbstractVector{T}, #required to get the type and the length
                             y0   :: AbstractVector{T}, #required to get number of functional constraints
                             f    :: Function,
                             Yedp :: FESpace,
                             Ycon :: Union{FESpace, Nothing},
                             Xedp :: FESpace,
                             Xcon :: Union{FESpace, Nothing},
                             trian,
                             quad;
                             op   :: Union{FEOperator, Nothing} = nothing,
                             g    :: Union{Function, Nothing}   = nothing,
                             H    :: Union{Function, Nothing}   = nothing,
                             c    :: Union{Function, Nothing}   = nothing,
                             lcon :: AbstractVector             = fill!(similar(y0), zero(T)),
                             ucon :: AbstractVector             = fill!(similar(y0), zero(T)),
                             J    :: Union{Function, Nothing}   = nothing,
                             kwargs...) where T <: AbstractFloat

                             if Xcon != nothing && Ycon != nothing
                                 _xedp = typeof(Xedp) <: MultiFieldFESpace ? Xedp : MultiFieldFESpace([Xedp])
                                 _xcon = typeof(Xcon) <: MultiFieldFESpace ? Xcon : MultiFieldFESpace([Xcon])
                                 #Handle the case where Yedp or Ycon are single field FE space(s).
                                 _yedp = typeof(Yedp) <: MultiFieldFESpace ? Yedp : MultiFieldFESpace([Yedp])
                                 _ycon = typeof(Ycon) <: MultiFieldFESpace ? Ycon : MultiFieldFESpace([Ycon])
                                 #Build Y (resp. X) the trial (resp. test) space of the Multi Field function [y,u]
                                 X     = MultiFieldFESpace(vcat(_xedp.spaces, _xcon.spaces))
                                 Y     = MultiFieldFESpace(vcat(_yedp.spaces, _ycon.spaces))
                             elseif (Xcon == nothing) ⊻ (Ycon == nothing)
                                 throw("Error: Xcon or Ycon are both nothing or must be specified.")
                             else
                                 _xedp = typeof(Xedp) <: MultiFieldFESpace ? Xedp : MultiFieldFESpace([Xedp])
                                 X = _xedp
                                 Y = Yedp
                             end

                             #Handle the case where Yedp or Ycon are single field FE space(s).
                             n_edp_fields     = typeof(Yedp) <: MultiFieldFESpace ? Gridap.MultiField.num_fields(Yedp) : 1
                             n_control_fields = Ycon != nothing ? (typeof(Ycon) <: MultiFieldFESpace ? Gridap.MultiField.num_fields(Ycon) : 1) : 0
                             nvar_edp         = Gridap.FESpaces.num_free_dofs(Yedp)
                             nvar_control     = Ycon != nothing ? Gridap.FESpaces.num_free_dofs(Ycon) : 0
                             nvar             = length(x0)
                             nparam           = nvar - (nvar_edp + nvar_control)

                             nfields = n_edp_fields + n_control_fields

                             nfuncon = nvar * length(y0)
                             ncon = length(y0)
                             @lencheck ncon lcon ucon

                             meta_func = NLPModelMeta(nfields; ncon = ncon, y0 = y0, lcon = lcon, ucon = ucon)

                             #nnz in the jacobian: jacobian of PDE + one constraint per free dofs
                             nnzj = 0

                             nvar_per_field = Int(round(nvar / nfields))
                             _lcon = Array{T,1}(undef, ncon * nvar_per_field)
                             _ucon = Array{T,1}(undef, ncon * nvar_per_field)
                             ncon_disc  = nvar_per_field * ncon
                             if ncon != 0
                                 #If this is not true we have a problem with the constraints
                                 #@assert Int(nvar_edp / n_edp_fields) == Int(nvar_control / n_control_fields)

                                 for i = 1:nvar_per_field
                                  _lcon[(i - 1) * ncon + 1: i * ncon] = lcon
                                  _ucon[(i - 1) * ncon + 1: i * ncon] = ucon
                                 end
                                 nnzj += nvar_per_field * ncon * nfields
                             end

                             rhs_edp = []
                             if op != nothing #has a PDE constraint
                                 ncon_disc  += nvar_edp
                                 nnzj += nvar_edp * nvar


                                 if typeof(op) <: Gridap.FESpaces.FEOperatorFromTerms
                                   rhs_edp = zeros(nvar_edp)
                                 elseif typeof(op) <: AffineFEOperator
                                   rhs_edp = zeros(nvar_edp) #get_vector(op)
                                 end
                             end
                             lcon_disc  = vcat(rhs_edp, _lcon)
                             ucon_disc  = vcat(rhs_edp, _ucon)

                             y0_disc    = zeros(T, ncon_disc)
                             @lencheck ncon_disc y0_disc lcon_disc ucon_disc

                             nnzh = nvar * (nvar + 1) / 2

                             meta = NLPModelMeta(nvar; ncon = ncon_disc, y0 = y0_disc, lcon = lcon_disc, ucon = ucon_disc, kwargs...)

                             analytic_gradient = g != nothing
                             analytic_hessian  = H != nothing
                             analytic_jacobian = J != nothing

     return new(meta_func, meta, Counters(), f, g, H, c, J,
                Yedp, Ycon, Xedp, Xcon, X, Y, op, trian, quad,
                n_edp_fields, nvar_edp, n_control_fields, nvar_control,
                nparam, nvar_per_field,
                analytic_gradient, analytic_hessian, analytic_jacobian)
    end
end

function GridapPDENLPModel(x0   :: AbstractVector{T},
                           f    :: Function,
                           Yedp :: FESpace,
                           Ycon :: Union{FESpace, Nothing},
                           Xedp :: FESpace,
                           Xcon :: Union{FESpace, Nothing},
                           trian,
                           quad;
                           op   :: Union{FEOperator, Nothing} = nothing,
                           g    :: Union{Function, Nothing}   = nothing,
                           H    :: Union{Function, Nothing}   = nothing,
                           kwargs...) where T <: AbstractFloat

 return GridapPDENLPModel(x0, zeros(0), f, Yedp, Ycon, Xedp, Xcon, trian, quad; op = op, g = g, H = H, kwargs...)
end

show_header(io :: IO, nlp :: GridapPDENLPModel) = println(io, "GridapPDENLPModel")

"""
Apply FEFunction to dofs corresponding to the solution `y` and the control `u`.
"""
function _split_FEFunction(nlp :: GridapPDENLPModel, x :: AbstractVector)

    yh = FEFunction(nlp.Yedp, x[1:nlp.nvar_edp])
    uh = (nlp.Ycon != nothing) ? FEFunction(nlp.Ycon, x[1+nlp.nvar_edp:nlp.nvar_edp+nlp.nvar_control]) : nothing

 return yh, uh
end

function obj(nlp :: GridapPDENLPModel, x :: AbstractVector)

 @lencheck nlp.meta.nvar x
 increment!(nlp, :neval_obj)

 yu  = FEFunction(nlp.Y, x)
 if nlp.nparam > 0
   @warn "Tanj: test obj"
   κ   = x[nlp.meta.nvar - nlp.nparam+1:nlp.meta.nvar]
   int = integrate(nlp.f(yu, κ), nlp.trian, nlp.quad)
 else
   int = integrate(nlp.f(yu), nlp.trian, nlp.quad)
 end

 return sum(int)
end

function grad!(nlp :: GridapPDENLPModel, x :: AbstractVector, g :: AbstractVector)

    @lencheck nlp.meta.nvar x g
    increment!(nlp, :neval_grad)

    assem = Gridap.FESpaces.SparseMatrixAssembler(nlp.Y, nlp.X)
    yu    = FEFunction(nlp.Y, x)

    cell_yu    = Gridap.FESpaces.get_cell_values(yu)
    cell_id_yu = Gridap.Arrays.IdentityVector(length(cell_yu))

    function _cell_obj_yu(cell_yu)
         yuh = CellField(nlp.Y, cell_yu)
        _yuh = Gridap.FESpaces.restrict(yuh, nlp.trian)
        integrate(nlp.f(_yuh), nlp.trian, nlp.quad)
    end

    #Compute the gradient with AD
    cell_r_yu = Gridap.Arrays.autodiff_array_gradient(_cell_obj_yu, cell_yu, cell_id_yu)
    #Put the result in the format expected by Gridap.FESpaces.assemble_matrix
    vecdata_yu = [[cell_r_yu], [cell_id_yu]]
    #Assemble the gradient in the "good" space
    g .= Gridap.FESpaces.assemble_vector(assem, vecdata_yu)

    return g
end

include("hessian_func.jl")

function hess_coo(nlp :: GridapPDENLPModel, x :: AbstractVector; obj_weight :: Real = one(eltype(x)))

    yu    = FEFunction(nlp.Y, x)

    cell_yu    = Gridap.FESpaces.get_cell_values(yu)
    cell_id_yu = Gridap.Arrays.IdentityVector(length(cell_yu))

    function _cell_obj_yu(cell_yu)
         yuh = CellField(nlp.Y, cell_yu)
        _yuh = Gridap.FESpaces.restrict(yuh, nlp.trian)
        integrate(nlp.f(_yuh), nlp.trian, nlp.quad)
    end

    #Compute the hessian with AD
    cell_r_yu  = Gridap.Arrays.autodiff_array_hessian(_cell_obj_yu, cell_yu, cell_id_yu)
    #Assemble the matrix in the "good" space
    assem      = Gridap.FESpaces.SparseMatrixAssembler(nlp.Y, nlp.X)
    (I ,J, V) = assemble_hess(assem, cell_r_yu, cell_id_yu)

    return (I ,J, V)
end

function hess(nlp :: GridapPDENLPModel, x :: AbstractVector; obj_weight :: Real = one(eltype(x)))
    @lencheck nlp.meta.nvar x v Hv
    increment!(nlp, :neval_hess)


    (I, J, V) = hess_coo(nlp, x, obj_weight = obj_weight)

    mdofs = Gridap.FESpaces.num_free_dofs(nlp.X)
    ndofs = Gridap.FESpaces.num_free_dofs(nlp.Y)

    @assert mdofs == ndofs #otherwise there is an error in the Trial/Test spaces

    hess_yu = sparse(I, J, V, mdofs, ndofs)

    return hess_yu
end

function hprod!(nlp :: GridapPDENLPModel, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  @lencheck nlp.meta.nvar x v Hv
  increment!(nlp, :neval_hprod)

  if obj_weight == zero(eltype(x))
      Hv .= zero(similar(x))
      return Hv
  end

  #Only one lower triangular of the Hessian
  (rows, cols, vals) = hess_coo(nlp, x, obj_weight = obj_weight)

  coo_sym_prod!(cols, rows, vals, v, Hv)

 return Hv
end

function hess_op!(nlp :: GridapPDENLPModel, x :: AbstractVector, Hv :: AbstractVector; obj_weight::Real=one(eltype(x)))
  @lencheck nlp.meta.nvar x Hv
  (rows, cols, vals) = hess_coo(nlp, x, obj_weight = obj_weight)
  prod = @closure v -> coo_sym_prod!(cols, rows, vals, v, Hv)
  return LinearOperator{eltype(x)}(nlp.meta.nvar, nlp.meta.nvar,
                                   true, true, prod, prod, prod)
end

function hess_op(nlp :: GridapPDENLPModel, x :: AbstractVector; obj_weight::Real=one(eltype(x)))
  @lencheck nlp.meta.nvar x
  Hv = similar(x)
  return hess_op!(nlp, x, Hv)
end

function cons!(nlp :: GridapPDENLPModel, x :: AbstractVector{T}, c :: AbstractVector{T})  where T

    edp_residual = Array{T,1}(undef, nlp.nvar_edp)

    _from_terms_to_residual!(nlp.op, x, nlp, edp_residual)

    c .= edp_residual

    return c
end

function _from_terms_to_residual!(op  :: Gridap.FESpaces.FEOperatorFromTerms,
                                  x   :: AbstractVector{T},
                                  nlp :: GridapPDENLPModel,
                                  res :: AbstractVector{T}) where T <: AbstractFloat

    yu = FEFunction(nlp.Y, x)
    v  = Gridap.FESpaces.get_cell_basis(nlp.Xedp) #nlp.op.test

    w, r = [], []
    for term in op.terms

     w, r = _from_term_to_terms!(term, yu, v, w, r)

    end

    assem_y = Gridap.FESpaces.SparseMatrixAssembler(nlp.Yedp, nlp.Xedp)
    Gridap.FESpaces.assemble_vector!(res, assem_y, (w,r))

    return res
end

function _from_term_to_terms!(term :: Union{Gridap.FESpaces.NonlinearFETermWithAutodiff, Gridap.FESpaces.NonlinearFETerm},
                              yu   :: Union{Gridap.FESpaces.SingleFieldFEFunction, Gridap.MultiField.MultiFieldFEFunction},
                              v    :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
                              w    :: AbstractVector,
                              r    :: AbstractVector)

 _v  = restrict(v,  term.trian)
 _yu = restrict(yu, term.trian)

 cellvals = integrate(term.res(_yu, _v), term.trian, term.quad)
 cellids  = Gridap.FESpaces.get_cell_id(term)

 Gridap.FESpaces._push_vector_contribution!(w, r, cellvals, cellids)

 return w, r
end

function _from_term_to_terms!(term :: Gridap.FESpaces.FETerm, #FESource, AffineFETerm
                              yu   :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction},
                              v    :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
                              w    :: AbstractVector,
                              r    :: AbstractVector)

 cellvals = Gridap.FESpaces.get_cell_residual(term, yu, v)
 cellids  = Gridap.FESpaces.get_cell_id(term)

 Gridap.FESpaces._push_vector_contribution!(w, r, cellvals, cellids)

 return w, r
end

"""
Note:
- mul! seems faster than doing:
rows, cols, vals = findnz(get_matrix(op))
coo_prod!(cols, rows, vals, v, res)

- get_matrix(op) is a sparse matrix
- Benchmark equivalent to Gridap.FESpaces.residual!(res, op_affine.op, xrand)
"""
function _from_terms_to_residual!(op  :: AffineFEOperator,
                                  x   :: AbstractVector{T},
                                  nlp :: GridapPDENLPModel,
                                  res :: AbstractVector{T}) where T <: AbstractFloat

 mul!(res, get_matrix(op), x)
 axpy!(-one(T), get_vector(op), res)

 return res
end

function jac(nlp :: GridapPDENLPModel, x :: AbstractVector{T}) where T <: AbstractFloat
  @lencheck nlp.meta.nvar x
  increment!(nlp, :neval_jac)

  edp_jac = _from_terms_to_jacobian(nlp.op, x, nlp)

  return edp_jac
end

function _from_terms_to_jacobian(op  :: AffineFEOperator,
                                 x   :: AbstractVector{T},
                                 nlp :: GridapPDENLPModel) where T <: AbstractFloat

 return get_matrix(op)
end

"""
Note:
1) Compute the derivatives w.r.t. y and u separately.

2) Use AD for those derivatives. Only for the following:
- NonlinearFETerm (we neglect the inapropriate jac function);
- NonlinearFETermWithAutodiff
- TODO: Gridap.FESpaces.FETerm & AffineFETerm ?
- FESource <: AffineFETerm (jacobian of a FESource is nothing)
- LinearFETerm <: AffineFETerm (not implemented)
"""
function _from_terms_to_jacobian(op  :: Gridap.FESpaces.FEOperatorFromTerms,
                                 x   :: AbstractVector{T},
                                 nlp :: GridapPDENLPModel) where T <: AbstractFloat

 yh, uh = _split_FEFunction(nlp, x)
 yu     = FEFunction(nlp.Y, x)

 dy  = Gridap.FESpaces.get_cell_basis(nlp.Yedp)
 du  = nlp.nvar_control != 0 ? Gridap.FESpaces.get_cell_basis(nlp.Ycon) : nothing #use only jac is furnished
 dyu = Gridap.FESpaces.get_cell_basis(nlp.Y)
 v   = Gridap.FESpaces.get_cell_basis(nlp.Xedp) #nlp.op.test

 wu, wy  = [], []
 ru, ry  = [], []
 cu, cy  = [], []
 w, r, c = [], [], []

 for term in nlp.op.terms

   _jac_from_term_to_terms!(term, yu,  yh, uh,
                                  dyu, dy, du,
                                  v,
                                  w,  r,  c,
                                  wu, ru, cu,
                                  wy, ry, cy)

 end

 if nlp.nvar_control != 0
     assem_u = Gridap.FESpaces.SparseMatrixAssembler(nlp.Ycon, nlp.Xedp)
     Au      = Gridap.FESpaces.assemble_matrix(assem_u, (wu, ru, cu))
 else
     Au      = zeros(nlp.nvar_edp, 0)
 end

 assem_y = Gridap.FESpaces.SparseMatrixAssembler(nlp.Yedp, nlp.Xedp)
 Ay      = Gridap.FESpaces.assemble_matrix(assem_y, (wy, ry, cy))

 S = hcat(Ay,Au)

 assem = Gridap.FESpaces.SparseMatrixAssembler(nlp.Y, nlp.Xedp)
 #doesn't work as we may not have the good sparsity pattern.
 #Gridap.FESpaces.assemble_matrix_add!(S, assem, (w, r, c))
 S += Gridap.FESpaces.assemble_matrix(assem, (w, r, c))

 return S
end

#https://github.com/gridap/Gridap.jl/blob/758a8620756e164ba0e6b83dc8dcbb278015b3d9/src/FESpaces/SparseMatrixAssemblers.jl#L242
import Gridap.FESpaces._get_block_layout
#Tanj: this is an error when the jacobian matrix are of size 1xn.
#unit test: poinsson-with-Neumann-and-Dirichlet, l. 160.
function _get_block_layout(a::AbstractArray)
  nothing
end

function _jac_from_term_to_terms!(term  :: Gridap.FESpaces.FETerm,
                                  yu    :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction},
                                  yh    :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction},
                                  uh    :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction,Nothing},
                                  dyu   :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
                                  dy    :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
                                  du    :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField,Nothing},
                                  v     :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
                                  w     :: AbstractVector,
                                  r     :: AbstractVector,
                                  c     :: AbstractVector,
                                  wu    :: AbstractVector,
                                  ru    :: AbstractVector,
                                  cu    :: AbstractVector,
                                  wy    :: AbstractVector,
                                  ry    :: AbstractVector,
                                  cy    :: AbstractVector)

 @warn "_jac_from_term_to_terms!(::FETerm, ...): If that works, good for you."

 cellvals = get_cell_jacobian(term, yu, dyu, v)
 cellids  = get_cell_id(term)
 _push_matrix_contribution!(w, r, c, cellvals, cellids)

end

#https://github.com/gridap/Gridap.jl/blob/758a8620756e164ba0e6b83dc8dcbb278015b3d9/src/FESpaces/FETerms.jl#L367
function _jac_from_term_to_terms!(term  :: Union{Gridap.FESpaces.LinearFETerm, Gridap.FESpaces.AffineFETermFromIntegration},
                                  yu    :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction},
                                  yh    :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction},
                                  uh    :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction,Nothing},
                                  dyu   :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
                                  dy    :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
                                  du    :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField,Nothing},
                                  v     :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
                                  w     :: AbstractVector,
                                  r     :: AbstractVector,
                                  c     :: AbstractVector,
                                  wu    :: AbstractVector,
                                  ru    :: AbstractVector,
                                  cu    :: AbstractVector,
                                  wy    :: AbstractVector,
                                  ry    :: AbstractVector,
                                  cy    :: AbstractVector)

 _v  = restrict(v,  term.trian)
 _yuh = restrict(yu, term.trian)

 cellids  = Gridap.FESpaces.get_cell_id(term)
 cellvals = integrate(term.biform(_yuh, _v), term.trian, term.quad)

 Gridap.FESpaces._push_matrix_contribution!(w, r, c, cellvals, cellids)

end

function _jac_from_term_to_terms!(term  :: Union{Gridap.FESpaces.NonlinearFETermWithAutodiff,Gridap.FESpaces.NonlinearFETerm},
                                  yu    :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction},
                                  yh    :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction},
                                  uh    :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction,Nothing},
                                  dyu   :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
                                  dy    :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
                                  du    :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField,Nothing},
                                  v     :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
                                  w     :: AbstractVector,
                                  r     :: AbstractVector,
                                  c     :: AbstractVector,
                                  wu    :: AbstractVector,
                                  ru    :: AbstractVector,
                                  cu    :: AbstractVector,
                                  wy    :: AbstractVector,
                                  ry    :: AbstractVector,
                                  cy    :: AbstractVector)

 if typeof(term) == Gridap.FESpaces.NonlinearFETerm
     @warn "_jac_from_term_to_terms!: For NonlinearFETerm, function jac is used to compute the derivative w.r.t. y."
 end

 if du != nothing
     _jac_from_term_to_terms_u!(term, yh, uh, du, v, wu, ru, cu)
 end

 _jac_from_term_to_terms_y!(term, yh, uh, dy, v, wy, ry, cy)

end

#https://github.com/gridap/Gridap.jl/blob/758a8620756e164ba0e6b83dc8dcbb278015b3d9/src/FESpaces/FETerms.jl#L332
function _jac_from_term_to_terms!(term  :: Gridap.FESpaces.FESource,
                                  yu    :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction},
                                  yh    :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction},
                                  uh    :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction,Nothing},
                                  dyu   :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
                                  dy    :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
                                  du    :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField,Nothing},
                                  v     :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
                                  w     :: AbstractVector,
                                  r     :: AbstractVector,
                                  c     :: AbstractVector,
                                  wu    :: AbstractVector,
                                  ru    :: AbstractVector,
                                  cu    :: AbstractVector,
                                  wy    :: AbstractVector,
                                  ry    :: AbstractVector,
                                  cy    :: AbstractVector)

 nothing
end

function _jac_from_term_to_terms_u!(term :: Union{Gridap.FESpaces.NonlinearFETermWithAutodiff,Gridap.FESpaces.NonlinearFETerm},
                                    yh   :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction},
                                    uh   :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction},
                                    du   :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
                                    v    :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
                                    w    :: AbstractVector,
                                    r    :: AbstractVector,
                                    c    :: AbstractVector)

 _v  = restrict(v,  term.trian)
 _yh = restrict(yh, term.trian)

 cellids  = Gridap.FESpaces.get_cell_id(term)

 function uh_to_cell_residual(uf)
   _uf = Gridap.FESpaces.restrict(uf, term.trian)
   integrate(term.res(vcat(_yh, _uf), _v), term.trian, term.quad)
 end
 cellvals_u = Gridap.FESpaces.autodiff_cell_jacobian_from_residual(uh_to_cell_residual, uh, cellids)

 Gridap.FESpaces._push_matrix_contribution!(w, r, c, cellvals_u, cellids)

 return w, r, c
end

function _jac_from_term_to_terms_y!(term :: Gridap.FESpaces.NonlinearFETermWithAutodiff,
                                    yh   :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction},
                                    uh   :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction, Nothing},
                                    dy   :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField, Nothing},
                                    v    :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
                                    w    :: AbstractVector,
                                    r    :: AbstractVector,
                                    c    :: AbstractVector)

 _v  = restrict(v,  term.trian)
 _uh = (uh == nothing) ? Array{Gridap.CellData.GenericCellField{true,()}}(undef,0) : restrict(uh, term.trian)

 cellids  = Gridap.FESpaces.get_cell_id(term)

 function yh_to_cell_residual(yf)
   _yf = Gridap.FESpaces.restrict(yf, term.trian)
   integrate(term.res(vcat(_yf,_uh),_v), term.trian, term.quad)
 end

 cellvals_y = Gridap.FESpaces.autodiff_cell_jacobian_from_residual(yh_to_cell_residual, yh, cellids)

 Gridap.FESpaces._push_matrix_contribution!(w, r, c ,cellvals_y, cellids)

 return w, r, c
end

function _jac_from_term_to_terms_y!(term :: Gridap.FESpaces.NonlinearFETerm,
                                    yh   :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction},
                                    uh   :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction, Nothing},
                                    dy   :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField, Nothing},
                                    v    :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
                                    w    :: AbstractVector,
                                    r    :: AbstractVector,
                                    c    :: AbstractVector)

 _v  = restrict(v,  term.trian)
 _yh = restrict(yh, term.trian)
 _uh = (uh == nothing) ? Array{Gridap.CellData.GenericCellField{true,()}}(undef,0) : restrict(uh, term.trian)
 _dy = restrict(dy, term.trian)

 cellids  = Gridap.FESpaces.get_cell_id(term)

 cellvals_y = integrate(term.jac(vcat(_yh, _uh), _du, _v), term.trian, term.quad)

 Gridap.FESpaces._push_matrix_contribution!(w, r, c ,cellvals_y, cellids)

 return w, r, c
end

"""
    Jv = jprod!(nlp, x, v, Jv)
Evaluate ``J(x)v``, the Jacobian-vector product at `x` in place.

Note for GridapPDENLPModel:
- Evaluate the jacobian and then use mul! (here coo_prod! is slower as we have to compute findnz).
- Alternative: benefit from the AD? Jv .= ForwardDiff.derivative(t->nlp.c(nlp, x + t * v), 0)
when the jacobian is obtained by AD.
"""
function jprod!(nlp :: GridapPDENLPModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  @lencheck nlp.meta.nvar x v
  @lencheck nlp.meta.ncon Jv
  increment!(nlp, :neval_jprod)

  Jx = jac(nlp, x)
  mul!(Jv, Jx, v)

  return Jv
end

"""
    Jv = jtprod!(nlp, x, v, Jv)
Evaluate ``J(x)'v``, the Jacobian-vector product at `x` in place.

Note for GridapPDENLPModel:
- Evaluate the jacobian and then use mul! (here coo_prod! is slower as we have to compute findnz).
- Alternative: benefit from the AD? Jtv .= ForwardDiff.gradient(x -> dot(nlp.c(x), v), x)
when the jacobian is obtained by AD.
"""
function jtprod!(nlp :: GridapPDENLPModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  @lencheck nlp.meta.nvar x Jtv
  @lencheck nlp.meta.ncon v
  increment!(nlp, :neval_jtprod)

  Jx = jac(nlp, x)
  mul!(Jtv, Jx', v)

  return Jtv
end

function jac_op(nlp :: AbstractNLPModel, x :: AbstractVector{T}) where T <: AbstractFloat
  @lencheck nlp.meta.nvar x

  Jx = jac(nlp, x)

  Jv  = Array{T,1}(undef, nlp.meta.ncon)
  Jtv = Array{T,1}(undef, nlp.meta.nvar)

  return jac_op!(nlp, x, Jv, Jtv)
end

function jac_op!(nlp :: AbstractNLPModel, x :: AbstractVector,
                 Jv :: AbstractVector, Jtv :: AbstractVector)
  @lencheck nlp.meta.nvar x Jtv
  @lencheck nlp.meta.ncon Jv

  Jx = jac(nlp, x)

  prod   = @closure v -> mul!(Jv,  Jx,  v)
  ctprod = @closure v -> mul!(Jtv, Jx', v)
  return LinearOperator{eltype(x)}(nlp.meta.ncon, nlp.meta.nvar,
                                   false, false, prod, ctprod, ctprod)
end


#ℓ(x) = obj_weight * nlp.f(x) + dot(nlp.c(x), y)
#Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x + t * v), 0)
function hprod!(nlp  :: GridapPDENLPModel, x :: AbstractVector, λ :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Real = one(eltype(x)))

  @lencheck nlp.meta.nvar x v Hv
  @lencheck nlp.meta.ncon λ
  increment!(nlp, :neval_hprod)

  λ_edp = λ[1:nlp.nvar_edp]
  λ_con = λ[nlp.nvar_edp + 1 : nlp.meta.ncon]

  _from_terms_to_hprod!(nlp.op, x, λ_edp, v, nlp, Hv, obj_weight)

  return Hv
end

function _from_terms_to_hprod!(op  :: Gridap.FESpaces.AffineFEOperator,
                               x   :: AbstractVector{T},
                               λ   :: AbstractVector{T},
                               v   :: AbstractVector{T},
                               nlp :: GridapPDENLPModel,
                               Hv  :: AbstractVector{T},
                               obj_weight :: T) where T <: AbstractFloat

 decrement!(nlp, :neval_hprod) #otherwise we would count 2 hprod!
 #By definition the hessian of an AffineFEOperator vanishes.
 return hprod!(nlp, x, v, Hv; obj_weight = obj_weight)
end

function _from_terms_to_hprod!(op  :: Gridap.FESpaces.FEOperatorFromTerms,
                               x   :: AbstractVector{T},
                               λ   :: AbstractVector,
                               v   :: AbstractVector{T},
                               nlp :: GridapPDENLPModel,
                               Hv  :: AbstractVector{T},
                               obj_weight :: T) where T <: AbstractFloat

 @warn "Almost there, what is not working? - update the hprod of the objective function"
 #Second part: obj_weight * nlp.f(x) + dot(nlp.c(x), y)
 yu = FEFunction(nlp.Y, x)
 vf = FEFunction(nlp.Y, v)
 λf = FEFunction(nlp.Yedp, λ_edp)
 vc = Gridap.FESpaces.get_cell_basis(nlp.Xedp) #X or Xedp? /nlp.op.test

 cell_yu = Gridap.FESpaces.get_cell_values(yu)
 cell_vf = Gridap.FESpaces.get_cell_values(vf)
 cell_λf = Gridap.FESpaces.get_cell_values(λf)

 ncells    = length(cell_yu)
 cell_id = Gridap.Arrays.IdentityVector(length(cell_yu))

 function _cell_lag_t(cell)
      yu  = CellField(nlp.Y, cell)

     _yu  = Gridap.FESpaces.restrict(yu, nlp.trian)
     _lag = integrate(nlp.f(_yu), nlp.trian, nlp.quad)

     _dotvalues = Array{Any, 1}(undef, ncells)
     for term in nlp.op.terms
       _vc = restrict(vc,  term.trian)
       _yu = restrict(yu, term.trian)
       cellvals = integrate(term.res(_yu, _vc), term.trian, term.quad)
       for i=1:ncells #doesn't work as ncells varies function of trian ...
           @show i
       _dotvalues[i] = dot(cellvals[i], cell_λf[i]) #setindex not defined for Gridap.Arrays
       end
     end
     #_lag += _dotvalues
     return _lag
 end

 #Compute the gradient with AD
 function _cell_lag_grad_t(t)

      ct    = t * ones(length(cell_vf[1]))
     _cell  = Array{typeof(ct .* cell_vf[1])}(undef, ncells)
     for i=1:ncells
         _cell[i] = cell_yu[i] + ct .* cell_vf[i]
     end
     Gridap.Arrays.autodiff_array_gradient(_cell_lag_t, _cell, cell_id) #returns NaN sometimes ? se pde-only-incompressible-NS.jl
 end

 #Compute the derivative w.r.t. to t of _cell_grad_t
 #This might be slow as it cannot be decomposed (easily) cell by cell
 cell_r_yu = ForwardDiff.derivative(_cell_lag_grad_t, 0.)

 #Put the result in the format expected by Gridap.FESpaces.assemble_matrix
 vecdata_yu = [[cell_r_yu], [cell_id]]
 assem = Gridap.FESpaces.SparseMatrixAssembler(nlp.Y, nlp.X)
 #Assemble the gradient in the "good" space
 Hv .= Gridap.FESpaces.assemble_vector(assem, vecdata_yu) + cons_residual

 return Hv
end

include("additional_functions.jl")
