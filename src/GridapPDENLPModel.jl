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

Example:
Unconstrained case:
`GridapPDENLPModel(x0, zeros(0), f, Yedp, Ycon, Xedp, Xcon, trian, quad)`
PDE-only case:
`GridapPDENLPModel(x0, zeros(0), x->0.0, Yedp, Ycon, Xedp, Xcon, trian, quad, op = op)`
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

  function GridapPDENLPModel(x0   :: AbstractVector{T}, #required to get the type
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
                             kwargs...) where T

                             if Xcon != nothing && Ycon != nothing
                                 _xedp = typeof(Xedp) <: MultiFieldFESpace ? Xedp : MultiFieldFESpace([Xedp])
                                 _xcon = typeof(Xcon) <: MultiFieldFESpace ? Xcon : MultiFieldFESpace([Xcon])
                                 X     = MultiFieldFESpace(vcat(_xedp.spaces, _xcon.spaces))
                                 Y     = MultiFieldFESpace(vcat(Yedp.spaces, Ycon.spaces))
                             else
                                 _xedp = typeof(Xedp) <: MultiFieldFESpace ? Xedp : MultiFieldFESpace([Xedp])
                                 X = _xedp
                                 Y = Yedp
                             end

                             n_edp_fields     = Gridap.MultiField.num_fields(Yedp)
                             n_control_fields = Ycon != nothing ? Gridap.MultiField.num_fields(Ycon) : 0
                             nvar_edp         = Gridap.FESpaces.num_free_dofs(Yedp)
                             nvar_control     = Ycon != nothing ? Gridap.FESpaces.num_free_dofs(Ycon) : 0
                             nvar             =  length(x0)
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
                                   rhs_edp = get_vector(op)
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

show_header(io :: IO, nlp :: GridapPDENLPModel) = println(io, "GridapPDENLPModel")

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

function hess(nlp :: GridapPDENLPModel, x :: AbstractVector; obj_weight :: Real = one(eltype(x)))

    assem = Gridap.FESpaces.SparseMatrixAssembler(nlp.Y, nlp.X)
    yu    = FEFunction(nlp.Y, x)

    cell_yu    = Gridap.FESpaces.get_cell_values(yu)
    cell_id_yu = Gridap.Arrays.IdentityVector(length(cell_yu))

    #
    function _cell_obj_yu(cell_yu)
         yuh = CellField(nlp.Y, cell_yu)
        _yuh = Gridap.FESpaces.restrict(yuh, nlp.trian)
        integrate(nlp.f(_yuh), nlp.trian, nlp.quad)
    end

    #Compute the hessian with AD
    cell_r_yu  = Gridap.Arrays.autodiff_array_hessian(_cell_obj_yu, cell_yu, cell_id_yu)
    #Put the result in the format expected by Gridap.FESpaces.assemble_matrix
    matdata_yu = [[cell_r_yu], [cell_id_yu], [cell_id_yu]]
    #Assemble the matrix in the "good" space
    hess_yu   = Gridap.FESpaces.assemble_matrix(assem, matdata_yu)

    @warn "hess(nlp, x) returns (for now) the full matrix (to check symmetry)"
    return hess_yu #LowerTriangular(_hess_yu)
end

#In the affine case, returns : - get_matrix(opl) * get_free_values(uh)
function cons!(nlp :: GridapPDENLPModel, x :: AbstractVector{T}, c :: AbstractVector{T})  where T

    #Notations
    nvar_edp         = nlp.nvar_edp
    nvar_per_field   = nlp.nvar_per_field
    n_control_fields = nlp.n_control_fields
    n_edp_fields     = nlp.n_edp_fields
    nfields          = nlp.meta_func.nvar
    nconf            = nlp.meta_func.ncon

    edp_residual = Array{T,1}(undef, nvar_edp)

    uy = FEFunction(nlp.Y, x)
    v  = Gridap.FESpaces.get_cell_basis(nlp.op.test) #Xedp

    w, r = [], []
    for term in nlp.op.terms
      _v  = restrict(v,  term.trian)
      _uy = restrict(uy, term.trian)

      cellvals = integrate(term.res(_uy, _v), term.trian, term.quad)
      cellids  = Gridap.FESpaces.get_cell_id(term)

      Gridap.FESpaces._push_vector_contribution!(w, r, cellvals, cellids)
    end
    assem_y = Gridap.FESpaces.SparseMatrixAssembler(nlp.Yedp, nlp.Xedp)
    Gridap.FESpaces.assemble_vector!(edp_residual, assem_y, (w,r))

    cons_residual = Array{T,1}(undef, nconf * nvar_per_field)
    if nlp.meta_func.ncon != 0
        _y, _u = _get_y_and_u(nlp, x)
        for i = 1:nvar_per_field
         cons_residual[(i - 1) * nconf + 1: i * nconf] = nlp.c(_y[:,i],_u[:,i])
        end
    end

    c .= vcat(edp_residual, cons_residual)

    return c
end

function jac(nlp :: GridapPDENLPModel, x :: AbstractVector{T}) where T
  @lencheck nlp.meta.nvar x
  increment!(nlp, :neval_jac)

  #Notations
  nvar_edp         = nlp.nvar_edp
  nvar_control     = nlp.nvar_control
  nvar_per_field   = nlp.nvar_per_field
  n_control_fields = nlp.n_control_fields
  n_edp_fields     = nlp.n_edp_fields
  nfields          = nlp.meta_func.nvar
  nconf            = nlp.meta_func.ncon

  Ay, Au = Array{T,2}(undef, nvar_edp, nvar_edp), Array{T,2}(undef, nvar_edp, nvar_control)

  yh = FEFunction(nlp.Yedp, x[1:nvar_edp])
  uh = (nlp.Ycon != nothing) ? FEFunction(nlp.Ycon, x[1+nvar_edp:nvar_edp+nvar_control]) : nothing
  #du = Gridap.FESpaces.get_cell_basis(nlp.op.trial) #use only jac is furnished
  v  = Gridap.FESpaces.get_cell_basis(nlp.op.test) #Xedp
  wu, wy = [], []
  ru, ry = [], []
  cu, cy = [], []
  for term in nlp.op.terms

    _v  = restrict(v,  term.trian)
    _yh = restrict(yh, term.trian)
    _uh = (uh == nothing) ? Array{Gridap.CellData.GenericCellField{true,()}}(undef,0) : restrict(uh, term.trian)

    cellids  = Gridap.FESpaces.get_cell_id(term)

    if nvar_control != 0
        function uh_to_cell_residual(uf)
          _uf = Gridap.FESpaces.restrict(uf, term.trian)
          integrate(term.res(vcat(_yh,_uf),_v), term.trian, term.quad)
        end
        cellvals_u = Gridap.FESpaces.autodiff_cell_jacobian_from_residual(uh_to_cell_residual, uh, cellids)
        Gridap.FESpaces._push_matrix_contribution!(wu,ru,cu,cellvals_u,cellids)
    end

    #if :jac in fieldnames(typeof(term))
    #else
    function yh_to_cell_residual(yf)
      _yf = Gridap.FESpaces.restrict(yf, term.trian)
      integrate(term.res(vcat(_yf,_uh),_v), term.trian, term.quad)
    end
    cellvals_y = Gridap.FESpaces.autodiff_cell_jacobian_from_residual(yh_to_cell_residual, yh, cellids)
    #end
    Gridap.FESpaces._push_matrix_contribution!(wy,ry,cy,cellvals_y,cellids)

  end

  @warn "Decompose assemble_matrix! as it returns a dense matrix. Check Au, Ay."
  if nvar_control != 0
      assem_u = Gridap.FESpaces.SparseMatrixAssembler(nlp.Ycon, nlp.Xedp)
      Gridap.FESpaces.assemble_matrix!(Au, assem_u, (wu,ru,cu))
  end

  assem_y = Gridap.FESpaces.SparseMatrixAssembler(nlp.Yedp, nlp.Xedp)
  Gridap.FESpaces.assemble_matrix!(Ay, assem_y, (wy,ry,cy))

  edp_jac = hcat(sparse(Ay),sparse(Au))

  rows, cols, vals = Array{Int,1}(undef, 0), Array{Int,1}(undef, 0), Array{T,1}(undef, 0)
  if nlp.meta_func.ncon != 0
      _y, _u = _get_y_and_u(nlp, x)
      for i = 1 : nvar_per_field
       _jac = nlp.J(_y[:,i],_u[:,i]) #size nlp.meta_func.ncon x nfields
       vals = vcat(vals, _jac[:])
       rows = vcat(rows, repeat((i - 1) * nconf + 1: i * nconf, n_edp_fields + n_control_fields))
       cols = vcat(cols, repeat((i-1) * n_edp_fields + 1 : i * n_edp_fields, nconf),
                         repeat(nvar_edp + (i-1) * n_control_fields + 1 : nvar_edp + i * n_control_fields, nconf))
      end
  end
  res_jac = length(rows) != 0 ? sparse(rows, cols, vals) : zeros(T, 0, nvar_edp + nvar_control)

  return vcat(edp_jac, res_jac)
end

#The function jac_structure is not efficient as we don't have the structure
#of the first block.
#function jac_structure!(nlp :: ADNLPModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
#Tanj; Using this would be efficient enough :)
#Jv = jprod!(nlp, rows, cols, vals, v, Jv)

#Est-ce qu'on a besoin d'un 'function jjtprod()' quand on utilise la jacobienne explicitement?
#Je dirais bien oui, mais pas nécessaire si on compte utilise les LinearOperator
#J = jac_op!(nlp, rows, cols, vals, Jv, Jtv)
#Tanj: bref jprod et jtprod ne servent à rien.
#La démarche c'est plutôt d'avoir un jac!(triplet) :). -> J = jac_op!(nlp, rows, cols, vals, Jv, Jtv)
# ou direct jac_op!(nlp, x, Jv, Jtv) qui crée le triplet et appelle jac_op!(nlp, rows, cols, vals, Jv, Jtv).
#sparse(rows, cols, vals, nlp.meta.ncon, nlp.meta.nvar)

"""
    Jv = jprod!(nlp, x, v, Jv)
Evaluate ``J(x)v``, the Jacobian-vector product at `x` in place.

Note for GridapPDENLPModel:
- Evaluate the jacobian and then use coo_prod!.
- Alternative would be: Jv .= jac(nlp, x) * v
- Alternative2: benefit from the AD? Jv .= ForwardDiff.derivative(t->nlp.c(nlp, x + t * v), 0)
when the jacobian is obtained by AD.
"""
function jprod!(nlp :: GridapPDENLPModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  @lencheck nlp.meta.nvar x v
  @lencheck nlp.meta.ncon Jv
  increment!(nlp, :neval_jprod)

  Jx = jac(nlp, x)
  rows, cols, vals = findnz(Jx)
  coo_prod!(rows, cols, vals, v, Jv)

  return Jv
end

"""
    Jv = jtprod!(nlp, x, v, Jv)
Evaluate ``J(x)'v``, the Jacobian-vector product at `x` in place.

Note for GridapPDENLPModel:
- Evaluate the jacobian and then use coo_prod!.
- Alternative would be: Jv .= jac(nlp, x)' * v
- Alternative2: benefit from the AD? Jtv .= ForwardDiff.gradient(x -> dot(nlp.c(x), v), x)
when the jacobian is obtained by AD.
"""
function jtprod!(nlp :: GridapPDENLPModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  @lencheck nlp.meta.nvar x Jtv
  @lencheck nlp.meta.ncon v
  increment!(nlp, :neval_jtprod)

  Jx = jac(nlp, x)
  rows, cols, vals = findnz(Jx)
  coo_prod!(cols, rows, vals, v, Jtv)

  return Jtv
end

#ℓ(x) = obj_weight * nlp.f(x)
#Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x + t * v), 0)
function hprod!(nlp :: GridapPDENLPModel, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  n = nlp.meta.nvar
  @lencheck n x v Hv
  increment!(nlp, :neval_hprod)

  if obj_weight == zero(eltype(x))
      @warn "Why would you do that?"
      Hv .= zero(similar(x))
      return Hv
  end

  assem = Gridap.FESpaces.SparseMatrixAssembler(nlp.Y, nlp.X)
  #We prepare computation of x + t * v
  #At t=0.
  yu    = FEFunction(nlp.Y, x)
  vf    = FEFunction(nlp.Y, v)

  cell_yu   = Gridap.FESpaces.get_cell_values(yu)
  cell_vf   = Gridap.FESpaces.get_cell_values(vf)
  ncells    = length(cell_yu)
  cell_id = Gridap.Arrays.IdentityVector(length(cell_yu))

  function _cell_obj_t(cell)
       th = CellField(nlp.Y, cell)
      _th = Gridap.FESpaces.restrict(th, nlp.trian)
      integrate(nlp.f(_th), nlp.trian, nlp.quad) #function needs to return array of size 1.
  end

  #Compute the gradient with AD
  function _cell_grad_t(t)
      ct     = t * ones(length(cell_vf[1]))
      _cell  = Array{typeof(ct .* cell_vf[1])}(undef, ncells)
      for i=1:ncells
          _cell[i] = cell_yu[i] + ct .* cell_vf[i]
      end
      Gridap.Arrays.autodiff_array_gradient(_cell_obj_t, _cell, cell_id)
  end

  #Compute the derivative w.r.t. to t of _cell_grad_t
  #This might be slow as it cannot be decomposed (easily) cell by cell
  cell_r_yu = ForwardDiff.derivative(_cell_grad_t, 0.)

  #Put the result in the format expected by Gridap.FESpaces.assemble_matrix
  vecdata_yu = [[cell_r_yu], [cell_id]]
  #Assemble the gradient in the "good" space
  Hv .= Gridap.FESpaces.assemble_vector(assem, vecdata_yu)

  return Hv
end

#ℓ(x) = obj_weight * nlp.f(x) + dot(nlp.c(x), y)
#Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x + t * v), 0)
function hprod!(nlp  :: GridapPDENLPModel, x :: AbstractVector, λ :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Real = one(eltype(x)))

  @lencheck nlp.meta.nvar x v Hv
  @lencheck nlp.meta.ncon λ
  increment!(nlp, :neval_hprod)
  @warn "Almost there, what is not working?"

  λ_edp = λ[1:nlp.nvar_edp]
  λ_con = λ[nlp.nvar_edp + 1 : nlp.meta.ncon]

  #First part we compute with AD line by line.
  cons_residual = zero(Hv)
  if nlp.meta_func.ncon != 0
      for i = 1:nlp.nvar_per_field-1
          yi, ui, vi = _get_y_and_u_i(nlp, x, v, i)
          λ_coni = λ_con[nlp.meta_func.ncon * (i-1) + 1 : nlp.meta_func.ncon * i]
          function _c(_x)
              _y = _x[1:nlp.n_edp_fields]
              _u = _x[nlp.n_edp_fields + 1:nlp.n_edp_fields + nlp.n_control_fields]
              _res = dot(nlp.c(_y, _u), λ_coni)
          end
          _temp = ForwardDiff.derivative(t -> ForwardDiff.gradient(_c, vcat(yi, ui) + t * vi), 0.)
          cons_residual[(i - 1) * nlp.meta_func.nvar + 1: i * nlp.meta_func.nvar] = _temp
      end
  end

  #Second part: obj_weight * nlp.f(x) + dot(nlp.c(x), y)
  yu = FEFunction(nlp.Y, x)
  vf = FEFunction(nlp.Y, v)
  λf = FEFunction(nlp.Yedp, λ_edp)
  vc = Gridap.FESpaces.get_cell_basis(nlp.op.test) #Xedp

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
        _vc = restrict(vc,  term.trian) #One error is here: ERROR: AssertionError: length(a) == length(b)
        _yu = restrict(yu, term.trian)
        cellvals = integrate(term.res(_yu, _vc), term.trian, term.quad)
        for i=1:ncells
        _dotvalues[i] = dot(cellvals[i], cell_λf[i]) #setindex not defined for Gridap.Arrays
        end
      end
      _lag += _dotvalues
  end

  #Compute the gradient with AD
  function _cell_lag_grad_t(t)

       ct    = t * ones(length(cell_vf[1]))
      _cell  = Array{typeof(ct .* cell_vf[1])}(undef, ncells)
      for i=1:ncells
          _cell[i] = cell_yu[i] + ct .* cell_vf[i]
      end
      Gridap.Arrays.autodiff_array_gradient(_cell_lag_t, _cell, cell_id)
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

"""
`_get_y_and_u(:: GridapPDENLPModel, :: AbstractVector{T}) `

Returns y and u in matrix format where
y ∈ n_edp_fields x nvar_per_field
u ∈ n_control_fields x nvar_per_field
It is useful when evaluating the function constraint (and jacobian) functions.
"""
function _get_y_and_u(nlp :: GridapPDENLPModel, x :: AbstractVector{T}) where T

    y = Array{T,2}(undef, nlp.n_edp_fields, nlp.nvar_per_field)
    for i=1:nlp.n_edp_fields
        y[i,:] = [x[k] for k in i:nlp.n_edp_fields:nlp.nvar_edp]
    end

    u = Array{T,2}(undef, nlp.n_control_fields, nlp.nvar_per_field)
    for i=1:nlp.n_control_fields
        u[i,:] = [x[k] for k in nlp.nvar_edp+i:nlp.n_control_fields:nlp.meta.nvar-nlp.nparam]
    end

    return y, u
end

function _get_y_and_u_i(nlp :: GridapPDENLPModel, x :: AbstractVector{T}, j :: Int) where T

    y = Array{T,1}(undef, nlp.n_edp_fields)
    for i=1:nlp.n_edp_fields
        y[i] = x[(j-1)*nlp.n_edp_fields + i]
    end

    u = Array{T,1}(undef, nlp.n_control_fields)
    for i=1:nlp.n_control_fields
        u[i] = x[nlp.nvar_edp+(j-1)*nlp.n_control_fields + i]
    end

    return y, u
end

function _get_y_and_u_i(nlp :: GridapPDENLPModel, x :: AbstractVector{T}, v :: AbstractVector{T}, j :: Int) where T

    _v = Array{T,1}(undef, nlp.n_edp_fields + nlp.n_control_fields)

    y = Array{T,1}(undef, nlp.n_edp_fields)
    for i=1:nlp.n_edp_fields
        y[i] = x[(j-1)*nlp.n_edp_fields + i]
       _v[i] = v[(j-1)*nlp.n_edp_fields + i]
    end

    u = Array{T,1}(undef, nlp.n_control_fields)
    for i=1:nlp.n_control_fields
        u[i] = x[nlp.nvar_edp + (j-1) * nlp.n_control_fields + i]
       _v[i] = v[nlp.n_edp_fields + nlp.nvar_edp + (j-1) * nlp.n_control_fields + i]
    end

    return y, u, _v
end
