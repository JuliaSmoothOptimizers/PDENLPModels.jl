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

TODO:
[ ] time evolution pde problems.   
[ ] Handle the case where g and H are given.   
[ ] Handle several terms in the objective function (via an FEOperator)?   
[ ] Could we control the Dirichlet boundary condition? (like classical control of heat equations)     

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
 - We handle two types of FEOperator: AffineFEOperator, and FEOperatorFromTerms
 which is the obtained by the FEOperator constructor.
 The terms supported in FEOperatorFromTerms are: FESource, NonlinearFETerm,
 NonlinearFETermWithAutodiff, LinearFETerm, AffineFETerm.
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
end

include("bounds_function.jl")
include("additional_constructors.jl")

show_header(io::IO, nlp::GridapPDENLPModel) = println(io, "GridapPDENLPModel")

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

#=GRIDAPv15
function hess_yu_obj_structure!(
  nlp::GridapPDENLPModel,
  rows::AbstractVector,
  cols::AbstractVector;
  nfirst::Int = 0,
)

  # Special case as nlp.tnrj has no field trian.    
  if typeof(nlp.tnrj) <: NoFETerm
    return nfirst
  end

  a = Gridap.FESpaces.SparseMatrixAssembler(nlp.Y, nlp.X)
  ncells = num_cells(nlp.tnrj.trian)
  cell_id_yu = Gridap.Arrays.IdentityVector(ncells)

  nini = struct_hess_coo_numeric!(
    rows,
    cols,
    a,
    cell_id_yu,
    nfirst = nfirst,
    cols_translate = nlp.nparam,
    rows_translate = nlp.nparam,
  )

  return nini
end
=#

#=GRIDAPv15
# DO WE REALLY NEED THIS???
function hess_coord(
  nlp::GridapPDENLPModel,
  x::AbstractVector{T};
  obj_weight::Real = one(T),
) where {T}
  @lencheck nlp.meta.nvar x
  #########################################################################################
  #The issue here is that there is no meta specific for the obj only
  #vals = Vector{eltype(x)}(undef, nlp.meta.nnzh)

  # Special case as nlp.tnrj has no field trian.    
  if typeof(nlp.tnrj) <: NoFETerm
    nnz_hess_yu = 0
  else
    a = Gridap.FESpaces.SparseMatrixAssembler(nlp.Y, nlp.X)
    ncells = num_cells(nlp.tnrj.trian)
    cell_id_yu = Gridap.Arrays.IdentityVector(ncells)
    nnz_hess_yu = count_hess_nnz_coo_short(a, cell_id_yu)
  end

  #add the nnz w.r.t. k; by default it is:
  if (typeof(nlp.tnrj) <: MixedEnergyFETerm && nlp.tnrj.inde) || typeof(nlp.tnrj) <: NoFETerm
    nnz_hess_k = Int(nlp.nparam * (nlp.nparam + 1) / 2)
  else
    nnz_hess_k = Int(nlp.nparam * (nlp.nparam + 1) / 2) + (nlp.meta.nvar - nlp.nparam) * nlp.nparam
  end

  nnzh = nnz_hess_yu + nnz_hess_k
  #USE get_nnzh here
  #########################################################################################
  vals = vcat(Vector{T}(undef, nnzh), zeros(T, nlp.meta.nnzh - nnzh))

  return hess_coord!(nlp, x, vals; obj_weight = obj_weight)
end
=#

# Shouldn't this be in the NRJ terms ??
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

  #Right now V1 cannot be computed separately
  if (typeof(nlp.tnrj) <: MixedEnergyFETerm && nlp.tnrj.inde) || typeof(nlp.tnrj) <: NoFETerm
    nnz_hess_k = Int(nlp.nparam * (nlp.nparam + 1) / 2)
  else
    nnz_hess_k = Int(nlp.nparam * (nlp.nparam + 1) / 2) + (nlp.meta.nvar - nlp.nparam) * nlp.nparam
  end
  nini = nnz_hess_k
  vals[1:nnz_hess_k] .= _compute_hess_k_vals(nlp, nlp.tnrj, κ, xyu)

  if typeof(nlp.tnrj) != NoFETerm
    a = Gridap.FESpaces.SparseMatrixAssembler(nlp.Y, nlp.X)
    ncells = num_cells(nlp.tnrj.trian)
    cell_id_yu = Gridap.Arrays.IdentityVector(ncells)

    cell_r_yu = if nlp.nparam > 0
      get_array(hessian(x -> nlp.tnrj.f(κ, x), yu))
    else
      get_array(hessian(nlp.tnrj.f, yu))
    end
    nini = vals_hess_coo_numeric!(vals, a, cell_r_yu, cell_id_yu, nfirst = nini)
  end
  vals .*= obj_weight
  return vals
end


#=GRIDAPv15
function hprod!(
  nlp::GridapPDENLPModel,
  x::AbstractVector{T},
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight::Real = one(T),
) where {T}
  @lencheck nlp.meta.nvar x v Hv
  increment!(nlp, :neval_hprod)

  if obj_weight == zero(T)
    Hv .= zero(T)
    return Hv
  end

  rows, cols = hess_structure(nlp)
  vals = hess_coord(nlp, x, obj_weight = obj_weight)
  decrement!(nlp, :neval_hess)

  coo_sym_prod!(cols, rows, vals, v, Hv)

  return Hv
end

function hess_structure!(
  nlp::GridapPDENLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck nlp.meta.nnzh rows cols
  n, p = nlp.meta.nvar, nlp.nparam
  prows = n
  nnz_hess_k =
    if (typeof(nlp.tnrj) <: MixedEnergyFETerm && nlp.tnrj.inde) || typeof(nlp.tnrj) <: NoFETerm
      prows = p
      Int(p * (p + 1) / 2)
    else
      Int(p * (p + 1) / 2) + (n - p) * p
    end
  #nnz_hess_k = Int(p * (p + 1) / 2) + (n - p) * p
  I = ((i, j) for i = 1:prows, j = 1:p if j ≤ i)
  rows[1:nnz_hess_k] .= getindex.(I, 1)[:]
  cols[1:nnz_hess_k] .= getindex.(I, 2)[:]

  nini = hess_yu_obj_structure!(nlp, rows, cols, nfirst = nnz_hess_k)

  if nlp.meta.ncon > 0
    ##########################################################
    # Isolate in a function
    if typeof(nlp.op) <: Gridap.FESpaces.FEOperatorFromWeakForm
      for term in nlp.op.terms
        if !(
          typeof(term) <:
          Union{Gridap.FESpaces.NonlinearFETermWithAutodiff, Gridap.FESpaces.NonlinearFETerm}
        )
          continue
        end

        a = Gridap.FESpaces.SparseMatrixAssembler(nlp.Y, nlp.X)
        ncells = num_cells(term.trian)
        cell_id_yu = Gridap.Arrays.IdentityVector(ncells)

        nini = struct_hess_coo_numeric!(
          rows,
          cols,
          a,
          cell_id_yu,
          nfirst = nini,
          cols_translate = nlp.nparam,
          rows_translate = nlp.nparam,
        )
      end
    end
    ##########################################################

    nnz_hess_k = Int(p * (p + 1) / 2) + (n - p) * p
    I = ((i, j) for i = 1:n, j = 1:p if j ≤ i)
    rows[(nini + 1):(nini + nnz_hess_k)] .= getindex.(I, 1)[:]
    cols[(nini + 1):(nini + nnz_hess_k)] .= getindex.(I, 2)[:]
    nini += nnz_hess_k
  end

  if nlp.meta.nnzh != nini
    @warn "hess_structure!: Size of vals and number of assignements didn't match $(nlp.meta.nnzh) vs $(nini)"
  end

  (rows, cols)
end
=#

#=GRIDAPv15
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

  nnzh_obj = get_nnzh(nlp.tnrj, nlp.Y, nlp.X, nlp.nparam, nlp.meta.nvar)
  hess_coord!(nlp, x, vals, obj_weight = obj_weight) # @view vals[1:nnzh_obj]
  decrement!(nlp, :neval_hess)

  κ, xyu = x[1:(nlp.nparam)], x[(nlp.nparam + 1):(nlp.meta.nvar)]
  yu = FEFunction(nlp.Y, xyu)

  cell_yu = Gridap.FESpaces.get_cell_dof_values(yu)
  cell_id_yu = Gridap.Arrays.IdentityVector(length(cell_yu)) #Is it?

  nini = nnzh_obj
  ##########################################################
  # Isolate in a function
  if typeof(nlp.op) <: Gridap.FESpaces.FEOperatorFromWeakForm
    for term in nlp.op.terms
      if !(
        typeof(term) <:
        Union{Gridap.FESpaces.NonlinearFETermWithAutodiff, Gridap.FESpaces.NonlinearFETerm}
      )
        continue
      end

      λf = FEFunction(nlp.Xpde, λ)
      cell_λf = Gridap.FESpaces.get_cell_dof_values(λf)
      lfh = CellField(nlp.Xpde, cell_λf)
      _lfh = Gridap.FESpaces.restrict(lfh, term.trian) #This is where the term play a first role.

      function _cell_res_yu(cell)
        xfh = CellField(nlp.Y, cell)
        _xfh = Gridap.FESpaces.restrict(xfh, term.trian)

        if length(κ) > 0
          _res = integrate(term.res(κ, _xfh, _lfh), term.quad)
        else
          _res = integrate(term.res(_xfh, _lfh), term.quad)
        end
        lag = _res
        return lag
      end

      #ncells     = num_cells(term.trian)
      #cell_id_yu = Gridap.Arrays.IdentityVector(ncells)

      #Compute the hessian with AD
      cell_r_yu = Gridap.Arrays.autodiff_array_hessian(_cell_res_yu, cell_yu, cell_id_yu)
      #Assemble the matrix in the "good" space
      assem = Gridap.FESpaces.SparseMatrixAssembler(nlp.Y, nlp.X)
      #(I, J, V)  = assemble_hess(assem, cell_r_yu, cell_id_yu)
      nini = vals_hess_coo_numeric!(vals, assem, cell_r_yu, cell_id_yu, nfirst = nini)
      #vals[nini+1:nini+length(V)] .= V
      #nini += length(V)
    end
  end

  if nlp.nparam > 0
    p, n = nlp.nparam, nlp.meta.nvar
    nnz_hess_k = Int(p * (p + 1) / 2) + (n - p) * p
    κ, xyu = x[1:(nlp.nparam)], x[(nlp.nparam + 1):(nlp.meta.nvar)]
    function ℓ(x, λ)
      c = similar(x, nlp.meta.ncon)
      _from_terms_to_residual!(nlp.op, x, nlp, c)
      return dot(c, λ)
    end
    agrad = @closure k -> ForwardDiff.gradient(x -> ℓ(x, λ), vcat(k, xyu))
    Hk = ForwardDiff.jacobian(agrad, κ)
    vals[(nini + 1):(nini + nnz_hess_k)] .= [Hk[i, j] for i = 1:n, j = 1:p if j ≤ i]
    nini += nnz_hess_k
  end

  return vals
end
=#

function cons!(nlp::GridapPDENLPModel, x::AbstractVector, c::AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon c
  increment!(nlp, :neval_cons)

  _from_terms_to_residual!(nlp.op, x, nlp, c)

  return c
end

function _from_terms_to_residual!(
  op::Gridap.FESpaces.FEOperatorFromWeakForm,
  x::AbstractVector,
  nlp::GridapPDENLPModel,
  res::AbstractVector,
)
  κ, xyu = x[1:(nlp.nparam)], x[(nlp.nparam + 1):(nlp.meta.nvar)]
  yu = FEFunction(nlp.Y, xyu)

  #=GRIDAPv15
  v = Gridap.FESpaces.get_cell_basis(nlp.Xpde) #Tanj: is it really Xcon ?

  w, r = [], []
  for term in op.terms
    w, r = _from_term_to_terms!(term, κ, yu, v, w, r)
  end

  assem_y = Gridap.FESpaces.SparseMatrixAssembler(nlp.Ypde, nlp.Xpde)
  Gridap.FESpaces.assemble_vector!(res, assem_y, (w, r))
  =#

  # Gridap.FESpaces.residual(nlp.op, FEFunction(nlp.Y, x))
  # Split the call of: b = allocate_residual(op, u)
  V = Gridap.FESpaces.get_test(op)
  v = Gridap.FESpaces.get_cell_shapefuns(V)
  if nlp.nparam == 0
    vecdata = Gridap.FESpaces.collect_cell_vector(op.res(yu, v))
  else
    vecdata = Gridap.FESpaces.collect_cell_vector(op.res(κ, yu, v))
  end
  # res = allocate_vector(op.assem, vecdata) # already done somewhere
  # Split the call of: residual!(b,op,u)
  Gridap.FESpaces.assemble_vector!(res, op.assem, vecdata)

  return res
end

#=GRIDAPv15
function _from_term_to_terms!(
  term::Union{Gridap.FESpaces.NonlinearFETermWithAutodiff, Gridap.FESpaces.NonlinearFETerm},
  κ::AbstractVector,
  yu::FEFunctionType,
  v::CellFieldType,
  w::AbstractVector,
  r::AbstractVector,
)
  _v = restrict(v, term.trian)
  _yu = restrict(yu, term.trian)

  if length(κ) > 0
    cellvals = integrate(term.res(κ, _yu, _v), term.quad)
  else
    cellvals = integrate(term.res(_yu, _v), term.quad)
  end
  cellids = Gridap.FESpaces.get_cell_id(term)

  Gridap.FESpaces._push_vector_contribution!(w, r, cellvals, cellids)

  return w, r
end
=#

#=GRIDAPv15
function _from_term_to_terms!(
  term::Gridap.FESpaces.FETerm, #FESource, AffineFETerm
  κ::AbstractVector,
  yu::FEFunctionType,
  v::CellFieldType,
  w::AbstractVector,
  r::AbstractVector,
)
  cellvals = Gridap.FESpaces.get_cell_residual(term, yu, v)
  cellids = Gridap.FESpaces.get_cell_id(term)

  Gridap.FESpaces._push_vector_contribution!(w, r, cellvals, cellids)

  return w, r
end
=#

#=
Note:
- mul! seems faster than doing:
rows, cols, vals = findnz(get_matrix(op))
coo_prod!(cols, rows, vals, v, res)

- get_matrix(op) is a sparse matrix
- Benchmark equivalent to Gridap.FESpaces.residual!(res, op_affine.op, xrand)
=#
function _from_terms_to_residual!(
  op::AffineFEOperator,
  x::AbstractVector,
  nlp::GridapPDENLPModel,
  res::AbstractVector,
)
  T = eltype(x)
  mul!(res, get_matrix(op), x)
  axpy!(-one(T), get_vector(op), res)

  return res
end

function jac(nlp::GridapPDENLPModel, x::AbstractVector{T}) where {T}
  @lencheck nlp.meta.nvar x
  increment!(nlp, :neval_jac)

  pde_jac = _from_terms_to_jacobian(nlp.op, x, nlp.Y, nlp.Xpde, nlp.Ypde, nlp.Ycon)

  if nlp.nparam > 0
    κ, xyu = x[1:(nlp.nparam)], x[(nlp.nparam + 1):(nlp.meta.nvar)]
    function _cons(nlp, xyu, k)
      c = similar(k, nlp.meta.ncon)
      _from_terms_to_residual!(nlp.op, vcat(k, xyu), nlp, c)
      return c
    end
    ck = @closure k -> _cons(nlp, xyu, k)
    jac_k = ForwardDiff.jacobian(ck, κ)
    return hcat(jac_k, pde_jac)
  end

  return pde_jac
end

function jprod!(nlp::GridapPDENLPModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck nlp.meta.nvar x v
  @lencheck nlp.meta.ncon Jv
  increment!(nlp, :neval_jprod)

  Jx = jac(nlp, x)
  decrement!(nlp, :neval_jac)
  mul!(Jv, Jx, v)

  return Jv
end

function jtprod!(nlp::GridapPDENLPModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  @lencheck nlp.meta.nvar x Jtv
  @lencheck nlp.meta.ncon v
  increment!(nlp, :neval_jtprod)

  Jx = jac(nlp, x)
  decrement!(nlp, :neval_jac)
  mul!(Jtv, Jx', v)

  return Jtv
end

function jac_structure!(
  nlp::GridapPDENLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck nlp.meta.nnzj rows cols
  return _jac_structure!(nlp.op, nlp, rows, cols)
end

function _jac_structure!(
  op::AffineFEOperator,
  nlp::GridapPDENLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)

  #In this case, the jacobian matrix is constant:
  I, J, V = findnz(get_matrix(op))
  rows .= I
  cols .= J

  return rows, cols
end

#=GRIDAPv15
#Adaptation of `function allocate_matrix(a::SparseMatrixAssembler,matdata) end` in Gridap.FESpaces.
function _jac_structure!(
  op::Gridap.FESpaces.FEOperatorFromWeakForm,
  nlp::GridapPDENLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  nini = jac_k_structure!(nlp, rows, cols)
  nini = allocate_coo_jac!(
    op,
    nlp.Y,
    nlp.Xpde,
    nlp.Ypde,
    nlp.Ycon,
    rows,
    cols,
    nfirst = nini,
    nparam = nlp.nparam,
  )

  return rows, cols
end
=#

#=GRIDAPv15
function jac_k_structure!(
  nlp::GridapPDENLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  p = nlp.nparam
  n = nlp.meta.ncon
  nnz_jac_k = p * n
  I = ((i, j) for i = 1:n, j = 1:p)
  rows[1:nnz_jac_k] .= getindex.(I, 1)[:]
  cols[1:nnz_jac_k] .= getindex.(I, 2)[:]

  return nnz_jac_k
end
=#

function jac_coord!(nlp::GridapPDENLPModel, x::AbstractVector, vals::AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnzj vals
  increment!(nlp, :neval_jac)
  return _jac_coord!(nlp.op, nlp, x, vals)
end

function _jac_coord!(
  op::AffineFEOperator,
  nlp::GridapPDENLPModel,
  x::AbstractVector,
  vals::AbstractVector,
)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnzj vals
  increment!(nlp, :neval_jac)
  I, J, V = findnz(get_matrix(op))
  vals .= V
  return vals
end

#=GRIDAPv15
function _jac_coord!(
  op::Gridap.FESpaces.FEOperatorFromWeakForm,
  nlp::GridapPDENLPModel,
  x::AbstractVector{T},
  vals::AbstractVector,
) where {T}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnzj vals
  increment!(nlp, :neval_jac)

  nnz_jac_k = nlp.nparam > 0 ? nlp.meta.ncon * nlp.nparam : 0
  if nlp.nparam > 0
    κ, xyu = x[1:(nlp.nparam)], x[(nlp.nparam + 1):(nlp.meta.nvar)]
    function _cons(nlp, xyu, k)
      c = similar(k, nlp.meta.ncon)
      _from_terms_to_residual!(nlp.op, vcat(k, xyu), nlp, c)
      return c
    end
    ck = @closure k -> _cons(nlp, xyu, k)
    jac_k = ForwardDiff.jacobian(ck, κ)
    vals[1:nnz_jac_k] .= jac_k[:]
  end

  nini = _from_terms_to_jacobian_vals!(
    nlp.op,
    x,
    nlp.Y,
    nlp.Xpde,
    nlp.Ypde,
    nlp.Ycon,
    vals,
    nfirst = nnz_jac_k,
  )

  if nlp.meta.nnzj != nini
    @warn "hess_coord!: Size of vals and number of assignements didn't match $(nlp.meta.nnzh) vs $(nini)"
  end
  return vals
end
=#

#=GRIDAPv15
function hprod!(
  nlp::GridapPDENLPModel,
  x::AbstractVector,
  λ::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight::Real = one(eltype(x)),
)
  @lencheck nlp.meta.nvar x v Hv
  @lencheck nlp.meta.ncon λ
  increment!(nlp, :neval_hprod)

  #λ_pde = λ[1:nlp.nvar_pde]
  #λ_con = λ[nlp.nvar_pde+1:nlp.meta.ncon]
  # _from_terms_to_hprod!(nlp.op, x, λ_pde, v, nlp, Hv, obj_weight)

  rows, cols = hess_structure(nlp)
  vals = hess_coord(nlp, x, λ, obj_weight = obj_weight)
  decrement!(nlp, :neval_hess)
  coo_sym_prod!(cols, rows, vals, v, Hv)

  return Hv
end
=#