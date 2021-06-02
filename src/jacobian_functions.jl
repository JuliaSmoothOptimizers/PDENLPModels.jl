#=
This is a modified version of
`function fill_matrix_coo_symbolic!(I,J,a::GenericSparseMatrixAssembler,matdata,n=0) end`
from Gridap.FESpaces.
The motivation is to avoid the use of the yet unknown values.
=#
function fill_jac_coo_symbolic!(
  I,
  J,
  a::Gridap.FESpaces.GenericSparseMatrixAssembler,
  cellidmatdata;
  n = 0,
)
  term_to_cellidsrows, term_to_cellidscols = cellidmatdata
  nini = n
  for (cellidsrows, cellidscols) in zip(term_to_cellidsrows, term_to_cellidscols)
    cell_rows = Gridap.FESpaces.get_cell_dofs(a.test, cellidsrows)
    cell_cols = Gridap.FESpaces.get_cell_dofs(a.trial, cellidscols)
    rows_cache = Gridap.FESpaces.array_cache(cell_rows)
    cols_cache = Gridap.FESpaces.array_cache(cell_cols)

    #In the unconstrained case: cellmat = cell_r_yu
    #cellmat_r = attach_constraints_cols(a.trial,cellmat_rc,cellidscols)
    #cellmat = attach_constraints_rows(a.test,cellmat_r,cellidsrows)
    #@assert length(cell_cols) == length(cell_rows)

    if length(cell_cols) > 0
      #mat = first(cellmat)
      Is = nothing #_get_block_layout(mat)
      nini = Gridap.FESpaces._allocate_matrix!(
        a.matrix_type,
        nini,
        I,
        J,
        rows_cache,
        cols_cache,
        cell_rows,
        cell_cols,
        a.strategy,
        Is,
      )
    end
  end
  nini
end

#=GRIDAPv15
function allocate_coo_jac!(
  op::Gridap.FESpaces.FEOperatorFromWeakForm,
  Y::FESpace,
  Xpde::FESpace,
  Ypde::FESpace,
  Ycon::FESpace,
  rows,
  cols;
  nfirst = 0,
  nparam = 0,
) where {T}
  ru, ry = [], []
  cu, cy = [], []
  r, c = [], []

  nvar_pde = num_free_dofs(Ypde)

  for term in op.terms
    _jac_from_term_to_terms_id!(term, r, c, ru, cu, ry, cy)
  end

  nini = nfirst

  assem_y = Gridap.FESpaces.SparseMatrixAssembler(Ypde, Xpde)
  ny = count_nnz_coo_short(assem_y, (ry, cy))
  Iy, Jy = allocate_coo_vectors_IJ(Gridap.FESpaces.get_matrix_type(assem_y), ny)
  nini = fill_jac_coo_symbolic!(rows, cols, assem_y, (ry, cy), n = nini)
  cols[(nfirst + 1):(nfirst + ny)] .+= nparam

  if Ycon != VoidFESpace()
    assem_u = Gridap.FESpaces.SparseMatrixAssembler(Ycon, Xpde)
    nu = count_nnz_coo_short(assem_u, (ru, cu))
    nini = fill_jac_coo_symbolic!(rows, cols, assem_u, (ru, cu), n = nini)
    cols[(ny + 1):(ny + nu)] .+= nparam + nvar_pde #translate the columns
  else
    nu = 0
  end

  assem = Gridap.FESpaces.SparseMatrixAssembler(Y, Xpde)
  nyu = count_nnz_coo_short(assem, (r, c))
  Iyu, Jyu = allocate_coo_vectors_IJ(Gridap.FESpaces.get_matrix_type(assem), nyu)
  nini = fill_jac_coo_symbolic!(rows, cols, assem, (r, c), n = nini)
  cols[(ny + nu + 1):(ny + nu + nyu)] .+= nparam

  return nini
end
=#

#=
This is a modified version of
`function count_matrix_nnz_coo(a::GenericSparseMatrixAssembler,matdata) end`
from Gridap.FESpaces.
The motivation is to avoid the use of the yet unknown values.
=#
function count_nnz_coo_short(a::Gridap.FESpaces.GenericSparseMatrixAssembler, cellidmatdata)
  n = 0
  for (cellidsrows, cellidscols) in zip(cellidmatdata...)
    cell_rows = Gridap.FESpaces.get_cell_dofs(a.test, cellidsrows)
    cell_cols = Gridap.FESpaces.get_cell_dofs(a.trial, cellidscols)
    rows_cache = Gridap.FESpaces.array_cache(cell_rows)
    cols_cache = Gridap.FESpaces.array_cache(cell_cols)

    #In the unconstrained case: cellmat = cell_r_yu
    #cellmat_r = attach_constraints_cols(a.trial,cellmat_rc,cellidscols)
    #cellmat = attach_constraints_rows(a.test,cellmat_r,cellidsrows)

    if length(cell_cols) > 0
      #mat = first(cellmat)
      Is = nothing #_get_block_layout(mat) ##Is = nothing if cellmat is a matrix
      n += Gridap.FESpaces._count_matrix_entries(
        a.matrix_type,
        rows_cache,
        cols_cache,
        cell_rows,
        cell_cols,
        a.strategy,
        Is,
      )
    end
  end
  n
end

function count_nnz_jac(
  op::Gridap.FESpaces.FEOperatorFromWeakForm,
  Y::FESpace,
  Xpde::FESpace,
  Ypde::FESpace,
  Ycon::FESpace,
) where {T}
  ru, ry = [], []
  cu, cy = [], []
  r, c = [], []

  # https://github.com/gridap/Gridap.jl/blob/9bbf92203d411e6dd9c22bf32e71f58733613c7a/src/FESpaces/FEOperatorsFromWeakForm.jl#L65
  Y = Gridap.FESpaces.get_trial(op)
  uh = FEFunction(Y, zeros(Gridap.FESpaces.num_free_dofs(Y)))
  A = Gridap.FESpaces.allocate_jacobian(op, uh)

  #=GRIDAPv15
  for term in op.terms
    _jac_from_term_to_terms_id!(term, r, c, ru, cu, ry, cy)
  end

  nini = 0
  if Ycon != VoidFESpace()
    assem_u = Gridap.FESpaces.SparseMatrixAssembler(Ycon, Xpde)
    nini += count_nnz_coo_short(assem_u, (ru, cu))
  end

  assem_y = Gridap.FESpaces.SparseMatrixAssembler(Ypde, Xpde)
  nini += count_nnz_coo_short(assem_y, (ry, cy))

  assem = Gridap.FESpaces.SparseMatrixAssembler(Y, Xpde)
  nini += count_nnz_coo_short(assem, (r, c))
  =#

  return nnz(A)
end

function count_nnz_jac(
  op::AffineFEOperator,
  Y::FESpace,
  Xpde::FESpace,
  Ypde::FESpace,
  Ycon::FESpace,
)
  return nnz(get_matrix(op))
end

function _from_terms_to_jacobian(
  op::AffineFEOperator,
  x::AbstractVector{T},
  Y::FESpace,
  Xpde::FESpace,
  Ypde::FESpace,
  Ycon::FESpace,
) where {T <: Number}
  return get_matrix(op)
end

function _from_terms_to_jacobian_vals!(
  op::AffineFEOperator,
  x::AbstractVector{T},
  Y::FESpace,
  Xpde::FESpace,
  Ypde::FESpace,
  Ycon::FESpace,
  vals::AbstractVector{T};
  nfirst::Integer = 0,
) where {T <: Number}
  nini = length(get_matrix(op).nzval)
  vals[(nfirst + 1):(nfirst + nini)] .= get_matrix(op).nzval
  return nfirst + nini
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
function _from_terms_to_jacobian(
  op::Gridap.FESpaces.FEOperatorFromWeakForm,
  x::AbstractVector{T},
  Y::FESpace,
  Xpde::FESpace,
  Ypde::FESpace,
  Ycon::FESpace,
) where {T <: Number}
  nvar = length(x)
  nyu = num_free_dofs(Y)
  nparam = nvar - nyu
  yh, uh = _split_FEFunction(x, Ypde, Ycon)
  κ, xyu = x[1:nparam], x[(nparam + 1):nvar]
  yu = FEFunction(Y, xyu)

#######################"
# WORKS WITHOUT CONTROL
  # Gridap.FESpaces.jacobian(nlp.op, FEFunction(nlp.Y, x))
  # Split the call of: b = allocate_jacobian(op, u)
  du = Gridap.FESpaces.get_cell_shapefuns_trial(Gridap.FESpaces.get_trial(op))
  v = Gridap.FESpaces.get_cell_shapefuns(Gridap.FESpaces.get_test(op))
  
  if nparam == 0
    matdata = Gridap.FESpaces.collect_cell_matrix(op.jac(yu, du, v))
  else
    matdata = Gridap.FESpaces.collect_cell_matrix(op.jac(κ, yu, du, v))
  end
  Ay = Gridap.FESpaces.allocate_matrix(op.assem, matdata)
  # res = allocate_vector(op.assem, vecdata) # already done somewhere
  # Split the call of: jacobian!(A,op,u)
  Gridap.FESpaces.assemble_matrix!(Ay, op.assem, matdata)
#######################"
#######################"

  if Ycon != VoidFESpace()
    @warn "jac doesn't work with control"
    # assem_u = Gridap.FESpaces.SparseMatrixAssembler(Ycon, Xpde)
    # Au = Gridap.FESpaces.assemble_matrix(assem_u, (wu, ru, cu))
  else
    Au = zeros(Gridap.FESpaces.num_free_dofs(Ypde), 0)
  end

  #=GRIDAPv15
  dy = Gridap.FESpaces.get_cell_basis(Ypde)
  du = Ycon != VoidFESpace() ? Gridap.FESpaces.get_cell_basis(Ycon) : nothing #use only jac is furnished
  dyu = Gridap.FESpaces.get_cell_basis(Y)
  v = Gridap.FESpaces.get_cell_basis(Xpde)

  wu, wy = [], []
  ru, ry = [], []
  cu, cy = [], []
  w, r, c = [], [], []

  for term in op.terms
    _jac_from_term_to_terms!(term, κ, yu, yh, uh, dyu, dy, du, v, w, r, c, wu, ru, cu, wy, ry, cy)
  end

  assem_y = Gridap.FESpaces.SparseMatrixAssembler(Ypde, Xpde)
  Ay = Gridap.FESpaces.assemble_matrix(assem_y, (wy, ry, cy))

  if Ycon != VoidFESpace()
    assem_u = Gridap.FESpaces.SparseMatrixAssembler(Ycon, Xpde)
    Au = Gridap.FESpaces.assemble_matrix(assem_u, (wu, ru, cu))
  else
    Au = zeros(Gridap.FESpaces.num_free_dofs(Ypde), 0)
  end

  S = hcat(Ay, Au)

  assem = Gridap.FESpaces.SparseMatrixAssembler(Y, Xpde)
  #doesn't work as we may not have the good sparsity pattern.
  #Gridap.FESpaces.assemble_matrix_add!(S, assem, (w, r, c))
  S += Gridap.FESpaces.assemble_matrix(assem, (w, r, c))
  =#
  S = hcat(Ay, Au)

  return S
end

#=GRIDAPv15
function _from_terms_to_jacobian_vals!(
  op::Gridap.FESpaces.FEOperatorFromWeakForm,
  x::AbstractVector{T},
  Y::FESpace,
  Xpde::FESpace,
  Ypde::FESpace,
  Ycon::FESpace,
  vals::AbstractVector{T};
  nfirst::Integer = 0,
) where {T <: Number}
  nvar = length(x)
  nyu = num_free_dofs(Y)
  nparam = nvar - nyu
  yh, uh = _split_FEFunction(x, Ypde, Ycon)
  κ, xyu = x[1:nparam], x[(nparam + 1):nvar]
  yu = FEFunction(Y, xyu)

  dy = Gridap.FESpaces.get_cell_basis(Ypde)
  du = Ycon != VoidFESpace() ? Gridap.FESpaces.get_cell_basis(Ycon) : nothing #use only jac is furnished
  dyu = Gridap.FESpaces.get_cell_basis(Y)
  v = Gridap.FESpaces.get_cell_basis(Xpde)

  wu, wy = [], []
  ru, ry = [], []
  cu, cy = [], []
  w, r, c = [], [], []

  for term in op.terms
    _jac_from_term_to_terms!(term, κ, yu, yh, uh, dyu, dy, du, v, w, r, c, wu, ru, cu, wy, ry, cy)
  end
  nini = nfirst

  assem_y = Gridap.FESpaces.SparseMatrixAssembler(Ypde, Xpde)
  nini = assemble_jac_vals!(vals, assem_y, (wy, ry, cy), n = nini)

  if Ycon != VoidFESpace()
    assem_u = Gridap.FESpaces.SparseMatrixAssembler(Ycon, Xpde)
    nini = assemble_jac_vals!(vals, assem_u, (wu, ru, cu), n = nini)
  end

  assem = Gridap.FESpaces.SparseMatrixAssembler(Y, Xpde)
  nini = assemble_jac_vals!(vals, assem, (w, r, c), n = nini)

  return nini
end
=#

#=
Adaptation of
`function assemble_matrix_add!(mat,a::GenericSparseMatrixAssembler,matdata) end`
from Gridap.FESpaces
=#
function assemble_jac_vals!(mat, a::Gridap.FESpaces.GenericSparseMatrixAssembler, matdata; n = 0)
  nini = n
  for (cellmat_rc, cellidsrows, cellidscols) in zip(matdata...)
    cell_rows = Gridap.FESpaces.get_cell_dofs(a.test, cellidsrows)
    cell_cols = Gridap.FESpaces.get_cell_dofs(a.trial, cellidscols)
    cellmat_r = Gridap.FESpaces.attach_constraints_cols(a.trial, cellmat_rc, cellidscols)
    cell_vals = Gridap.FESpaces.attach_constraints_rows(a.test, cellmat_r, cellidsrows)
    rows_cache = Gridap.FESpaces.array_cache(cell_rows)
    cols_cache = Gridap.FESpaces.array_cache(cell_cols)
    vals_cache = Gridap.FESpaces.array_cache(cell_vals)
    @assert length(cell_cols) == length(cell_rows)
    @assert length(cell_vals) == length(cell_rows)
    nini = _assemble_jac!(
      mat,
      vals_cache,
      rows_cache,
      cols_cache,
      cell_vals,
      cell_rows,
      cell_cols,
      a.strategy,
      n = nini,
    )
  end
  nini
end

@noinline function _assemble_jac!(
  mat,
  vals_cache,
  rows_cache,
  cols_cache,
  cell_vals,
  cell_rows,
  cell_cols,
  strategy;
  n = 0,
)
  nini = n
  for cell = 1:length(cell_cols)
    rows = getindex!(rows_cache, cell_rows, cell)
    cols = getindex!(cols_cache, cell_cols, cell)
    vals = getindex!(vals_cache, cell_vals, cell)
    nini = _assemble_jac_at_cell!(mat, rows, cols, vals, strategy, n = nini)
  end
  nini
end

@inline function _assemble_jac_at_cell!(mat, rows, cols, vals, strategy; n = 0)
  for (j, gidcol) in enumerate(cols)
    if gidcol > 0 && Gridap.FESpaces.col_mask(strategy, gidcol)
      _gidcol = Gridap.FESpaces.col_map(strategy, gidcol)
      for (i, gidrow) in enumerate(rows)
        if gidrow > 0 && Gridap.FESpaces.row_mask(strategy, gidrow)
          _gidrow = Gridap.FESpaces.row_map(strategy, gidrow)
          n += 1
          mat[n] = vals[i, j]
          #v = vals[i,j]
          #add_entry!(mat,v,_gidrow,_gidcol)
        end
      end
    end
  end
  n
end

#https://github.com/gridap/Gridap.jl/blob/758a8620756e164ba0e6b83dc8dcbb278015b3d9/src/FESpaces/SparseMatrixAssemblers.jl#L242
import Gridap.FESpaces._get_block_layout
#Tanj: this is an error when the jacobian matrix are of size 1xn.
#unit test: poinsson-with-Neumann-and-Dirichlet, l. 160.
function _get_block_layout(a::AbstractArray)
  nothing
end

#=
function _jac_from_term_to_terms!(
  term::Gridap.FESpaces.FETerm,
  κ::AbstractVector,
  yu::FEFunctionType,
  yh::FEFunctionType,
  uh::Union{FEFunctionType, Nothing},
  dyu::CellFieldType,
  dy::CellFieldType,
  du::Union{CellFieldType, Nothing},
  v::CellFieldType,
  w::AbstractVector,
  r::AbstractVector,
  c::AbstractVector,
  wu::AbstractVector,
  ru::AbstractVector,
  cu::AbstractVector,
  wy::AbstractVector,
  ry::AbstractVector,
  cy::AbstractVector,
)
  @warn "_jac_from_term_to_terms!(::FETerm, ...): If that works, good for you."

  cellvals = get_cell_jacobian(term, yu, dyu, v)
  cellids = get_cell_id(term)
  _push_matrix_contribution!(w, r, c, cellvals, cellids)
end
=#

#=
function _jac_from_term_to_terms_id!(
  term::Gridap.FESpaces.FETerm,
  r::AbstractVector,
  c::AbstractVector,
  ru::AbstractVector,
  cu::AbstractVector,
  ry::AbstractVector,
  cy::AbstractVector,
)
  @warn "_jac_from_term_to_terms_id!(::FETerm, ...): If that works, good for you."

  cellids = get_cell_id(term)
  w = [] #just to reuse Gridap functions
  Gridap.FESpaces._push_matrix_contribution!(w, r, c, [], cellids)
end
=#

#https://github.com/gridap/Gridap.jl/blob/758a8620756e164ba0e6b83dc8dcbb278015b3d9/src/FESpaces/FETerms.jl#L367
#=GRIDAPv15
function _jac_from_term_to_terms!(
  term::Union{Gridap.FESpaces.LinearFETerm, Gridap.FESpaces.AffineFETermFromIntegration},
  κ::AbstractVector,
  yu::FEFunctionType,
  yh::FEFunctionType,
  uh::Union{FEFunctionType, Nothing},
  dyu::CellFieldType,
  dy::CellFieldType,
  du::Union{CellFieldType, Nothing},
  v::CellFieldType,
  w::AbstractVector,
  r::AbstractVector,
  c::AbstractVector,
  wu::AbstractVector,
  ru::AbstractVector,
  cu::AbstractVector,
  wy::AbstractVector,
  ry::AbstractVector,
  cy::AbstractVector,
)
  _v = restrict(v, term.trian)
  _yuh = restrict(yu, term.trian)

  cellids = Gridap.FESpaces.get_cell_id(term)
  cellvals = integrate(term.biform(_yuh, _v), term.quad)

  Gridap.FESpaces._push_matrix_contribution!(w, r, c, cellvals, cellids)
end
=#

#=
function _jac_from_term_to_terms_id!(
  term::Union{Gridap.FESpaces.LinearFETerm, Gridap.FESpaces.AffineFETermFromIntegration},
  r::AbstractVector,
  c::AbstractVector,
  ru::AbstractVector,
  cu::AbstractVector,
  ry::AbstractVector,
  cy::AbstractVector,
)
  cellids = Gridap.FESpaces.get_cell_id(term)
  w = []

  Gridap.FESpaces._push_matrix_contribution!(w, r, c, [], cellids)
end
=#

#=
function _jac_from_term_to_terms!(
  term::Union{Gridap.FESpaces.NonlinearFETermWithAutodiff, Gridap.FESpaces.NonlinearFETerm},
  κ::AbstractVector,
  yu::FEFunctionType,
  yh::FEFunctionType,
  uh::Union{FEFunctionType, Nothing},
  dyu::CellFieldType,
  dy::CellFieldType,
  du::Union{CellFieldType, Nothing},
  v::CellFieldType,
  w::AbstractVector,
  r::AbstractVector,
  c::AbstractVector,
  wu::AbstractVector,
  ru::AbstractVector,
  cu::AbstractVector,
  wy::AbstractVector,
  ry::AbstractVector,
  cy::AbstractVector,
)
  if typeof(term) == Gridap.FESpaces.NonlinearFETerm
    @warn "_jac_from_term_to_terms!: For NonlinearFETerm, function jac is used to compute the derivative w.r.t. y."
  end

  if du != nothing
    _jac_from_term_to_terms_u!(term, κ, yh, uh, du, v, wu, ru, cu)
  end

  _jac_from_term_to_terms_y!(term, κ, yh, uh, dy, v, wy, ry, cy)
end
=#

#=GRIDAPv15
function _jac_from_term_to_terms_id!(
  term::Union{Gridap.FESpaces.NonlinearFETermWithAutodiff, Gridap.FESpaces.NonlinearFETerm},
  r::AbstractVector,
  c::AbstractVector,
  ru::AbstractVector,
  cu::AbstractVector,
  ry::AbstractVector,
  cy::AbstractVector,
)
  _jac_from_term_to_terms_u_id!(term, ru, cu)
  _jac_from_term_to_terms_y_id!(term, ry, cy)
end
=#

#=GRIDAPv15
#https://github.com/gridap/Gridap.jl/blob/758a8620756e164ba0e6b83dc8dcbb278015b3d9/src/FESpaces/FETerms.jl#L332
function _jac_from_term_to_terms!(
  term::Gridap.FESpaces.FESource,
  κ::AbstractVector,
  yu::FEFunctionType,
  yh::FEFunctionType,
  uh::Union{FEFunctionType, Nothing},
  dyu::CellFieldType,
  dy::CellFieldType,
  du::Union{CellFieldType, Nothing},
  v::CellFieldType,
  w::AbstractVector,
  r::AbstractVector,
  c::AbstractVector,
  wu::AbstractVector,
  ru::AbstractVector,
  cu::AbstractVector,
  wy::AbstractVector,
  ry::AbstractVector,
  cy::AbstractVector,
)
  nothing
end
=#

#=GRIDAPv15
function _jac_from_term_to_terms_id!(
  term::Gridap.FESpaces.FESource,
  r::AbstractVector,
  c::AbstractVector,
  ru::AbstractVector,
  cu::AbstractVector,
  ry::AbstractVector,
  cy::AbstractVector,
)
  nothing
end
=#

#=GRIDAPv15
include("test_autodiff.jl")
=#

#=GRIDAPv15
function _jac_from_term_to_terms_u!(
  term::Union{Gridap.FESpaces.NonlinearFETermWithAutodiff, Gridap.FESpaces.NonlinearFETerm},
  κ::AbstractVector,
  yh::FEFunctionType,
  uh::FEFunctionType,
  du::CellFieldType,
  v::CellFieldType,
  w::AbstractVector,
  r::AbstractVector,
  c::AbstractVector,
)
  _v = restrict(v, term.trian)
  _yh = restrict(yh, term.trian)

  cellids = Gridap.FESpaces.get_cell_id(term)
  function uh_to_cell_residual(uf)
    _uf = Gridap.FESpaces.restrict(uf, term.trian)
    if length(κ) > 0
      return integrate(term.res(κ, vcat(_yh, _uf), _v), term.quad)
    else
      return integrate(term.res(vcat(_yh, _uf), _v), term.quad)
    end
  end

  #cellvals_u = Gridap.FESpaces.autodiff_cell_jacobian_from_residual(uh_to_cell_residual, uh, cellids)
  ###########
  U = Gridap.FESpaces.get_fe_space(uh)
  cell_u_to_cell_residual = Gridap.FESpaces._change_argument_to_cell_u(uh_to_cell_residual, U)
  cell_u = Gridap.FESpaces.get_cell_dof_values(uh)
  _temp = cell_u_to_cell_residual(cell_u)
  ncu = length(_temp[1])
  cell_j = autodiff_array_jacobian2(cell_u_to_cell_residual, cell_u, ncu, cellids)
  cellvals_u = cell_j
  ##########

  Gridap.FESpaces._push_matrix_contribution!(w, r, c, cellvals_u, cellids)

  return w, r, c
end
=#

#=GRIDAPv15
function _jac_from_term_to_terms_u_id!(
  term::Union{Gridap.FESpaces.NonlinearFETermWithAutodiff, Gridap.FESpaces.NonlinearFETerm},
  r::AbstractVector,
  c::AbstractVector,
)
  cellids = Gridap.FESpaces.get_cell_id(term)
  w = []
  Gridap.FESpaces._push_matrix_contribution!(w, r, c, [], cellids)

  return r, c
end
=#

#=GRIDAPv15
function _jac_from_term_to_terms_y_id!(
  term::Union{Gridap.FESpaces.NonlinearFETermWithAutodiff, Gridap.FESpaces.NonlinearFETerm},
  r::AbstractVector,
  c::AbstractVector,
)
  cellids = Gridap.FESpaces.get_cell_id(term)
  w = []
  Gridap.FESpaces._push_matrix_contribution!(w, r, c, [], cellids)

  return r, c
end
=#

#=GRIDAPv15
function _jac_from_term_to_terms_y!(
  term::Gridap.FESpaces.NonlinearFETermWithAutodiff,
  κ::AbstractVector,
  yh::FEFunctionType,
  uh::Union{FEFunctionType, Nothing},
  dy::Union{CellFieldType, Nothing},
  v::CellFieldType,
  w::AbstractVector,
  r::AbstractVector,
  c::AbstractVector,
)
  _v = restrict(v, term.trian)
  #_uh = (uh == nothing) ? Array{Gridap.CellData.GenericCellField{true,()}}(undef,0) : restrict(uh, term.trian)
  _uh = (uh != nothing) ? restrict(uh, term.trian) : nothing
  cellids = Gridap.FESpaces.get_cell_id(term)

  function yh_to_cell_residual(yf) #Tanj: improved solution is to declare the function outside
    _yf = Gridap.FESpaces.restrict(yf, term.trian)
    if length(κ) > 0 && uh != nothing
      return integrate(term.res(κ, vcat(_yf, _uh), _v), term.quad)
    elseif length(κ) > 0 #&& uh == nothing
      return integrate(term.res(κ, _yf, _v), term.quad)
    elseif length(κ) == 0 && uh == nothing
      return integrate(term.res(_yf, _v), term.quad)
    else #length(κ) == 0 && uh != nothing
      return integrate(term.res(vcat(_yf, _uh), _v), term.quad)
    end
  end

  cellvals_y =
    Gridap.FESpaces.autodiff_cell_jacobian_from_residual(yh_to_cell_residual, yh, cellids)

  Gridap.FESpaces._push_matrix_contribution!(w, r, c, cellvals_y, cellids)

  return w, r, c
end
=#

#=GRIDAPv15
function _jac_from_term_to_terms_y!(
  term::Gridap.FESpaces.NonlinearFETerm,
  κ::AbstractVector,
  yh::FEFunctionType,
  uh::Union{FEFunctionType, Nothing},
  dy::Union{CellFieldType, Nothing},
  v::CellFieldType,
  w::AbstractVector,
  r::AbstractVector,
  c::AbstractVector,
)
  _v = restrict(v, term.trian)
  _yh = restrict(yh, term.trian)
  _uh =
    (uh == nothing) ? Array{Gridap.CellData.GenericCellField{true, ()}}(undef, 0) :
    restrict(uh, term.trian)
  _dy = restrict(dy, term.trian)

  cellids = Gridap.FESpaces.get_cell_id(term)
  if length(κ) > 0
    cellvals_y = integrate(term.jac(κ, vcat(_yh, _uh), _du, _v), term.quad)
  else
    cellvals_y = integrate(term.jac(vcat(_yh, _uh), _du, _v), term.quad)
  end

  Gridap.FESpaces._push_matrix_contribution!(w, r, c, cellvals_y, cellids)

  return w, r, c
end
=#
