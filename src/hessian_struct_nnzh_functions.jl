"""
Return a triplet with the structure (rows, cols) and the number of non-zeros elements
in the hessian w.r.t. y and u of the objective function.

The rows and cols returned by _compute_hess_structure_obj are already shifter by `nparam`.
"""
function _compute_hess_structure(tnrj::AbstractEnergyTerm, Y, X, x0, nparam)
  rk, ck, nk = _compute_hess_structure_k(tnrj, Y, X, x0, nparam)
  ro, co, no = _compute_hess_structure_obj(tnrj, Y, X, x0, nparam)
  return vcat(rk, ro), vcat(ck, co), nk + no
end

function _compute_hess_structure(tnrj::AbstractEnergyTerm, op, Y, Ypde, X, x0, nparam)
  robj, cobj, nobj = _compute_hess_structure(tnrj, Y, X, x0, nparam)
  # we should also add the derivative w.r.t. to the parameter
  rck, cck, nck = _compute_hess_structure_k(op, Y, X, x0, nparam)
  # p, n = nparam, length(x0)
  # nnz_hess_k = Int(p * (p + 1) / 2) + (n - p) * p

  rc, cc, nc = _compute_hess_structure(op, Y, Ypde, X, x0, nparam)
  return vcat(robj, rck, rc), vcat(cobj, cck, cc), nobj + nck + nc
end

function _compute_hess_structure_obj(tnrj::AbstractEnergyTerm, Y, X, x0, nparam)
  nini = 0
  nvar = length(x0)
  κ, xyu = x0[1:nparam], x0[(nparam + 1):nvar]
  xh = FEFunction(Y, xyu)
  if nparam > 0
    luh = tnrj.f(κ, xh)
    lag_hess = Gridap.FESpaces._hessian(x -> tnrj.f(κ, x), xh, luh)
  else
    luh = tnrj.f(xh)
    lag_hess = Gridap.FESpaces._hessian(tnrj.f, xh, luh)
  end
  matdata = Gridap.FESpaces.collect_cell_matrix(lag_hess)
  assem = SparseMatrixAssembler(Y, X)
  A = Gridap.FESpaces.allocate_matrix(assem, matdata)
  rows, cols, _ = findnz(tril(A))
  nini = length(rows)
  return rows .+ nparam, cols .+ nparam, nini
end

function _compute_hess_structure_obj(tnrj::NoFETerm, Y, X, x0, nparam)
  return Int[], Int[], 0
end

function _compute_hess_structure_k(tnrj::AbstractEnergyTerm, Y, X, x0, nparam)
  n, p = length(x0), nparam
  prows = n
  nnz_hess_k =
    if (typeof(tnrj) <: MixedEnergyFETerm && tnrj.inde) || typeof(tnrj) <: NoFETerm
      prows = p
      Int(p * (p + 1) / 2)
    else
      Int(p * (p + 1) / 2) + (n - p) * p
    end
  I = ((i, j) for i = 1:prows, j = 1:p if j ≤ i)
  rows = getindex.(I, 1)[:]
  cols = getindex.(I, 2)[:]

  return rows, cols, nnz_hess_k
end

function get_nnz_hess_k(tnrj::AbstractEnergyTerm, nvar, nparam)
  if (typeof(tnrj) <: MixedEnergyFETerm && tnrj.inde) || typeof(tnrj) <: NoFETerm
    nnz_hess_k = Int(nparam * (nparam + 1) / 2)
  else
    nnz_hess_k = Int(nparam * (nparam + 1) / 2) + (nvar - nparam) * nparam
  end
  return nnz_hess_k
end

function _compute_hess_structure(op::AffineFEOperator, Y, Ypde, X, x0, nparam) where {T}
  return Int[], Int[], 0
end

function _compute_hess_structure(op::Gridap.FESpaces.FEOperatorFromWeakForm, Y, Ypde, X, x0, nparam) where {T}
  λ = zeros(Gridap.FESpaces.num_free_dofs(Ypde))
  λf = FEFunction(Ypde, λ) # or Ypde
  nvar = length(x0)
  κ, xyu = x0[1:nparam], x0[(nparam + 1):nvar]
  xh = FEFunction(Y, xyu)
  nvar = length(xyu)
  
  function split_res(x, λ)
    if Gridap.FESpaces.num_free_dofs(Ypde) == Gridap.FESpaces.num_free_dofs(Y)
      if nparam > 0
        return op.res(κ, x, λ)
      else
        return op.res(x, λ)
      end
    else
      y, u = x
      if nparam > 0
        return op.res(κ, y, u, λ)
      else
        return op.res(y, u, λ)
      end
    end
  end
  luh = split_res(xh, λf)

  lag_hess = Gridap.FESpaces._hessian(x -> split_res(x, λf), xh, luh)
  matdata = Gridap.FESpaces.collect_cell_matrix(lag_hess)
  assem = SparseMatrixAssembler(Y, X)
  A = Gridap.FESpaces.allocate_matrix(assem, matdata)
  rows, cols, _ = findnz(tril(A))

  return rows .+ nparam, cols .+ nparam, length(rows)
end

function _compute_hess_structure_k(op::FEOperator, Y, X, x0, nparam)
  n, p = length(x0), nparam
  nnz_hess_k = Int(p * (p + 1) / 2) + (n - p) * p
  I = ((i, j) for i = 1:n, j = 1:p if j ≤ i)
  rows = getindex.(I, 1)[:]
  cols = getindex.(I, 2)[:]

  return rows, cols, nnz_hess_k
end

#=
Programmer note: this function is used in the constructors to set meta.nnzh
=#
"""
`get_nnzh`: return the number of non-zeros elements in the hessian matrix.

Different variants:
- `get_nnzh(tnrj :: T, Y, X, nparam, nvar)`: consider the hessian of the 
objective function only.   
- `get_nnzh(tnrj :: T, op :: AffineFEOperator, Y, X, nparam, nvar)`: consider 
the hessian of the objective function only.    
- `get_nnzh(tnrj :: T, op :: Gridap.FESpaces.FEOperatorFromWeakForm, Y, X, nparam, nvar)`: 
concatenate non-zeros of the objective-hessian and the hessian of each term composing `op`.    

TODO: Do not handle non-linear discrete parameters in the constraints.
"""
function get_nnzh(tnrj::T, Y, X, nparam, nvar) where {T}
  # Special case as tnrj has no field trian.    
  if typeof(tnrj) <: NoFETerm
    nnz_hess_yu = 0
  else
    a = Gridap.FESpaces.SparseMatrixAssembler(Y, X)
    ncells = num_cells(tnrj.trian)
    cell_id_yu = Gridap.Arrays.IdentityVector(ncells)
    nnz_hess_yu = count_hess_nnz_coo_short(a, cell_id_yu)
  end

  #add the nnz w.r.t. k; by default it is:
  if (typeof(tnrj) <: MixedEnergyFETerm && tnrj.inde) || typeof(tnrj) <: NoFETerm
    nnz_hess_k = Int(nparam * (nparam + 1) / 2)
  else
    nnz_hess_k = Int(nparam * (nparam + 1) / 2) + (nvar - nparam) * nparam
  end

  nnzh = nnz_hess_yu + nnz_hess_k
  return nnzh
end

function get_nnzh(tnrj::T, op::AffineFEOperator, Y, Ypde, X, nparam, nvar) where {T}
  return get_nnzh(tnrj, Y, X, nparam, nvar)
end

function get_nnzh(tnrj::T, op::Gridap.FESpaces.FEOperatorFromWeakForm, Y, Ypde, X, nparam, nvar) where {T}
  nnz_hess_obj = get_nnzh(tnrj, Y, X, nparam, nvar)

  nnz_hess_yu = 0
  @warn "Hessian contribution from the operator to be reviewed. get_nnzh L113"
  #=GRIDAPv15
  for term in op.terms
    if typeof(term) <: Gridap.FESpaces.FESourceFromIntegration
      continue #assuming they don't depend on yu
    end
    a = Gridap.FESpaces.SparseMatrixAssembler(Y, X)
    ncells = num_cells(term.trian) #Don't think it works for `BoundaryTriangulation`
    cell_id_yu = Gridap.Arrays.IdentityVector(ncells)
    nnz_hess_yu += count_hess_nnz_coo_short(a, cell_id_yu)
  end
  =#
###################################################################################
  λ = zeros(Gridap.FESpaces.num_free_dofs(Ypde))
  λf = FEFunction(Ypde, λ) # or Ypde
  x = zeros(nvar)
  xh = FEFunction(Y, x)
  
  function split_res(x, λ)
    if Gridap.FESpaces.num_free_dofs(Ypde) == nvar - nparam
      return op.res(x, λ)
    else
      y, u = x
      return op.res(y, u, λ)
    end
  end
  luh = split_res(xh, λf)

  lag_hess = Gridap.FESpaces._hessian(x->split_res(x, λf), xh, luh)
  matdata = Gridap.FESpaces.collect_cell_matrix(lag_hess)
  assem = SparseMatrixAssembler(Y, X)
  A = Gridap.FESpaces.allocate_matrix(assem, matdata)

  nnz_hess_yu += nnz(A)
###################################################################################

  #add the nnz w.r.t. k; by default it is:
  if nparam > 0
    nnz_hess_yu += Int(nparam * (nparam + 1) / 2) + (nvar - nparam) * nparam
  end

  nnzh = nnz_hess_obj + nnz_hess_yu
  return nnzh
end

#= Gridap v15 - don't think it works
function count_hess_nnz_coo_short(
  a::Gridap.FESpaces.GenericSparseMatrixAssembler,
  cell_id_yu::Gridap.Arrays.IdentityVector{I},
) where {I}

  #cellmat_rc  = cell_r_yu
  cellidsrows = cell_id_yu
  cellidscols = cell_id_yu

  cell_rows = Gridap.FESpaces.get_cell_dof_ids(a.test, cellidsrows)
  cell_cols = Gridap.FESpaces.get_cell_dof_ids(a.trial, cellidscols)
  rows_cache = Gridap.FESpaces.array_cache(cell_rows)
  cols_cache = Gridap.FESpaces.array_cache(cell_cols)

  #In the unconstrained case: cellmat = cell_r_yu
  #cellmat_r   = Gridap.FESpaces.attach_constraints_cols(a.trial, cellmat_rc, cellidscols)
  #cellmat     = Gridap.FESpaces.attach_constraints_rows(a.test,  cellmat_r,  cellidsrows)

  #mat = first(cellmat)
  #Is  = Gridap.FESpaces._get_block_layout(mat) #Is = nothing if cellmat is a matrix
  n = _count_hess_entries(
    a.matrix_type,
    rows_cache,
    cols_cache,
    cell_rows,
    cell_cols,
    a.strategy,
    nothing,
  )

  n
end

@noinline function _count_hess_entries(
  ::Type{M},
  rows_cache,
  cols_cache,
  cell_rows,
  cell_cols,
  strategy,
  Is,
) where {M}
  n = 0
  for cell = 1:length(cell_cols)
    rows = getindex!(rows_cache, cell_rows, cell)
    cols = getindex!(cols_cache, cell_cols, cell)
    n += _count_hess_entries_at_cell(M, rows, cols, strategy, Is)
  end
  n
end

@inline function _count_hess_entries_at_cell(::Type{M}, rows, cols, strategy, Is) where {M}
  n = 0
  for gidcol in cols
    if gidcol > 0 && Gridap.FESpaces.col_mask(strategy, gidcol)
      _gidcol = Gridap.FESpaces.col_map(strategy, gidcol)
      for gidrow in rows
        if gidrow > 0 && Gridap.FESpaces.row_mask(strategy, gidrow)
          _gidrow = Gridap.FESpaces.row_map(strategy, gidrow)
          if Gridap.FESpaces.is_entry_stored(M, _gidrow, _gidcol) && (_gidrow >= _gidcol)
            n += 1
          end
        end
      end
    end
  end
  n
end
=#