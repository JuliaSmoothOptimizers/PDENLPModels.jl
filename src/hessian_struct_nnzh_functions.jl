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

function _compute_hess_structure(tnrj::AbstractEnergyTerm, op, Y, Ypde, Ycon, X, Xpde, x0, nparam)
  robj, cobj, nobj = _compute_hess_structure(tnrj, Y, X, x0, nparam)
  rck, cck, nck = _compute_hess_structure_k(op, Y, X, x0, nparam)
  rc, cc, nc = _compute_hess_structure(op, Y, Ypde, Ycon, X, Xpde, x0, nparam)
  return vcat(robj, rck, rc), vcat(cobj, cck, cc), nobj + nck + nc
end

function _compute_hess_structure_obj(tnrj::AbstractEnergyTerm, Y, X, x0, nparam)
  nini = 0
  κ = @view x0[1:nparam]
  xyu = @view x0[(nparam + 1):end]
  xh = FEFunction(Y, xyu)

  luh = _obj_integral(tnrj, κ, xh)
  lag_hess = Gridap.FESpaces._hessian(x -> _obj_integral(tnrj, κ, x), xh, luh)

  matdata = Gridap.FESpaces.collect_cell_matrix(nlp.Y, nlp.X, lag_hess)
  assem = SparseMatrixAssembler(Y, X)
  n = Gridap.FESpaces.count_matrix_nnz_coo(assem, matdata)
  rows, cols, _ = Gridap.FESpaces.allocate_coo_vectors(Gridap.FESpaces.get_matrix_type(assem), n)
  nini = fill_hessstruct_coo_numeric!(rows, cols, assem, matdata)

  return rows[1:nini] .+ nparam, cols[1:nini] .+ nparam, nini
end

function _compute_hess_structure_obj(tnrj::NoFETerm, Y, X, x0, nparam)
  return Int[], Int[], 0
end

function _compute_hess_structure_k(tnrj::AbstractEnergyTerm, Y, X, x0, nparam)
  n, p = length(x0), nparam
  prows = n
  nnz_hess_k = if (typeof(tnrj) <: MixedEnergyFETerm && tnrj.inde) || typeof(tnrj) <: NoFETerm
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

function _compute_hess_structure(op::AffineFEOperator, Y, Ypde, Ycon, X, Xpde, x0, nparam) where {T}
  return Int[], Int[], 0
end

function _compute_hess_structure(
  op::Gridap.FESpaces.FEOperatorFromWeakForm,
  Y,
  Ypde,
  Ycon,
  X,
  Xpde,
  x0,
  nparam,
) where {T}
  λ = zeros(Gridap.FESpaces.num_free_dofs(Ypde))
  λf = FEFunction(Xpde, λ)
  κ = @view x0[1:nparam]
  xyu = @view x0[(nparam + 1):end]
  xh = FEFunction(Y, xyu)

  function split_res(x, λ)
    if typeof(Ycon) <: VoidFESpace
      if nparam > 0
        return op.res(κ, x, λ)
      else
        return op.res(x, λ)
      end
    else
      y, u = _split_FEFunction(x, Ypde, Ycon)
      if nparam > 0
        return op.res(κ, y, u, λ)
      else
        return op.res(y, u, λ)
      end
    end
  end
  luh = split_res(xh, λf)

  lag_hess = Gridap.FESpaces._hessian(x -> split_res(x, λf), xh, luh)
  matdata = Gridap.FESpaces.collect_cell_matrix(nlp.Y, nlp.X, lag_hess)
  assem = SparseMatrixAssembler(Y, X)
  #=
  A = Gridap.FESpaces.allocate_matrix(assem, matdata)
  rows, cols, _ = findnz(tril(A))
  =#
  n = Gridap.FESpaces.count_matrix_nnz_coo(assem, matdata)
  rows, cols, _ = Gridap.FESpaces.allocate_coo_vectors(Gridap.FESpaces.get_matrix_type(assem), n)
  nini = fill_hessstruct_coo_numeric!(rows, cols, assem, matdata)

  return rows[1:nini] .+ nparam, cols[1:nini] .+ nparam, nini
end

function _compute_hess_structure_k(op::FEOperator, Y, X, x0, nparam)
  n, p = length(x0), nparam
  nnz_hess_k = Int(p * (p + 1) / 2) + (n - p) * p
  I = ((i, j) for i = 1:n, j = 1:p if j ≤ i)
  rows = getindex.(I, 1)[:]
  cols = getindex.(I, 2)[:]

  return rows, cols, nnz_hess_k
end
