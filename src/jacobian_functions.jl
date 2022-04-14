function jac_k_structure(p, n)
  nnz_jac_k = p * n
  I = ((i, j) for i = 1:n, j = 1:p)
  rows = getindex.(I, 1)[:]
  cols = getindex.(I, 2)[:]
  return rows, cols, nnz_jac_k
end

include("test_autodiff.jl")

function _from_terms_to_residual!(
  op::AffineFEOperator,
  x::AbstractVector{T},
  nparam::Integer,
  Y::FESpace,
  Ypde::FESpace,
  Ycon::FESpace,
  res::AbstractVector,
) where {T}
  mul!(res, get_matrix(op), x)
  axpy!(-one(T), get_vector(op), res)
  return res
end

function _jacobian_struct(
  op::AffineFEOperator,
  x::AbstractVector{T},
  Y::FESpace,
  Xpde::FESpace,
  Ypde::FESpace,
  Ycon::FESpace,
) where {T <: Number}
  rows, cols, _ = findnz(get_matrix(op))
  return rows, cols, length(rows)
end

function _jac_coord!(
  op::AffineFEOperator,
  nparam::Integer,
  ncon::Integer,
  Y::FESpace,
  Ypde::FESpace,
  Xpde::FESpace,
  Ycon::FESpace,
  x::AbstractVector,
  vals::AbstractVector,
  c::AbstractVector,
  Jk,
)
  _, _, V = findnz(get_matrix(op))
  vals .= V
  return vals
end

function _from_terms_to_residual!(
  op::Gridap.FESpaces.FEOperatorFromWeakForm,
  x::AbstractVector,
  nparam::Integer,
  Y::FESpace,
  Ypde::FESpace,
  Ycon::FESpace,
  res::AbstractVector,
)
  κ = @view x[1:(nparam)]
  xyu = @view x[(nparam + 1):end]
  y, u = _split_FEFunction(xyu, Ypde, Ycon)

  # Gridap.FESpaces.residual(op, FEFunction(Y, x))
  # Split the call of: b = allocate_residual(op, u)
  V = Gridap.FESpaces.get_test(op)
  v = Gridap.FESpaces.get_fe_basis(V)
  if nparam == 0
    if typeof(Ycon) <: VoidFESpace
      vecdata = Gridap.FESpaces.collect_cell_vector(V, op.res(y, v))
    else
      vecdata = Gridap.FESpaces.collect_cell_vector(V, op.res(y, u, v))
    end
  else
    if typeof(Ycon) <: VoidFESpace
      vecdata = Gridap.FESpaces.collect_cell_vector(V, op.res(κ, y, v))
    else
      vecdata = Gridap.FESpaces.collect_cell_vector(V, op.res(κ, y, u, v))
    end
  end
  # res = allocate_vector(op.assem, vecdata) # already done somewhere
  # Split the call of: residual!(b,op,u)
  Gridap.FESpaces.assemble_vector!(res, op.assem, vecdata)

  return res
end

function _jacobian_struct(
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
  κ = @view x[1:nparam]

  v = Gridap.FESpaces.get_fe_basis(Xpde)
  #du = Gridap.FESpaces.get_trial_fe_basis(Ypde)

  if nparam == 0
    if typeof(Ycon) <: VoidFESpace
      resuh = op.res(yh, v)
      dcjacy = Gridap.FESpaces._jacobian(x -> op.res(x, v), yh, resuh)
    else
      resuh = op.res(yh, uh, v)
      dcjacy = Gridap.FESpaces._jacobian(x -> op.res(x, uh, v), yh, resuh)
    end
  else
    if typeof(Ycon) <: VoidFESpace
      resuh = op.res(κ, yh, v)
      dcjacy = Gridap.FESpaces._jacobian(x -> op.res(κ, x, v), yh, resuh)
    else
      resuh = op.res(κ, yh, uh, v)
      dcjacy = Gridap.FESpaces._jacobian(x -> op.res(κ, x, uh, v), yh, resuh)
    end
  end
  matdata_y = Gridap.FESpaces.collect_cell_matrix(Ypde, Xpde, dcjacy)
  assem = SparseMatrixAssembler(Ypde, Xpde)
  #= Gridap 0.15.5
  n = Gridap.FESpaces.count_matrix_nnz_coo(assem, matdata_y)
  Iy, Jy, Vy = Gridap.FESpaces.allocate_coo_vectors(Gridap.FESpaces.get_matrix_type(assem), n)
  ny = Gridap.FESpaces.fill_matrix_coo_numeric!(Iy, Jy, Vy, assem, matdata_y)
  =#
  m1 = Gridap.FESpaces.nz_counter(
    Gridap.FESpaces.get_matrix_builder(assem),
    (Gridap.FESpaces.get_rows(assem), Gridap.FESpaces.get_cols(assem)),
  ) # Gridap.Algebra.CounterCS
  Gridap.FESpaces.symbolic_loop_matrix!(m1, assem, matdata_y)
  m2 = Gridap.FESpaces.nz_allocation(m1) # Gridap.Algebra.InserterCSC
  Gridap.FESpaces.symbolic_loop_matrix!(m2, assem, matdata_y)
  m3 = sparse(Gridap.FESpaces.create_from_nz(m2))
  Iy, Jy, _ = findnz(m3) # If I remember correctly, this is what I wanted to avoid...
  # Gridap.FESpaces.numeric_loop_matrix!(m2, assem, matdata)
  ny = length(Iy)
  Iy, Jy = Iy[1:ny], Jy[1:ny]

  if Ycon != VoidFESpace()
    if nparam == 0
      resuh = op.res(yh, uh, v)
      dcjacu = _jacobian2(x -> op.res(yh, x, v), uh, resuh)
    else
      resuh = op.res(κ, yh, uh, v)
      dcjacu = _jacobian2(x -> op.res(κ, yh, x, v), uh, resuh)
    end
    matdata_u = Gridap.FESpaces.collect_cell_matrix(Ycon, Xpde, dcjacu)
    assem = SparseMatrixAssembler(Ycon, Xpde)
    #=
    n = Gridap.FESpaces.count_matrix_nnz_coo(assem, matdata_u)
    Iu, Ju, Vu = Gridap.FESpaces.allocate_coo_vectors(Gridap.FESpaces.get_matrix_type(assem), n)
    nu = Gridap.FESpaces.fill_matrix_coo_numeric!(Iu, Ju, Vu, assem, matdata_u)
    =#
    m1 = Gridap.FESpaces.nz_counter(
      Gridap.FESpaces.get_matrix_builder(assem),
      (Gridap.FESpaces.get_rows(assem), Gridap.FESpaces.get_cols(assem)),
    ) # Gridap.Algebra.CounterCS
    Gridap.FESpaces.symbolic_loop_matrix!(m1, assem, matdata_u)
    m2 = Gridap.FESpaces.nz_allocation(m1) # Gridap.Algebra.InserterCSC
    Gridap.FESpaces.symbolic_loop_matrix!(m2, assem, matdata_u)
    m3 = sparse(Gridap.FESpaces.create_from_nz(m2))
    Iu, Ju, _ = findnz(m3) # If I remember correctly, this is what I wanted to avoid...
    # Gridap.FESpaces.numeric_loop_matrix!(m2, assem, matdata)
    nu = length(Iu)
    Iu, Ju = Iu[1:nu], Ju[1:nu]
  else
    Iu, Ju, nu = Int[], Int[], 0
  end

  return Iy, Iu, Jy, Ju, ny, nu
end

function _jac_coord!(
  op::Gridap.FESpaces.FEOperatorFromWeakForm,
  nparam::Integer,
  ncon::Integer,
  Y::FESpace,
  Ypde::FESpace,
  Xpde::FESpace,
  Ycon::FESpace,
  x::AbstractVector{T},
  Jyrows,
  Jycols,
  Jurows,
  Jucols,
  vals::AbstractVector,
  c::AbstractVector,
  Jk,
) where {T}
  nnz_jac_k = nparam > 0 ? ncon * nparam : 0
  if nparam > 0
    κ = @view x[1:(nparam)]
    xyu = @view x[(nparam + 1):end]
    ck = @closure (c, k) -> _from_terms_to_residual!(op, vcat(k, xyu), nparam, Y, Ypde, Ycon, c)
    ForwardDiff.jacobian!(Jk, ck, c, κ)
    vals[1:nnz_jac_k] .= Jk[:]
  end
  nini = _from_terms_to_jacobian_vals!(op, x, Y, Xpde, Ypde, Ycon, Jyrows, Jycols, Jurows, Jucols, vals, nfirst = nnz_jac_k)
  return vals
end

# T.M.: We could now split this in two
# To avoid using  a sparse matrix, one would have to dive into this:
# https://github.com/gridap/Gridap.jl/blob/baa3ced8f28418bd63f12ac9b5d167ea78a9ae6d/src/FESpaces/SparseMatrixAssemblers.jl#L35
function _from_terms_to_jacobian_vals!(
  op::Gridap.FESpaces.FEOperatorFromWeakForm,
  x::AbstractVector{T},
  Y::FESpace,
  Xpde::FESpace,
  Ypde::FESpace,
  Ycon::FESpace,
  Jyrows,
  Jycols,
  Jurows,
  Jucols,
  vals::AbstractVector{T};
  nfirst::Integer = 0,
) where {T <: Number}
  nvar = length(x)
  nyu = num_free_dofs(Y)
  nparam = nvar - nyu
  κ = @view x[1:nparam]
  xyu = @view x[(nparam + 1):end]
  yh, uh = _split_FEFunction(xyu, Ypde, Ycon)

  nini = nfirst
  ny = num_free_dofs(Ypde)
  nnzh_y = length(Jyrows)
  nu = num_free_dofs(Ycon) 
  nnzh_u = length(Jurows)

  v = Gridap.FESpaces.get_fe_basis(Xpde)
  # du = Gridap.FESpaces.get_trial_fe_basis(Ypde)

  if nparam == 0
    if typeof(Ycon) <: VoidFESpace
      resuh = op.res(yh, v)
      dcjacy = Gridap.FESpaces._jacobian(x -> op.res(x, v), yh, resuh)
    else
      resuh = op.res(yh, uh, v)
      dcjacy = Gridap.FESpaces._jacobian(x -> op.res(x, uh, v), yh, resuh)
    end
  else
    if typeof(Ycon) <: VoidFESpace
      resuh = op.res(κ, yh, v)
      dcjacy = Gridap.FESpaces._jacobian(x -> op.res(κ, x, v), yh, resuh)
    else
      resuh = op.res(κ, yh, uh, v)
      dcjacy = Gridap.FESpaces._jacobian(x -> op.res(κ, x, uh, v), yh, resuh)
    end
  end
  matdata_y = Gridap.FESpaces.collect_cell_matrix(Ypde, Xpde, dcjacy)
  assem = SparseMatrixAssembler(Ypde, Xpde)

  ms = sparse(Jyrows, Jycols, zeros(T, nnzh_y), ny, ny) # SparseArrays.sparse!
  Gridap.FESpaces.numeric_loop_matrix!(ms, assem, matdata_y)
  vals[(nini + 1):(nini + nnzh_y)] .= ms.nzval
  nini += nnzh_y

  if Ycon != VoidFESpace()
    if nparam == 0
      resuh = op.res(yh, uh, v)
      dcjacu = _jacobian2(x -> op.res(yh, x, v), uh, resuh)
    else
      resuh = op.res(κ, yh, uh, v)
      dcjacu = _jacobian2(x -> op.res(κ, yh, x, v), uh, resuh)
    end
    matdata_u = Gridap.FESpaces.collect_cell_matrix(Ycon, Xpde, dcjacu)
    assem = SparseMatrixAssembler(Ycon, Xpde)

    ms = sparse(Jurows, Jucols, zeros(T, nnzh_u), ny, nu) # SparseArrays.sparse!
    Gridap.FESpaces.numeric_loop_matrix!(ms, assem, matdata_u)
    vals[(nini + 1):(nini + nnzh_u)] .= ms.nzval
    nini += nnzh_u
  else
    vals[(nini + 1):end] .= zero(T)
  end

  return nini
end
