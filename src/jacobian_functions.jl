function jac_k_structure(p, n)
  nnz_jac_k = p * n
  I = ((i, j) for i = 1:n, j = 1:p)
  rows = getindex.(I, 1)[:]
  cols = getindex.(I, 2)[:]
  return rows, cols, nnz_jac_k
end

include("test_autodiff.jl")

function _jacobian_struct(
  op::AffineFEOperator,
  x::AbstractVector{T},
  Y::FESpace,
  Xpde::FESpace,
  Ypde::FESpace,
  Ycon::FESpace,
) where {T <: Number}
  rows, cols , _ = findnz(get_matrix(op))
  return rows, cols, length(rows)
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
  κ, xyu = x[1:nparam], x[(nparam + 1):nvar]
  yu = FEFunction(Y, xyu)

  v = Gridap.FESpaces.get_cell_shapefuns(Xpde)

  du = Gridap.FESpaces.get_cell_shapefuns_trial(Ypde)
  if nparam == 0
    if typeof(Ycon) <: VoidFESpace
      resuh = op.res(yh, v)
      dcjacy = Gridap.FESpaces._jacobian(x -> op.res(x, du), yh, resuh)
    else
      resuh = op.res(yh, uh, v)
      dcjacy = Gridap.FESpaces._jacobian(x -> op.res(x, uh, du), yh, resuh)
    end
  else
    if typeof(Ycon) <: VoidFESpace
      resuh = op.res(κ, yh, v)
      dcjacy = Gridap.FESpaces._jacobian(x -> op.res(κ, x, du), yh, resuh)
    else
      resuh = op.res(κ, yh, uh, v)
      dcjacy = Gridap.FESpaces._jacobian(x -> op.res(κ, x, uh, du), yh, resuh)
    end
  end
  matdata_y = Gridap.FESpaces.collect_cell_matrix(dcjacy)
  assem = SparseMatrixAssembler(Ypde, Xpde)
  n = Gridap.FESpaces.count_matrix_nnz_coo(assem, matdata_y)
  Iy,Jy,Vy = Gridap.FESpaces.allocate_coo_vectors(Gridap.FESpaces.get_matrix_type(assem), n)
  ny = Gridap.FESpaces.fill_matrix_coo_numeric!(Iy, Jy, Vy, assem, matdata_y)
  Iy, Jy = Iy[1:ny], Jy[1:ny]

  if Ycon != VoidFESpace()
    if nparam == 0
      resuh = op.res(yh, uh, v)
      dcjacu = _jacobian2(x -> op.res(yh, x, du), uh, resuh)
    else
      resuh = op.res(κ, yh, uh, v)
      dcjacu = _jacobian2(x -> op.res(κ, yh, x, du), uh, resuh)
    end
    matdata_u = Gridap.FESpaces.collect_cell_matrix(dcjacu)
    assem = SparseMatrixAssembler(Ycon, Xpde)
    n = Gridap.FESpaces.count_matrix_nnz_coo(assem, matdata_u)
    Iu, Ju, Vu = Gridap.FESpaces.allocate_coo_vectors(Gridap.FESpaces.get_matrix_type(assem), n)
    nu = Gridap.FESpaces.fill_matrix_coo_numeric!(Iu, Ju, Vu, assem, matdata_u)
    Iu, Ju = Iu[1:nu], Ju[1:nu] .+ num_free_dofs(Ypde)
  else
    Iu, Ju, nu = Int[], Int[], 0
  end
  
  return vcat(Iy, Iu), vcat(Jy, Ju), ny + nu
end

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

  v = Gridap.FESpaces.get_cell_shapefuns(Xpde)

  du = Gridap.FESpaces.get_cell_shapefuns_trial(Ypde)
  if nparam == 0
    if typeof(Ycon) <: VoidFESpace
      resuh = op.res(yh, v)
      dcjacy = Gridap.FESpaces._jacobian(x -> op.res(x, du), yh, resuh)
    else
      resuh = op.res(yh, uh, v)
      dcjacy = Gridap.FESpaces._jacobian(x -> op.res(x, uh, du), yh, resuh)
    end
  else
    if typeof(Ycon) <: VoidFESpace
      resuh = op.res(κ, yh, v)
      dcjacy = Gridap.FESpaces._jacobian(x -> op.res(κ, x, du), yh, resuh)
    else
      resuh = op.res(κ, yh, uh, v)
      dcjacy = Gridap.FESpaces._jacobian(x -> op.res(κ, x, uh, du), yh, resuh)
    end
  end
  matdata_y = Gridap.FESpaces.collect_cell_matrix(dcjacy)
  assem = SparseMatrixAssembler(Ypde, Xpde)
  I, J = zeros(Int, length(vals)), zeros(Int, length(vals)) # nlp.Jrows, nlp.Jcols
  nini = Gridap.FESpaces.fill_matrix_coo_numeric!(I, J, vals, assem, matdata_y, nfirst)

  if Ycon != VoidFESpace()
    if nparam == 0
      resuh = op.res(yh, uh, v)
      dcjacu = _jacobian2(x -> op.res(yh, x, du), uh, resuh)
    else
      resuh = op.res(κ, yh, uh, v)
      dcjacu = _jacobian2(x -> op.res(κ, yh, x, du), uh, resuh)
    end
    matdata_u = Gridap.FESpaces.collect_cell_matrix(dcjacu)
    assem = SparseMatrixAssembler(Ycon, Xpde)
    I, J = zeros(Int, length(vals)), zeros(Int, length(vals)) # nlp.Jrows, nlp.Jcols
    nini = Gridap.FESpaces.fill_matrix_coo_numeric!(I, J, vals, assem, matdata_u, nini)
  else
    vals[(nini + 1):end] .= zero(T)
  end

  return nini
end
