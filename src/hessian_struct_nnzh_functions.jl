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

# https://github.com/gridap/Gridap.jl/blob/30b52053cba997008a75cc6fc79242fbf19a31ec/src/MultiField/MultiFieldFEAutodiff.jl#L123
# Tanj: November, 5th 2021. Why is this function commented in Gridap?
function Gridap.FESpaces._change_argument(
  op::typeof(hessian),f,trian,uh::Gridap.MultiField.MultiFieldFEFunction) 

  U = Gridap.FESpaces.get_fe_space(uh)
  function g(cell_u)
    single_fields = Gridap.FESpaces.GenericCellField[]
    nfields = length(U.spaces)
    for i in 1:nfields
      cell_values_field = lazy_map(a->a.array[i],cell_u)
      cf = CellField(U.spaces[i],cell_values_field)
      cell_data = lazy_map(Gridap.FESpaces.BlockMap((nfields,nfields),i),Gridap.FESpaces.get_data(cf))
      uhi = Gridap.FESpaces.GenericCellField(cell_data,Gridap.FESpaces.get_triangulation(cf),DomainStyle(cf))
      push!(single_fields,uhi)
    end
    xh = Gridap.MultiField.MultiFieldCellField(single_fields)
    cell_grad = f(xh)
    Gridap.FESpaces.get_contribution(cell_grad,trian)
  end
  g
end

#=
# this is a direct adaptation of the one for the jacobian
function Gridap.FESpaces._change_argument(
  op::typeof(hessian),f,trian,uh::Gridap.MultiField.MultiFieldFEFunction)

  U = Gridap.FESpaces.get_fe_space(uh)
  function g(cell_u)
    single_fields = Gridap.FESpaces.GenericCellField[]
    nfields = length(U.spaces)
    cell_dofs_field_offsets=Gridap.MultiField._get_cell_dofs_field_offsets(uh)
    for i in 1:nfields
      view_range=cell_dofs_field_offsets[i]:cell_dofs_field_offsets[i+1]-1
      cell_values_field = lazy_map(a->view(a,view_range),cell_u)
      cf = CellField(U.spaces[i],cell_values_field)
      push!(single_fields,cf)
    end
    xh = Gridap.MultiField.MultiFieldCellField(single_fields)
    cell_grad = f(xh)
    cell_grad_cont_block=Gridap.FESpaces.get_contribution(cell_grad,trian)
    bs = [cell_dofs_field_offsets[i+1]-cell_dofs_field_offsets[i] for i=1:nfields]
    lazy_map(Gridap.MultiField.DensifyInnerMostBlockLevelMap(),
             Gridap.MultiField.Fill(bs,length(cell_grad_cont_block)),
             cell_grad_cont_block)
  end
  g
end
=#

function _compute_hess_structure_obj(tnrj::AbstractEnergyTerm, Y, X, x0, nparam)
  nini = 0
  κ = @view x0[1:nparam]
  xyu = @view x0[(nparam + 1):end]
  xh = FEFunction(Y, xyu)

  luh = _obj_integral(tnrj, κ, xh)
  lag_hess = Gridap.FESpaces._hessian(x -> _obj_integral(tnrj, κ, x), xh, luh) # issue with the _change_argument function
  # lag_hess = Gridap.FESpaces.jacobian(xx -> Gridap.FESpaces._gradient(x -> _obj_integral(tnrj, κ, x), xx, luh), xh) # try again here converting the output of _gradient....

  matdata = Gridap.FESpaces.collect_cell_matrix(Y, X, lag_hess)
  assem = SparseMatrixAssembler(Y, X)
  # n = Gridap.FESpaces.count_matrix_nnz_coo(assem, matdata)
  # rows, cols, _ = Gridap.FESpaces.allocate_coo_vectors(Gridap.FESpaces.get_matrix_type(assem), n)
  m1 = Gridap.FESpaces.nz_counter(
    Gridap.FESpaces.get_matrix_builder(assem),
    (Gridap.FESpaces.get_rows(assem), Gridap.FESpaces.get_cols(assem)),
  ) # Gridap.Algebra.CounterCS
  Gridap.FESpaces.symbolic_loop_matrix!(m1, assem, matdata)
  m2 = Gridap.FESpaces.nz_allocation(m1) # Gridap.Algebra.InserterCSC
  Gridap.FESpaces.symbolic_loop_matrix!(m2, assem, matdata)
  m3 = sparse(LowerTriangular(Gridap.FESpaces.create_from_nz(m2)))
  rows, cols, _ = findnz(m3) # If I remember correctly, this is what I wanted to avoid...
  # Gridap.FESpaces.numeric_loop_matrix!(m2, assem, matdata)
  nini = length(rows) # Gridap 0.15 fill_hessstruct_coo_numeric!(rows, cols, assem, matdata)

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
  #lag_hess = Gridap.FESpaces.jacobian(Gridap.FESpaces._gradient(x -> split_res(x, λf), xh, luh), xh)
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
