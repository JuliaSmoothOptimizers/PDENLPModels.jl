function _from_terms_to_hprod!_wish_it_would_work(op  :: Gridap.FESpaces.FEOperatorFromTerms,
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
  λf = FEFunction(nlp.Xpde, λ) #λf = FEFunction(nlp.Xpde, λ_pde)
  vc = Gridap.FESpaces.get_cell_basis(nlp.Xpde) #X or Xpde? /nlp.op.test

  cell_yu = Gridap.FESpaces.get_cell_values(yu)
  cell_vf = Gridap.FESpaces.get_cell_values(vf)
  cell_λf = Gridap.FESpaces.get_cell_values(λf)

  ncells  = length(cell_yu)
  cell_id = Gridap.Arrays.IdentityVector(length(cell_yu))

  function _cell_lag_t(cell)
    yu  = CellField(nlp.Y, cell)

    _lag = _obj_cell_integral(nlp.tnrj, yu, κ)

    _dotvalues = Array{Any, 1}(undef, ncells)
    for term in nlp.op.terms
      _vc = restrict(vc,  term.trian)
      _yu = restrict(yu, term.trian)
      cellvals = integrate(term.res(_yu, _vc), term.trian, term.quad)
      for i=1:ncells
        _dotvalues[i] = dot(cellvals[i], cell_λf[i]) #setindex not defined for Gridap.Arrays
      end
    end
    _lag += _dotvalues
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
  Hv .= Gridap.FESpaces.assemble_vector(assem, vecdata_yu)

  return Hv
end

function hess_coo(nlp :: GridapPDENLPModel, x :: AbstractVector, λ :: AbstractVector; obj_weight :: Real = one(eltype(x)))

  @warn "hess_coo: This doesn't work."

  yu    = FEFunction(nlp.Y, x)
  λf    = FEFunction(nlp.Xpde, λ)
  v   = Gridap.FESpaces.get_cell_basis(nlp.Xpde)

  cell_yu    = Gridap.FESpaces.get_cell_values(yu)
  cell_λf    = Gridap.FESpaces.get_cell_values(λf)

  ncells = length(cell_yu)
  cell_id_yu = Gridap.Arrays.IdentityVector(ncells)
  term = nlp.op.terms[1]
  _v   = restrict(v,  term.trian)

  function _cell_res_yu(cell)
    yuh = CellField(nlp.Y, cell)
    _yuh = Gridap.FESpaces.restrict(yuh, term.trian)
    _res = integrate(term.res(_yuh,_v), term.trian, term.quad)
    lag  = Array{Any,1}(nothing, ncells)#Array{Any,1}(undef,ncells)
    for j in 1:ncells
      lag[j] = sum(_res[j])#dot(_res[j], cell_λf[j])
    end
    return lag
  end

  #Compute the hessian with AD
  agrad      = i_to_y -> Gridap.Arrays.autodiff_array_gradient(_cell_res_yu, i_to_y, cell_id_yu)
  cell_r_yu  = Gridap.Arrays.autodiff_array_jacobian(agrad, cell_yu, cell_id_yu)
  #cell_r_yu  = Gridap.Arrays.autodiff_array_hessian(_cell_res_yu, cell_yu, cell_id_yu)
  #Assemble the matrix in the "good" space
  assem      = Gridap.FESpaces.SparseMatrixAssembler(nlp.Y, nlp.X)
  (I ,J, V) = Main.PDENLPModels.assemble_hess(assem, cell_r_yu, cell_id_yu)

  return (I ,J, V)
end
