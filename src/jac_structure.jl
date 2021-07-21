
function _jac_structure!(
  op::AffineFEOperator,
  nlp::GridapPDENLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)

  #In this case, the jacobian matrix is constant:
  I, J, _ = findnz(get_matrix(op))
  rows .= I
  cols .= J .+ nlp.nparam

  return rows, cols
end

#Adaptation of `function allocate_matrix(a::SparseMatrixAssembler,matdata) end` in Gridap.FESpaces.
function _jac_structure!(
  op::Gridap.FESpaces.FEOperatorFromWeakForm,
  nlp::GridapPDENLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  A = _from_terms_to_jacobian(op, nlp.meta.x0, nlp.Y, nlp.Xpde, nlp.Ypde, nlp.Ycon)
  r, c, _ = findnz(A)
  rows .= r
  cols .= c .+ nlp.nparam

  return rows, cols
end