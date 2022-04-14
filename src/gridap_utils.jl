export AffineFEOperatorControl

function AffineFEOperatorControl(a::Function, ℓ::Function, args...)
  AffineFEOperatorControl(args...) do y, u, v
    a(y, u, v), ℓ(v)
  end
end

function AffineFEOperatorControl(weakform, Ypde, Xpde, Ycon, Xcon)
  Y, _ = _to_multifieldfespace(Ypde, Xpde, Ycon, Xcon)
  assem = SparseMatrixAssembler(Y, Xpde)
  return AffineFEOperatorControl(weakform, rhs, Y, Ypde, Xpde, Ycon, Xcon, assem)
end

function AffineFEOperatorControl(weakform, Ypde, Xpde, Ycon, Xcon)
  Y, _ = _to_multifieldfespace(Ypde, Xpde, Ycon, Xcon)
  assem = SparseMatrixAssembler(Y, Xpde)
  return AffineFEOperatorControl(weakform, Y, Ypde, Xpde, Ycon, Xcon, assem)
end

# Adaptation of https://github.com/gridap/Gridap.jl/blob/1dae8117dc5ad5b6276a3f2961a847ecbabc696b/src/FESpaces/AffineFEOperators.jl#L23
function AffineFEOperatorControl(
  weakform::Function,
  trialY,
  trial::FESpace,
  test::FESpace,
  trialu::FESpace,
  ::FESpace,
  assem::Gridap.FESpaces.Assembler,
)
  @assert ! isa(test,TrialFESpace) """\n
  It is not allowed to build an AffineFEOperator with a test space of type TrialFESpace.

  Make sure that you are writing first the trial space and then the test space when
  building an AffineFEOperator or a FEOperator.
  """

  u = get_trial_fe_basis(trial) # get_trial_fe_basis(trialu) # weird
  y = get_trial_fe_basis(trial)
  v = get_fe_basis(test)

  uhd = zero(trial)
  matcontribs, veccontribs = weakform(y, u, v)
  data = Gridap.FESpaces.collect_cell_matrix_and_vector(trial,test,matcontribs,veccontribs,uhd)
  A,b = assemble_matrix_and_vector(assem,data)
  #GC.gc()

  #T.M.: we could go further and have an AffineControlOperator with Ay, Au and b.
  AffineFEOperator(trialY,test,A,b)
end
