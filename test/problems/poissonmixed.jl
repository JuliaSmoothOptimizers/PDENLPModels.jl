#using Gridap, PDENLPModels, LinearAlgebra, SparseArrays, NLPModels, NLPModelsTest

###############################################################################
#
# This test case consider the optimization of a parameter in a Poisson equation
# with Dirichlet boundary conditions.
#
# Aim:
# * Test mixed problem
# * with integral term in the objective function
# * |k| = 2
#
###############################################################################
function poissonmixed(args...; n = 3, kwargs...)
  domain = (0, 1, 0, 1)
  partition = (n, n)
  model = CartesianDiscreteModel(domain, partition)

  #We use a manufactured solution of the PDE:
  sol(x) = sin(2 * pi * x[1]) * x[2]

  valuetype = Float64
  reffe = ReferenceFE(lagrangian, valuetype, 1)
  V0 = TestFESpace(model, reffe; conformity = :H1, dirichlet_tags = "boundary")
  Ug = TrialFESpace(V0, x -> 0.0) # sol
  trian = Triangulation(model)
  degree = 2
  dΩ = Measure(trian, degree)

  #We deduce the rhs of the Poisson equation with our manufactured solution:
  f(x) = (2 * pi^2) * sin(2 * pi * x[1]) * x[2]

  function res(k, y, v)
    ∫(k[1] * ∇(v) ⊙ ∇(y) - v * f * k[2])dΩ
  end
  op = FEOperator(res, Ug, V0)

  function fk(k, y)
    ∫(
      0.5 * (sol - y) * (sol - y) +
      0.5 * (k[1] - 1.0) * (k[1] - 1.0) +
      0.5 * (k[2] - 1.0) * (k[2] - 1.0),
    )dΩ
  end
  Vp = TestFESpace(model, reffe; conformity = :H1)
  #Large = MultiFieldFESpace(repeat([Vp], 2))
  #nrj = MixedEnergyFETerm(fk, trian, dΩ, 2, Large) #length(k)=2
  nrj = MixedEnergyFETerm(fk, trian, dΩ, 2, true)

  nUg = num_free_dofs(Ug)
  x0 = zeros(nUg + 2) #zeros(nUg + 2)
  return GridapPDENLPModel(x0, nrj, Ug, V0, op)
end

function poissonmixed_test()
  return true
end
