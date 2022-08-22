function burger1d_param(args...; n = 512, kwargs...)
  domain = (0, 1)
  partition = n
  model = CartesianDiscreteModel(domain, partition)

  labels = get_face_labeling(model)
  add_tag_from_tags!(labels, "diri1", [2])
  add_tag_from_tags!(labels, "diri0", [1])

  trian = Triangulation(model)
  degree = 1
  dΩ = Measure(trian, degree)

  order = 1
  valuetype = Float64
  reffe = ReferenceFE(lagrangian, valuetype, 1)
  V = TestFESpace(
    model,
    reffe;
    conformity = :H1,
    labels = labels,
    dirichlet_tags = ["diri0", "diri1"],
  )

  h(x) = 2 * (nu + x[1]^3)
  uD0 = VectorValue(0)
  uD1 = VectorValue(-1)
  U = TrialFESpace(V, [uD0, uD1])

  conv(u, ∇u) = (∇u ⋅ one(∇u)) ⊙ u
  c(u, v) = v ⊙ (conv ∘ (u, ∇(u)))
  nu = 0.08

  # Now we move to the optimization:
  yd(x) = -x[1]^2
  α = 1e-2

  # objective function:
  function f(θ, y, u)
    ∫(0.5 * (yd - y) * (yd - y) + 0.5 * α * u * u + 0.5 * (θ[1] - 0.08) * (θ[1] - 0.08))dΩ
  end

  function res(k, y, u, v) #u is the solution of the PDE and z the control
    k1(x) = -k[1]
    ∫(nu * (∇(v) ⊙ ∇(y)) + k1 + c(y, v) - v * u - v * h)dΩ
  end

  Xcon = TestFESpace(model, reffe; conformity = :H1)
  Ycon = TrialFESpace(Xcon)
  Ycon = TrialFESpace(Xcon)

  Y = MultiFieldFESpace([U, Ycon])
  xin = zeros(Gridap.FESpaces.num_free_dofs(Y) + 1)
  return GridapPDENLPModel(xin, f, dΩ, U, Ycon, V, Xcon, res)
end

function burger1d_param_test(; udc = false)
  return true
end
