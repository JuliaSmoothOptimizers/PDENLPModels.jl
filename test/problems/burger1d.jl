function burger1d(args...; n = 512, kwargs...)
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

  nu = 0.08

  # Now we move to the optimization:
  yd(x) = -x[1]^2
  α = 1e-2

  # objective function:
  function f(y, u)
    ∫(0.5 * (yd - y) * (yd - y) + 0.5 * α * u * u)dΩ
  end

  function res(y, u, v) #u is the solution of the PDE and z the control
    ∫(-nu * (∇(v) ⊙ ∇(y)) + dt(y, v) - v * u - v * h)dΩ
  end
  op = FEOperator(res, U, V)

  Xcon = TestFESpace(model, reffe; conformity = :H1)
  Ycon = TrialFESpace(Xcon)
  Ycon = TrialFESpace(Xcon)

  Y = MultiFieldFESpace([U, Ycon])
  xin = zeros(Gridap.FESpaces.num_free_dofs(Y))
  return GridapPDENLPModel(xin, f, trian, U, Ycon, V, Xcon, op)
end

function burger1d_test(; udc = false)
  n = 512
  domain = (0, 1)
  partition = n
  model = CartesianDiscreteModel(domain, partition)
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels, "diri1", [2])
  add_tag_from_tags!(labels, "diri0", [1])

  order = 1
  valuetype = Float64
  reffe = ReferenceFE(lagrangian, valuetype, order)
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
  dconv(du, ∇du, u, ∇u) = conv ∘ (u, ∇du) + conv ∘ (du, ∇u)
  c(u, v) = v ⊙ conv ∘ (u, ∇(u))
  nu = 0.08

  trian = Triangulation(model)
  @test Gridap.FESpaces.num_cells(trian) == 512
  degree = 1
  dΩ = Measure(trian, degree)

  function res(y, v) #u is the solution of the PDE and z the control
    z(x) = 0.5
    ∫(-nu * (∇(v) ⊙ ∇(y)) + c(y, v) - v * z - v * h)dΩ
  end
  op_pde = FEOperator(res, U, V)

  #=
  # Check resolution for z given.
  nls = NLSolver(show_trace = true, method = :newton)
  solver = FESolver(nls)

  uh = solve(solver, op_pde)
  sol_gridap = vcat(get_free_dof_values(uh), 0.5 * ones(513))

  Y = MultiFieldFESpace([nlp.pdemeta.Ypde, nlp.pdemeta.Ycon])
  xin = zeros(Gridap.FESpaces.num_free_dofs(Y))
  nlp = burger1d()

  @test nlp.meta.nvar == 1024
  @test nlp.meta.ncon == 511

  _fx = obj(nlp, sol_gridap)
  _gx = grad(nlp, sol_gridap)
  _Hx = hess(nlp, sol_gridap)
  _Hxl = hess(nlp, sol_gridap, nlp.meta.y0)
  _cx = cons(nlp, sol_gridap)

  _Jx = jac(nlp, sol_gridap)
  _Jx2 = jac(nlp, zeros(nlp.meta.nvar))
  _Jx3 = jac(nlp, ones(nlp.meta.nvar))
  # Note that the derivative w.r.t. to the control is constant.
  @test norm(_Jx2[:, 512:1024] - _Jx3[:, 512:1024]) == 0.0
  @test norm(_Jx[:, 512:1024] - _Jx3[:, 512:1024]) == 0.0

  # Test hprod
  function vector_hessian(nlp, x, l, v)
    n = length(x)
    agrad(t) = ForwardDiff.gradient(x -> dot(cons(nlp, x), l), x + t * v)
    out = ForwardDiff.derivative(t -> agrad(t), 0.0)
    return out
  end

  # hprod!(nlp  :: GridapPDENLPModel, x :: AbstractVector, λ :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector
  Hv = hprod(nlp, sol_gridap, rand(nlp.meta.ncon), rand(nlp.meta.nvar), obj_weight = 0.0)
  @test Hv[512:1024] == zeros(513)
  Hvo = hprod(nlp, sol_gridap, zeros(nlp.meta.ncon), rand(nlp.meta.nvar), obj_weight = 0.0)
  @test Hvo == zeros(1024)
  =#
end
