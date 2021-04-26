function burger1d(args...; n = 512, kwargs...)
  domain = (0, 1)
  partition = n
  model = CartesianDiscreteModel(domain, partition)

  labels = get_face_labeling(model)
  add_tag_from_tags!(labels, "diri1", [2])
  add_tag_from_tags!(labels, "diri0", [1])

  D = 1
  order = 1
  V = TestFESpace(
    reffe = :Lagrangian,
    conformity = :H1,
    valuetype = Float64,
    model = model,
    labels = labels,
    order = order,
    dirichlet_tags = ["diri0", "diri1"],
  )

  h(x) = 2 * (nu + x[1]^3)
  uD0 = VectorValue(0)
  uD1 = VectorValue(-1)
  U = TrialFESpace(V, [uD0, uD1])

  @law conv(u, ∇u) = (∇u ⋅ one(∇u)) ⊙ u
  @law dconv(du, ∇du, u, ∇u) = conv(u, ∇du) + conv(du, ∇u)

  function a(u, v)
    ∇(v) ⊙ ∇(u)
  end

  c(u, v) = v ⊙ conv(u, ∇(u))
  nu = 0.08
  function res_pde(u, v)
    z(x) = 0.5
    -nu * (∇(v) ⊙ ∇(u)) + c(u, v) - v * z - v * h
  end

  trian = Triangulation(model)
  degree = 1
  quad = CellQuadrature(trian, degree)
  t_Ω = FETerm(res_pde, trian, quad)
  op_pde = FEOperator(U, V, t_Ω)

  # Now we move to the optimization:
  ud(x) = -x[1]^2
  α = 1e-2

  # objective function:
  f(u, z) = 0.5 * (ud - u) * (ud - u) + 0.5 * α * z * z
  function f(yu) #:: Union{Gridap.MultiField.MultiFieldFEFunction, Gridap.CellData.GenericCellField}
    u, z = yu
    f(u, z)
  end

  function res(yu, v) #u is the solution of the PDE and z the control
    u, z = yu
    -nu * (∇(v) ⊙ ∇(u)) + c(u, v) - v * z - v * h
  end
  t_Ω = FETerm(res, trian, quad)
  op = FEOperator(U, V, t_Ω)

  Xcon = TestFESpace(
    reffe = :Lagrangian,
    order = 1,
    valuetype = Float64,
    conformity = :H1,
    model = model,
  )
  Ycon = TrialFESpace(Xcon)

  Y = MultiFieldFESpace([U, Ycon])
  xin = zeros(Gridap.FESpaces.num_free_dofs(Y))
  return GridapPDENLPModel(xin, f, trian, quad, U, Ycon, V, Xcon, op)
end

#using LineSearches: BackTracking

function burger1d_test(; udc = false)
  n = 512
  domain = (0, 1)
  partition = n
  model = CartesianDiscreteModel(domain, partition)
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels, "diri1", [2])
  add_tag_from_tags!(labels, "diri0", [1])

  D = 1
  order = 1
  V = TestFESpace(
    reffe = :Lagrangian,
    conformity = :H1,
    valuetype = Float64,
    model = model,
    labels = labels,
    order = order,
    dirichlet_tags = ["diri0", "diri1"],
  )

  h(x) = 2 * (nu + x[1]^3)
  uD0 = VectorValue(0)
  uD1 = VectorValue(-1)
  U = TrialFESpace(V, [uD0, uD1])

  @law conv(u, ∇u) = (∇u ⋅ one(∇u)) ⊙ u
  @law dconv(du, ∇du, u, ∇u) = conv(u, ∇du) + conv(du, ∇u)
  c(u, v) = v ⊙ conv(u, ∇(u))
  nu = 0.08
  function res_pde(u, v)
    z(x) = 0.5
    -nu * (∇(v) ⊙ ∇(u)) + c(u, v) - v * z - v * h
  end

  trian = Triangulation(model)
  @test Gridap.FESpaces.num_cells(trian) == 512
  degree = 1
  quad = CellQuadrature(trian, degree)
  t_Ω = FETerm(res_pde, trian, quad)
  op_pde = FEOperator(U, V, t_Ω)

  # Check resolution for z given.
  nls = NLSolver(show_trace = true, method = :newton) #, linesearch=BackTracking()
  solver = FESolver(nls)

  uh = solve(solver, op_pde)
  sol_gridap = vcat(get_free_values(uh), 0.5 * ones(513))

  Y = MultiFieldFESpace([nlp.Ypde, nlp.Ycon])
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
  Hv =
    hprod(nlp, sol_gridap, rand(nlp.meta.ncon), rand(nlp.meta.nvar), obj_weight = 0.0)
  @test Hv[512:1024] == zeros(513)
  Hvo =
    hprod(nlp, sol_gridap, zeros(nlp.meta.ncon), rand(nlp.meta.nvar), obj_weight = 0.0)
  @test Hvo == zeros(1024)
end
