function incompressiblenavierstokes(args...; n = 3, kwargs...)
  domain = (0, 1, 0, 1)
  partition = (n, n)
  model = CartesianDiscreteModel(domain, partition)

  labels = get_face_labeling(model)
  add_tag_from_tags!(labels, "diri1", [6])
  add_tag_from_tags!(labels, "diri0", [1, 2, 3, 4, 5, 7, 8])

  D = 2
  order = 2
  valuetype = VectorValue{D, Float64}
  reffeᵤ = ReferenceFE(lagrangian, valuetype, order)
  V = TestFESpace(
    model,
    reffeᵤ,
    conformity = :H1,
    labels = labels,
    dirichlet_tags = ["diri0", "diri1"],
  )

  reffeₚ = ReferenceFE(lagrangian, Float64, order - 1; space = :P)
  Q = TestFESpace(model, reffeₚ, conformity = :L2, constraint = :zeromean)

  uD0 = VectorValue(0, 0)
  uD1 = VectorValue(1, 0)
  U = TrialFESpace(V, [uD0, uD1])
  P = TrialFESpace(Q)

  X = MultiFieldFESpace([V, Q])
  Y = MultiFieldFESpace([U, P])

  degree = order # degree = (order - 1) * 2
  Ωₕ = Triangulation(model)
  dΩ = Measure(Ωₕ, degree)

  Re = 10.0
  conv(u, ∇u) = Re * (∇u') ⋅ u
  dconv(du, ∇du, u, ∇u) = conv(u, ∇du) + conv(du, ∇u)

  a((u, p), (v, q)) = ∫(∇(v) ⊙ ∇(u) - (∇ ⋅ v) * p + q * (∇ ⋅ u))dΩ

  c(u, v) = ∫(v ⊙ (conv ∘ (u, ∇(u))))dΩ # c(u, v) = v ⊙ conv(u, ∇(u))
  dc(u, du, v) = ∫(v ⊙ (dconv ∘ (du, ∇(du), u, ∇(u))))dΩ

  res((u, p), (v, q)) = a((u, p), (v, q)) + c(u, v)
  jac((u, p), (du, dp), (v, q)) = a((du, dp), (v, q)) + dc(u, du, v)

  # t_Ω = FETerm(res, Ωₕ, dΩ)
  # op = FEOperator(Y, X, t_Ω)
  op = FEOperator(res, Y, X)
  # t_with_jac_Ω = FETerm(res, ja, Ωₕ, dΩ)
  op_with_jac = FEOperator(res, jac, Y, X)

  ndofs = Gridap.FESpaces.num_free_dofs(Y)
  xin = zeros(ndofs)
  # Ycon, Xcon = nothing, nothing
  # @time nlp = GridapPDENLPModel(xin, x->0.0, Ωₕ, dΩ, Y, Ycon, X, Xcon, op)
  return GridapPDENLPModel(xin, x -> ∫(0.0)dΩ, Ωₕ, Y, X, op)
end

function incompressiblenavierstokes_test(; udc = false)
  nlp = incompressiblenavierstokes()
  xin = nlp.meta.x0
  ndofs = nlp.meta.nvar

  fx = obj(nlp, xin)
  @test fx == 0.0
  gx = grad(nlp, xin)
  @test gx == zeros(nlp.meta.nvar)
  # @test gradient_check(nlp) == Dict{Int64,Float64}()
  _Hxv = hprod(nlp, rand(nlp.meta.nvar), ones(nlp.meta.nvar))
  @test _Hxv == zeros(nlp.meta.nvar)

  # We also compare cons and Gridap.FESpaces.residual using @btime:
  # @btime cons(nlp, xin)
  # @btime Gridap.FESpaces.residual(op, FEFunction(Gridap.FESpaces.get_trial(op), xin))
  # expected result:
  # 1.377 ms (10895 allocations: 1.21 MiB) for  cons :)
  # 2.492 ms (19611 allocations: 2.16 MiB) for residual

  cx = cons(nlp, xin)
  Gcx = Gridap.FESpaces.residual(nlp.pdemeta.op, FEFunction(Gridap.FESpaces.get_trial(nlp.pdemeta.op), xin))
  @test norm(cx - Gcx, Inf) == 0.0
  @test length(cx) == ndofs

  # We also compare jac and Gridap.FESpaces.jacobian using @btime:
  # Note: Avoid confusion with the function jac from the problem
  # @btime jac(nlp, xin); #for now we use AD to compute jacobian
  # @btime Gridap.FESpaces.jacobian(op, FEFunction(Gridap.FESpaces.get_trial(op), xin));
  # @btime Gridap.FESpaces.jacobian(op_with_jac, FEFunction(Gridap.FESpaces.get_trial(op), xin));
  # expected results:
  # 9.656 ms (28898 allocations: 12.78 MiB) for jac :)
  # 25.290 ms (71788 allocations: 31.69 MiB) for jacobian without analytical jacobian
  # 8.562 ms (56321 allocations: 6.61 MiB) for jacobian with analytical jacobian
  Jx = jac(nlp, xin)
  GJx = Gridap.FESpaces.jacobian(nlp.pdemeta.op, FEFunction(Gridap.FESpaces.get_trial(nlp.pdemeta.op), xin))
  #=
  GJx_with_jac = Gridap.FESpaces.jacobian(op_with_jac, FEFunction(Gridap.FESpaces.get_trial(nlp.pdemeta.op), xin))
  @test issparse(Jx)
  @test size(Jx) == (ndofs, ndofs)
  @test norm(GJx - Jx, Inf) <= eps(Float64)
  @test norm(GJx_with_jac - Jx, Inf) <= eps(Float64)
  #@test jacobian_check(nlp) == Dict{Tuple{Int64,Int64},Float64}()
  =#

  Jxx = jprod(nlp, xin, xin)
  @test norm(Jx * xin - Jxx, Inf) <= eps(Float64)
  @test norm(GJx * xin - Jxx, Inf) <= eps(Float64)
  Jtxo = jtprod(nlp, xin, zeros(ndofs))
  @test norm(Jtxo, Inf) <= eps(Float64)
  Jtxu = jtprod(nlp, xin, ones(ndofs))
  @test norm(GJx' * ones(ndofs) - Jtxu, Inf) <= eps(Float64)

  # jac_op = (rows, cols, vals) = findnz(Jx)
  # jacop = jac_op(nlp, xin)
  # jacoptxu = jacop.tprod(ones(ndofs))
  # @test norm(Jtxu - jacoptxu, Inf) <= eps(Float64)

  # Gridap way of solving the equation:
  nls = NLSolver(show_trace = true, method = :newton) #, linesearch=BackTracking()
  solver = FESolver(nls)
  uh, ph = solve(solver, nlp.pdemeta.op)
  sol_gridap = vcat(get_free_values(uh), get_free_values(ph))

  cGx = Gridap.FESpaces.residual(nlp.pdemeta.op, FEFunction(Gridap.FESpaces.get_trial(nlp.pdemeta.op), sol_gridap))
  cx = cons(nlp, sol_gridap)
  @test norm(cx - cGx, Inf) <= eps(Float64)

  JGsolx =
    Gridap.FESpaces.jacobian(nlp.pdemeta.op, FEFunction(Gridap.FESpaces.get_trial(nlp.pdemeta.op), sol_gridap))
  Jsolx = jac(nlp, sol_gridap)
  @test norm(JGsolx - Jsolx, Inf) <= eps(Float64)

  # This is an example where the jacobian is not symmetric
  @test norm(Jx' - Jx) > 1.0

  #Test hprod
  #=
  function vector_hessian(nlp, x, l, v)
    n = length(x)
    agrad(t) = ForwardDiff.gradient(x->dot(cons(nlp, x),l), x + t*v)
    out = ForwardDiff.derivative(t -> agrad(t), 0.)
    return out
  end
  =#

  _Hxou = hprod(nlp, xin, zeros(nlp.meta.ncon), ones(ndofs))
  @test norm(_Hxou, Inf) <= eps(Float64)
  lr, xr = rand(nlp.meta.ncon), rand(ndofs)
  # _Hxrr = hprod(nlp, xin, lr, xr)
  # @test norm(_Hxrr - vector_hessian(nlp, xin, lr, xr) ) <= eps(Float64)

  # H_errs = hessian_check(nlp) #slow
  # @test H_errs[0] == Dict{Int, Dict{Tuple{Int,Int}, Float64}}()
  # H_errs_fg = hessian_check_from_grad(nlp)
  # @test H_errs_fg[0] == Dict{Int, Dict{Tuple{Int,Int}, Float64}}()
end
