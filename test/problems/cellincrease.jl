function cellincrease(args...; x0 = [0.6, 0.1], n = 10, T = 7, kwargs...)
  kp(x) = 1.01
  kr(x) = 2.03

  model = CartesianDiscreteModel((0, T), n)
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels, "diri0", [1]) #initial time condition

  valuetype = Float64
  reffe = ReferenceFE(lagrangian, valuetype, 1)

  Vcon = TestFESpace(model, reffe, conformity = :L2)
  Ucon = TrialFESpace(Vcon)
  Xcon = MultiFieldFESpace([Vcon])
  Ycon = MultiFieldFESpace([Ucon])

  function f(yu)
    cf, pf, uf = yu
    ∫( kp * pf )dΩ
  end

  VI = TestFESpace(model, reffe; conformity = :H1, labels = labels, dirichlet_tags = ["diri0"])
  UI = TrialFESpace(VI, x0[1])
  VS = TestFESpace(model, reffe; conformity = :H1, labels = labels, dirichlet_tags = ["diri0"])
  US = TrialFESpace(VS, x0[2])
  Xpde = MultiFieldFESpace([VI, VS])
  Ypde = MultiFieldFESpace([UI, US])

  trian = Triangulation(model)
  degree = 1
  dΩ = Measure(trian, degree)
  
  conv(u, ∇u) = (∇u ⋅ one(∇u)) ⊙ u
  c(u, v) = conv∘(v, ∇(u)) #v⊙conv(u,∇(u))

  function res(y, u, v)
    cf, pf = y
    p, q = v
    ∫( - p * (kp * pf * (1.0 - cf) - kr * cf * (1.0 - cf - pf)) )dΩ #  c(cf, p) + c(pf, q) )dΩ  + q * (u * kr * cf * (1.0 - cf - pf) - kp * pf * pf)
  end

  Y = MultiFieldFESpace([UI, US, Ucon])
  op_sir = FEOperator(res, Ypde, Xpde)

  xin = zeros(Gridap.FESpaces.num_free_dofs(Y))
  return GridapPDENLPModel(xin, f, trian, dΩ, Ypde, Ycon, Xpde, Xcon, op_sir)
end

################################################################
# Testing:
#=
function cellincrease_test(args...; x0 = [0.6, 0.1], n = 10, T = 7, kwargs...)
  atol, rtol = √eps(), √eps()
  n = 10
  nlp = cellincrease(x0, n, T)
  xr = rand(nlp.meta.nvar)

  #check derivatives
  @test gradient_check(nlp, x = xr, atol = atol, rtol = rtol) ==
        Dict{Tuple{Int64, Int64}, Float64}()
  @test jacobian_check(nlp, x = xr, atol = atol, rtol = rtol) ==
        Dict{Tuple{Int64, Int64}, Float64}()
  ymp = hessian_check(nlp, x = xr, atol = atol, rtol = rtol)
  @test !any(x -> x != Dict{Tuple{Int64, Int64}, Float64}(), values(ymp))
  ymp2 = hessian_check_from_grad(nlp, x = xr, atol = atol, rtol = rtol) #uses the jacobian
  @test !any(x -> x != Dict{Tuple{Int64, Int64}, Float64}(), values(ymp2))
end
=#
