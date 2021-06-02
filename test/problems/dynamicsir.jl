function dynamicsir(args...; x0 = [1, 2], n = 10, T = 1, kwargs...)
  model = CartesianDiscreteModel((0, T), n)
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels, "diri0", [1]) #initial time condition

  #If we rewrite it as one? and then split yu = bf, cf
  valuetype = Float64
  reffe = ReferenceFE(lagrangian, valuetype, 1)
  VI = TestFESpace(model, reffe; conformity = :H1, labels = labels, dirichlet_tags = ["diri0"])
  UI = TrialFESpace(VI, x0[1])
  VS = TestFESpace(model, reffe; conformity = :H1, labels = labels, dirichlet_tags = ["diri0"])
  US = TrialFESpace(VS, x0[2])
  Xpde = MultiFieldFESpace([VI, VS])
  Ypde = MultiFieldFESpace([UI, US])

  conv(u, ∇u) = (∇u ⋅ one(∇u)) ⊙ u
  c(u, v) = conv(v, ∇(u)) #v⊙conv(u,∇(u))
  function res_pde_nl(yu, v)
    I, S, bf, cf = yu
    p, q = v
    ∫( c(I, p) + c(S, q) )dΩ
  end
  function res_pde(yu, v)
    I, S, bf, cf = yu
    p, q = v
    ∫( -p * (bf * S * I - cf * I) + q * bf * S * I )dΩ
  end

  trian = Triangulation(model)
  degree = 1
  dΩ = Measure(trian, degree)
  t_Ω_nl = FETerm(res_pde_nl, trian, dΩ)
  t_Ω = FETerm(res_pde, trian, dΩ)
  op_sir = FEOperator(Ypde, Xpde, t_Ω_nl, t_Ω)

  Xbcon = TestFESpace(model, reffe; conformity = :H1)
  Ybcon = TrialFESpace(Xbcon)
  Xccon = TestFESpace(model, reffe; conformity = :H1)
  Yccon = TrialFESpace(Xccon)
  Xcon = MultiFieldFESpace([Xbcon, Xccon])
  Ycon = MultiFieldFESpace([Ybcon, Yccon])

  w0(x) = 1.0 + x
  w1(x) = 1.0 #1. /(x+1.)
  w2(x) = 2.0 #1. /(1. + x) #/(x+1.)
  #we need to be smart to avoid divisions
  function f(yu) #:: Union{Gridap.MultiField.MultiFieldFEFunction, Gridap.CellData.GenericCellField}
    I, S, bf, cf = yu
    ∫( 0.5 * ((bf ⋅ w0) - w1) ⋅ ((bf ⋅ w0) - w1) + 0.5 * ((cf ⋅ w0) - w2) ⋅ ((cf ⋅ w0) - w2) )dΩ
  end

  ndofs = Gridap.FESpaces.num_free_dofs(Ypde) + Gridap.FESpaces.num_free_dofs(Ycon)
  xin = zeros(ndofs)
  return GridapPDENLPModel(xin, f, trian, dΩ, Ypde, Ycon, Xpde, Xcon, op_sir)
end

#=
function dynamicsir_test(; x0 = [1, 2], n = 10, T = 1)
  h = T / n
  N = sum(x0)
  #=
  AI = 1/h * Bidiagonal(ones(n), -ones(n-1), :L)
  AS = 1/h * Bidiagonal(ones(n), -ones(n-1), :L)
  A0 = zeros(2 * n); A0[1] = -x0[1] / h; A0[n+1] = -x0[2] / h;
  c(x) = vcat(AI * x[1:n], AS * x[n+1:2*n]) + A0 - F(x)
  =#
  ################################################################
  # The exact solution of the ODE is given by:
  # known in Exact Solution to a Dynamic SIR Model, M. Bohner, S. Streipert, D. F. M. Torres, Researchgate preprint 2018
  # if b(t)=1/(t+1) and c(t)=2/(t+1)
  function solI(t)
    κ = x0[1] / x0[2]
    ρ = (κ + 1) * (t + 1)
    I = x0[1] * (κ + 1 + t) / ((κ + 1) * (t + 1)^2)
    S = x0[2] * (κ + 1 + t) / ((κ + 1) * (t + 1))
    R = N - (κ + 1 + t) / (t + 1) * (x0[2] / (κ + 1) - x0[1] / ρ)
    return I
  end

  function solS(t)
    κ = x0[1] / x0[2]
    ρ = (κ + 1) * (t + 1)
    I = x0[1] * (κ + 1 + t) / ((κ + 1) * (t + 1)^2)
    S = x0[2] * (κ + 1 + t) / ((κ + 1) * (t + 1))
    R = N - (κ + 1 + t) / (t + 1) * (x0[2] / (κ + 1) - x0[1] / ρ)
    return S
  end
  ################################################################
  # Some checks and plots
  # Vectorized solution
  sol_Ih = [solI(t) for t = h:h:T]
  sol_Sh = [solS(t) for t = h:h:T]

  # @show norm(c(vcat(sol_Ih, sol_Sh)), Inf) #check the discretization by hand
  # plot(0:h:T, vcat(x0[1], sol_Ih))
  # plot!(0:h:T, vcat(x0[2], sol_Sh))

  atol, rtol = √eps(), √eps()
  # check the value at the solution:
  kmax = 6 #beyond is tough
  for k = 1:kmax
    local sol_Ih , sol_Sh , h , n , nlp
    n = 10^k
    nlp = dynamicsir(x0 = x0, n = n, T = T)
    h = T / n
    sol_Ih = [solI(t) for t = h:h:T]
    sol_Sh = [solS(t) for t = h:h:T]
    sol_b = [1 / (t + 1) for t = 0:h:T]
    sol_c = [2 / (t + 1) for t = 0:h:T]
    sol = vcat(sol_Ih, sol_Sh, sol_b, sol_c)
    res = norm(cons(nlp, sol), Inf)
    val = obj(nlp, sol)
    if res <= 1e-5 && val <= 1e-10
      @test true
      break
    end
    if k == kmax
      @test false
    end
  end

  n = 10
  nlp = dynamicsir(x0 = x0, n = n, T = T)
  xr = rand(nlp.meta.nvar)
  @test obj(nlp, nlp.meta.x0) == 2.5 #:-)

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
