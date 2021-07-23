###############################################################################
#
# Exercice 10.2.4 (p. 308) in Allaire, Analyse numérique et optimisation, Les éditions de Polytechnique
#
# 2nd test on unconstrained problem. We solve
# min_{u}   0.5 ∫_Ω​ |∇u|^2 - f * u dx
# s.t.      u(x) = 0,     for    x ∈ ∂Ω
#
# The minimizer of this problem is the solution of (using Euler equation):
#
# ∫_Ω​ (∇u ∇v - f*v)dx = 0, ∀ v ∈ Ω
# u = 0, x ∈ ∂Ω
function penalizedpoisson(args...; n = 2^4, kwargs...)
  domain = (0, 1, 0, 1)
  partition = (n, n)
  model = CartesianDiscreteModel(domain, partition)

  order = 1
  valuetype = Float64
  reffe = ReferenceFE(lagrangian, valuetype, order)
  V0 = TestFESpace(model, reffe; conformity = :H1, dirichlet_tags = "boundary")
  y0(x) = 0.0
  U = TrialFESpace(V0, y0)

  Ypde = U
  Xpde = V0

  trian = Triangulation(model)
  degree = 2
  dΩ = Measure(trian, degree)

  w(x) = 1
  function f(yu)
    y = yu
    ∫(0.5 * ∇(y) ⊙ ∇(y) - w * y) * dΩ
  end

  xin = zeros(Gridap.FESpaces.num_free_dofs(Ypde))
  return GridapPDENLPModel(xin, f, trian, dΩ, Ypde, Xpde)
end

function penalizedpoisson_test(; udc = false)
  nlp = penalizedpoisson()
  x1 = vcat(rand(Gridap.FESpaces.num_free_dofs(nlp.pdemeta.Ypde)))
  v = x1

  fx = obj(nlp, x1)
  gx = grad(nlp, x1)
  _fx, _gx = objgrad(nlp, x1)
  @test norm(gx - _gx) <= eps(Float64)
  @test norm(fx - _fx) <= eps(Float64)

  Hx = hess(nlp, x1)
  _Hx = hess(nlp, rand(nlp.meta.nvar))
  @test norm(Hx - _Hx) <= eps(Float64) #the hesian is constant
  Hxv = Symmetric(Hx, :L) * v
  _Hxv = hprod(nlp, x1, v)
  @test norm(Hxv - _Hxv) <= sqrt(eps(Float64))

  if udc
    # Check derivatives using NLPModels tools:
    # https://github.com/JuliaSmoothOptimizers/NLPModels.jl/blob/main/src/dercheck.jl
    @test gradient_check(nlp) == Dict{Int64, Float64}()
    @test jacobian_check(nlp) == Dict{Tuple{Int64, Int64}, Float64}() #not a surprise as there are no constraints...
    H_errs = hessian_check(nlp) #slow
    @test H_errs[0] == Dict{Int, Dict{Tuple{Int, Int}, Float64}}()
    H_errs_fg = hessian_check_from_grad(nlp)
    @test H_errs_fg[0] == Dict{Int, Dict{Tuple{Int, Int}, Float64}}()
  end

  ###############################################################################
  # Gridap resolution:
  # This corresponds to a Poisson equation with Dirichlet and Neumann conditions
  # described here: https://gridap.github.io/Tutorials/stable/pages/t001_poisson/
  domain = (0, 1, 0, 1)
  n = 2^4
  partition = (n, n)
  model = CartesianDiscreteModel(domain, partition)

  order = 1
  valuetype = Float64
  reffe = ReferenceFE(lagrangian, valuetype, order)
  V0 = TestFESpace(model, reffe; conformity = :H1, dirichlet_tags = "boundary")
  y0(x) = 0.0
  U = TrialFESpace(V0, y0)

  trian = Triangulation(model)
  degree = 2
  dΩ = Measure(trian, degree)
  a(u, v) = ∫(∇(v) ⊙ ∇(u)) * dΩ
  w(x) = 1
  b_Ω(v) = ∫(v * w) * dΩ

  op_pde = AffineFEOperator(a, b_Ω, U, V0)
  ls = LUSolver()
  solver = LinearFESolver(ls)
  uh = solve(solver, op_pde)
  sol = get_free_values(uh)

  _fxsol = obj(nlp, sol)
  _ngxsol = norm(grad(nlp, sol))
  _ngxsol <= 1 / n
end
