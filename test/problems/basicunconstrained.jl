function basicunconstrained(args...; n = 2^4, kwargs...)
  domain = (0, 1, 0, 1)
  partition = (n, n)
  model = CartesianDiscreteModel(domain, partition)

  order = 1
  valuetype = Float64
  reffe = ReferenceFE(lagrangian, valuetype, order)
  V0 = TestFESpace(model, reffe; conformity = :H1, dirichlet_tags = "boundary")
  U = TrialFESpace(V0, x -> 0.0)

  Ypde = U
  Xpde = V0
  Xcon = TestFESpace(model, reffe; conformity = :H1)
  Ucon = TrialFESpace(Xcon)
  Ycon = Ucon
  trian = Triangulation(model)
  degree = 2
  dΩ = Measure(trian, degree) # CellQuadrature(trian, degree)

  ubis(x) = x[1]^2 + x[2]^2
  function f(yu)
    y, u = yu
    # 0.5 * (ubis - u) * (ubis - u) + 0.5 * y * y
    ∫(0.5 * (ubis - u) * (ubis - u) + 0.5 * y * y) * dΩ
  end

  Y = MultiFieldFESpace([U, Ucon])
  X = MultiFieldFESpace([V0, Xcon])
  xin = zeros(Gridap.FESpaces.num_free_dofs(Y))
  return GridapPDENLPModel(xin, f, trian, dΩ, Y, X)
end

function basicunconstrained_test(; udc = false)
  n = 10
  nlp = basicunconstrained(n = n)
  ubis(x) = x[1]^2 + x[2]^2
  domain = (0, 1, 0, 1)
  partition = (n, n)
  model = CartesianDiscreteModel(domain, partition)

  order = 1
  valuetype = Float64
  reffe = ReferenceFE(lagrangian, valuetype, order)
  V0 = TestFESpace(model, reffe; conformity = :H1, dirichlet_tags = "boundary")
  U = TrialFESpace(V0, x -> 0.0)

  Ypde = U
  Xpde = V0
  Xcon = TestFESpace(model, reffe; conformity = :H1)
  Ucon = TrialFESpace(Xcon)
  Ycon = Ucon
  trian = Triangulation(model)

  x1 = vcat(rand(Gridap.FESpaces.num_free_dofs(Ypde)), ones(Gridap.FESpaces.num_free_dofs(Ycon)))
  x = x1
  v = x1

  fx = obj(nlp, x1)
  gx = grad(nlp, x1)
  _fx, _gx = objgrad(nlp, x1)
  @test norm(gx - _gx) <= eps(Float64)
  @test norm(fx - _fx) <= eps(Float64)

  Hx = hess(nlp, x1)
  _Hx = hess(nlp, rand(nlp.meta.nvar))
  @test norm(Hx - _Hx) <= eps(Float64) # the hesian is constant
  Hxv = Symmetric(Hx, :L) * v
  _Hxv = hprod(nlp, x1, v)
  @test norm(Hxv - _Hxv) <= eps(Float64)

  # Check the solution:
  cell_xs = get_cell_coordinates(trian)
  midpoint(xs) = sum(xs) / length(xs)
  cell_xm = lazy_map(midpoint, cell_xs) #this is a vector of size num_cells(trian)
  cell_ubis = lazy_map(ubis, cell_xm) #this is a vector of size num_cells(trian)
  # Warning: `interpolate(fs::SingleFieldFESpace, object)` is deprecated, use `interpolate(object, fs::SingleFieldFESpace)` instead.
  solu = get_free_dof_values(Gridap.FESpaces.interpolate(cell_ubis, Ucon))
  soly = get_free_dof_values(zero(Ypde))
  sol = vcat(soly, solu)

  @test obj(nlp, sol) <= 1 / n
  @test norm(grad(nlp, sol)) <= 1 / n

  if udc
    println("derivatives check. This may take approx. 5 minutes.")
    #Check derivatives using NLPModels tools:
    #https://github.com/JuliaSmoothOptimizers/NLPModels.jl/blob/main/src/dercheck.jl
    @test gradient_check(nlp) == Dict{Int64, Float64}()
    @test jacobian_check(nlp) == Dict{Tuple{Int64, Int64}, Float64}() #not a surprise as there are no constraints...
    H_errs = hessian_check(nlp) #slow
    @test H_errs[0] == Dict{Int, Dict{Tuple{Int, Int}, Float64}}()
    H_errs_fg = hessian_check_from_grad(nlp)
    @test H_errs_fg[0] == Dict{Int, Dict{Tuple{Int, Int}, Float64}}()
  end
end
