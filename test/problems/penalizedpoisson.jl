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
    w(x) = 1
    function f(yu) #:: Union{Gridap.MultiField.MultiFieldFEFunction, Gridap.CellData.GenericCellField}
        y = yu
        0.5 * ∇(y) ⊙ ∇(y) - w * y
    end

    domain = (0, 1, 0, 1)
    partition = (n, n)
    @time model = CartesianDiscreteModel(domain, partition)

    order = 1
    @time V0 = TestFESpace(
        reffe = :Lagrangian,
        order = order,
        valuetype = Float64,
        conformity = :H1,
        model = model,
        dirichlet_tags = "boundary",
    )
    y0(x) = 0.0
    U = TrialFESpace(V0, y0)

    Ypde = U
    Xpde = V0

    trian = Triangulation(model)
    degree = 2
    @time quad = CellQuadrature(trian, degree)

    xin = zeros(Gridap.FESpaces.num_free_dofs(Ypde))
    return GridapPDENLPModel(xin, f, trian, quad, Ypde, Xpde)
end


function penalizedpoisson_test(; udc = false)

    nlp = penalizedpoisson()
    x1 = vcat(rand(Gridap.FESpaces.num_free_dofs(nlp.Ypde)))
    v = x1

    @time fx = obj(nlp, x1)
    @time gx = grad(nlp, x1)
    @time _fx, _gx = objgrad(nlp, x1)
    @test norm(gx - _gx) <= eps(Float64)
    @test norm(fx - _fx) <= eps(Float64)

    @time Hx = hess(nlp, x1)
    @time _Hx = hess(nlp, rand(nlp.meta.nvar))
    @test norm(Hx - _Hx) <= eps(Float64) #the hesian is constant
    @time Hxv = Symmetric(Hx, :L) * v
    @time _Hxv = hprod(nlp, x1, v)
    @test norm(Hxv - _Hxv) <= sqrt(eps(Float64))

    if udc
        #Check derivatives using NLPModels tools:
        #https://github.com/JuliaSmoothOptimizers/NLPModels.jl/blob/master/src/dercheck.jl
        @test gradient_check(nlp) == Dict{Int64,Float64}()
        @test jacobian_check(nlp) == Dict{Tuple{Int64,Int64},Float64}() #not a surprise as there are no constraints...
        H_errs = hessian_check(nlp) #slow
        @test H_errs[0] == Dict{Int,Dict{Tuple{Int,Int},Float64}}()
        H_errs_fg = hessian_check_from_grad(nlp)
        @test H_errs_fg[0] == Dict{Int,Dict{Tuple{Int,Int},Float64}}()
    end

    ###############################################################################
    #Gridap resolution:
    #This corresponds to a Poisson equation with Dirichlet and Neumann conditions
    #described here: https://gridap.github.io/Tutorials/stable/pages/t001_poisson/
    domain = (0, 1, 0, 1)
    n = 2^4
    partition = (n, n)
    model = CartesianDiscreteModel(domain, partition)

    order = 1
    @time V0 = TestFESpace(
        reffe = :Lagrangian,
        order = order,
        valuetype = Float64,
        conformity = :H1,
        model = model,
        dirichlet_tags = "boundary",
    )
    y0(x) = 0.0
    U = TrialFESpace(V0, y0)

    trian = Triangulation(model)
    degree = 2
    quad = CellQuadrature(trian, degree)
    a(u, v) = ∇(v) ⊙ ∇(u)
    w(x) = 1
    b_Ω(v) = v * w
    t_Ω = AffineFETerm(a, b_Ω, trian, quad)

    op_pde = AffineFEOperator(U, V0, t_Ω)
    ls = LUSolver()
    solver = LinearFESolver(ls)
    uh = solve(solver, op_pde)
    sol = get_free_values(uh)

    @time _fxsol = obj(nlp, sol)
    @time _ngxsol = norm(grad(nlp, sol))
    @test _ngxsol <= 1 / n
    ###############################################################################

    #=
    @time _tron = tron(nlp, x = copy(x1), rtol = 0.0, atol = 1/n, max_time = 120.)
      
    @test norm(_tron.solution - sol, Inf) <= 2/n
    @time _fxtron  = obj(nlp, _tron.solution)
    @time _ngxtron = norm(grad(nlp, _tron.solution))
      
    @time _stats = ipopt(nlp, x0 = copy(x1))
    =#

    true
end
