###############################################################################
#
# This test case consider the optimization of a parameter in a Poisson equation
# with Dirichlet boundary conditions.
#
# Aim:
# * Test mixed problem
# * no integral term in the objective function
# * |k| = 1
#
###############################################################################

function _poissonwithparameteroptim(;udc = false)
    n = 10
    domain = (0,1,0,1)
    partition = (n,n)
    model = CartesianDiscreteModel(domain,partition)

    #We use a manufactured solution of the PDE:
    sol(x) = sin(2*pi*x[1]) * x[2]

    V0 = TestFESpace(
      reffe=:Lagrangian, order=1, valuetype=Float64,
      conformity=:H1, model=model, dirichlet_tags="boundary")

    Ug = TrialFESpace(V0, sol)

    trian = Triangulation(model)
    degree = 2
    quad = CellQuadrature(trian,degree)

    #We deduce the rhs of the Poisson equation with our manufactured solution:
    f(x) = (2*pi^2) * sin(2*pi*x[1]) * x[2]

    function res(k, y, v)
     k[1] * ∇(v)⊙∇(y) - v*f
    end
    t_Ω = FETerm(res, trian, quad)
    op = FEOperator(Ug, V0, t_Ω)

    fk(k) = 0.5*dot(k .- 1.,k .- 1.)
    nrj = NoFETerm(fk) #length(k)=1

    nUg = num_free_dofs(Ug)
    x0  = zeros(nUg + 1)
    nlp = GridapPDENLPModel(x0, nrj, Ug, V0, op)

    @test nlp.nparam == 1
    x1 = vcat(1., rand(nUg))
    @test obj(nlp, x0) == 0.5
    @test obj(nlp, x1) == 0.
    @test grad(nlp, x0) == vcat(-1., zeros(nUg)) #false
    @test grad(nlp, x1) == vcat( 0., zeros(nUg))
    _Hess = sparse(LowerTriangular(vcat(hcat(1., zeros(1,nUg)), hcat(zeros(nUg,1), zeros(nUg,nUg)))))
    @test hess(nlp, x0) == _Hess
    @test hess(nlp, x1) == _Hess

    @test length(cons(nlp, x0)) == nUg
    @test length(cons(nlp, x1)) == nUg

    _J0, _J1 = jac(nlp, x0), jac(nlp, x1)
    @test issparse(_J0)
    @test issparse(_J1)
    @test size(_J0) == (nUg, nUg + 1)
    @test size(_J1) == (nUg, nUg + 1)

    @test hprod(nlp, x1, zeros(nUg + 1)) == zeros(nUg + 1)
    vr = rand(nUg + 1)
    @test hprod(nlp, x1, vr) == hess(nlp, x1) * vr #hess ia diagonal matrix

    l = rand(nUg)
    @test hprod(nlp, x1, l, zeros(nUg + 1)) == zeros(nUg + 1)
    @test hprod(nlp, x1, l, vr) ≈ Symmetric(hess(nlp, x1, l), :L) * vr atol = 1e-14

    if udc #works but slow
        @test gradient_check(nlp) == Dict{Int64,Float64}()
        H_errs_fg = hessian_check_from_grad(nlp)
        @test H_errs_fg[0] == Dict{Int, Dict{Tuple{Int,Int}, Float64}}()
        @test jacobian_check(nlp) == Dict{Tuple{Int64,Int64},Float64}()
        #H_errs = hessian_check(nlp) #slow
        #@test H_errs[0] == Dict{Int, Dict{Tuple{Int,Int}, Float64}}()
    end

    true
end

_poissonwithparameteroptim(udc = use_derivative_check)
