function _test_unconstrained(;udc = false )
    ubis(x) =  x[1]^2+x[2]^2
    f(y,u) = 0.5 * (ubis - u) * (ubis - u) + 0.5 * y * y
    function f(yu) #:: Union{Gridap.MultiField.MultiFieldFEFunction, Gridap.CellData.GenericCellField}
        y, u = yu
        f(y,u)
    end

    domain = (0,1,0,1)
    n = 2^4
    partition = (n,n)
    @time model = CartesianDiscreteModel(domain,partition)

    order = 1
    @time V0 = TestFESpace(reffe=:Lagrangian, order=order, valuetype=Float64,
                           conformity=:H1, model=model, dirichlet_tags="boundary")
    u(x) = 0.0
    U    = TrialFESpace(V0,u)

    Ypde = U
    Xpde = V0
    Xcon = TestFESpace(reffe=:Lagrangian, order=order, valuetype=Float64,
                       conformity=:H1, model=model)
    Ucon = TrialFESpace(Xcon)
    Ycon = Ucon

    trian = Triangulation(model)
    degree = 2
    @time quad = CellQuadrature(trian,degree)

    Y = MultiFieldFESpace([U, Ucon])
    X = MultiFieldFESpace([V0, Xcon])
    xin = zeros(Gridap.FESpaces.num_free_dofs(Y))
    #@time nlp = GridapPDENLPModel(xin, f, trian, quad, Ypde, Ycon, Xpde, Xcon)
    @time nlp = GridapPDENLPModel(xin, f, trian, quad, Y, X)

    x1 = vcat(rand(Gridap.FESpaces.num_free_dofs(Ypde)), ones(Gridap.FESpaces.num_free_dofs(Ycon))); x=x1;v=x1;

    @time fx = obj(nlp, x1);
    @time gx = grad(nlp, x1);
    @time _fx, _gx = objgrad(nlp, x1);
    @test norm(gx - _gx) <= eps(Float64)
    @test norm(fx - _fx) <= eps(Float64)

    @time  Hx = hess(nlp, x1);
    @time _Hx = hess(nlp, rand(nlp.meta.nvar))
    @test norm(Hx - _Hx) <= eps(Float64) #the hesian is constant
    @time  Hxv = Symmetric(Hx,:L) * v;
    @time _Hxv = hprod(nlp, x1, v);
    @test norm(Hxv - _Hxv) <= eps(Float64)

    #Check the solution:
    cell_xs = get_cell_coordinates(trian)
    midpoint(xs) = sum(xs)/length(xs)
    cell_xm = apply(midpoint, cell_xs) #this is a vector of size num_cells(trian)
    cell_ubis = apply(ubis, cell_xm) #this is a vector of size num_cells(trian)
    #Warning: `interpolate(fs::SingleFieldFESpace, object)` is deprecated, use `interpolate(object, fs::SingleFieldFESpace)` instead.
    solu = get_free_values(Gridap.FESpaces.interpolate(Ucon, cell_ubis))
    soly = get_free_values(zero(Ypde))
    sol = vcat(soly, solu)

    @test obj(nlp, sol) <= 1/n
    @test norm(grad(nlp, sol)) <= 1/n

    #lbfgs solves the problem with too much precision.
    @time _t = lbfgs(nlp, x = x1, rtol = 0.0, atol = 1e-10) #lbfgs modifies the initial point !!
    nn  = Gridap.FESpaces.num_free_dofs(Ypde)
    @test norm(_t.solution[1:nn] - soly, Inf) <= 1/n
    @test obj(nlp, _t.solution) <= 1/n
    @test norm(_t.solution[nn + 1: nlp.meta.nvar] - solu, Inf) <= sqrt(1/n)

    if udc
        println("derivatives check. This may take approx. 5 minutes.")
        #Check derivatives using NLPModels tools:
        #https://github.com/JuliaSmoothOptimizers/NLPModels.jl/blob/master/src/dercheck.jl
        @test gradient_check(nlp) == Dict{Int64,Float64}()
        @test jacobian_check(nlp) == Dict{Tuple{Int64,Int64},Float64}() #not a surprise as there are no constraints...
        H_errs = hessian_check(nlp) #slow
        @test H_errs[0] == Dict{Int, Dict{Tuple{Int,Int}, Float64}}()
        H_errs_fg = hessian_check_from_grad(nlp)
        @test H_errs_fg[0] == Dict{Int, Dict{Tuple{Int,Int}, Float64}}()
    end

    true
end

@test _test_unconstrained(udc = use_derivative_check)
