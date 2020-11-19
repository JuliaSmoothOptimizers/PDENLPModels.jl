domain = (0,1)
n = 4
partition = (n)
model = CartesianDiscreteModel(domain,partition)
Xpde = TestFESpace(reffe=:Lagrangian, order=1, valuetype=Float64,
                       conformity=:H1, model=model, dirichlet_tags="boundary")
u0(x) = 0.0
Ypde = TrialFESpace(Xpde,u0)

f(x) = dot(x,x)
trian = Triangulation(model)
quad = CellQuadrature(trian, 1)

Xcon = TestFESpace(reffe=:Lagrangian, order=1, valuetype=Float64,
                       conformity=:H1, model=model)
Ycon = TrialFESpace(Xcon)
Y = MultiFieldFESpace([Ypde, Ycon])


@testset "Constructors for GridapPDENLPModel" begin
    x0   = zeros(3)
    lvar, uvar = -ones(3), ones(3)
    badlvar, baduvar = -ones(4), ones(4)
    badx0 = ones(2)

    nlp = GridapPDENLPModel(x0, f, trian, quad, Ypde, Xpde)
    nlp = GridapPDENLPModel(f, trian, quad, Ypde, Xpde)
    nlp = GridapPDENLPModel(x0, f, trian, quad, Ypde, Xpde, lvar, uvar)
    nlp = GridapPDENLPModel(f, trian, quad, Ypde, Xpde, lvar, uvar)

    @test_throws DimensionError GridapPDENLPModel(x0, f, trian, quad, Ypde, Xpde, badlvar, baduvar)
    @test_throws DimensionError GridapPDENLPModel(f, trian, quad, Ypde, Xpde, badlvar, baduvar)
    @test_throws DimensionError GridapPDENLPModel(badx0, f, trian, quad, Ypde, Xpde)
    @test_throws DimensionError GridapPDENLPModel(badx0, f, trian, quad, Ypde, Xpde, lvar, uvar)

    x0   = zeros(8)
    function res(yu,v)
        y,u = yu
        y*v
    end

    term = FETerm(res, trian, quad)
    cter = FEOperator(Y, Xpde, term)
    taff = AffineFETerm(res, v->0*v, trian, quad)
    caff = AffineFEOperator(Y, Xpde, taff)

    lvar, uvar, lcon, ucon, y0 = -ones(8), ones(8), -ones(3), ones(3), zeros(3)
    badlvar, baduvar, badlcon, baducon, bady0 = -ones(3), ones(3), -ones(2), ones(2), zeros(2)
    nlp = GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, cter)
    nlp = GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, cter)
    nlp = GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, cter, lcon, ucon, y0 = y0)
    nlp = GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, cter, lcon, ucon, y0 = y0)
    nlp = GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, cter)
    nlp = GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, cter)
    nlp = GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, cter, lcon, ucon, y0 = y0)
    nlp = GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, cter, lcon, ucon, y0 = y0)

    nlp = GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, caff)
    nlp = GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, caff)
    nlp = GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, caff, lcon, ucon, y0 = y0)
    nlp = GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, caff, lcon, ucon, y0 = y0)
    nlp = GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, caff)
    nlp = GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, caff)
    nlp = GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, caff, lcon, ucon, y0 = y0)
    nlp = GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, caff, lcon, ucon, y0 = y0)

    @test_throws DimensionError GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, cter, badlcon, ucon, y0 = bady0)
    @test_throws DimensionError GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, cter, lcon, baducon, y0 = bady0)
    @test_throws DimensionError GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, caff, lcon, ucon, y0 = bady0)
    @test_throws DimensionError GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, caff, badlcon, baducon, y0 = y0)
    @test_throws DimensionError GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, badlvar, baduvar, cter)
    @test_throws DimensionError GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, badlvar, baduvar, cter)
    @test_throws DimensionError GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, badlvar, baduvar, cter, badlcon, baducon, y0 = y0)
    @test_throws DimensionError GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, badlvar, baduvar, cter, badlcon, ucon, y0 = bady0)
    @test_throws DimensionError GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, badlvar, baduvar, caff, lcon, baducon, y0 = bady0)
    @test_throws DimensionError GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, badlvar, baduvar, caff, lcon, baducon, y0 = bady0)
    @test_throws DimensionError GridapPDENLPModel(badx0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, cter)
    @test_throws DimensionError GridapPDENLPModel(badx0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, cter, lcon, ucon, y0 = y0)
    @test_throws DimensionError GridapPDENLPModel(badx0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, cter)
    @test_throws DimensionError GridapPDENLPModel(badx0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, cter, lcon, ucon, y0 = y0)
    @test_throws DimensionError GridapPDENLPModel(badx0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, caff)
    @test_throws DimensionError GridapPDENLPModel(badx0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, caff, lcon, ucon, y0 = y0)
    @test_throws DimensionError GridapPDENLPModel(badx0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, caff)
    @test_throws DimensionError GridapPDENLPModel(badx0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, caff, lcon, ucon, y0 = y0)
end

x0 = ones(8)
yh,uh = _split_FEFunction(x0, Ypde, Ycon)
@test typeof(yh) <: FEFunctionType && typeof(uh) <: FEFunctionType
x0 = ones(3)
yh,uh = _split_FEFunction(x0, Ypde, nothing)
@test typeof(yh) <: FEFunctionType && typeof(uh) == Nothing
x0 = ones(8)
y,u,k = _split_vector(x0, Ypde, Ycon)
@test y == x0[1:3] && u == x0[4:8]
x0 = ones(3)
y,u,k = _split_vector(x0, Ypde, nothing)
@test y == x0 && u == [] && k == []
x0 = ones(11)
y,u,k = _split_vector(x0, Ypde, Ycon)
@test y == x0[1:3] && u == x0[4:8] && k == x0[9:11]
y,u,k = _split_vector(x0, Ypde, nothing)
@test y == x0[1:3] && u == [] && k == x0[4:11]
