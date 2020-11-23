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

    NT   = NoFETerm()
    NTf  = NoFETerm(f)
    EFT  = EnergyFETerm(f, trian, quad)
    MEFT = MixteEnergyFETerm(f, trian, quad, 1)

    nlp = GridapPDENLPModel(x0, NT, Ypde, Xpde)
    nlp = GridapPDENLPModel(x0, NTf, Ypde, Xpde)
    nlp = GridapPDENLPModel(x0, EFT, Ypde, Xpde)
    nlp = GridapPDENLPModel(x0, MEFT, Ypde, Xpde)
    nlp = GridapPDENLPModel(x0, f, trian, quad, Ypde, Xpde)
    nlp = GridapPDENLPModel(f, trian, quad, Ypde, Xpde)
    nlp = GridapPDENLPModel(x0, NT, Ypde, Xpde, lvar, uvar)
    nlp = GridapPDENLPModel(x0, NTf, Ypde, Xpde, lvar, uvar)
    nlp = GridapPDENLPModel(x0, EFT, Ypde, Xpde, lvar, uvar)
    nlp = GridapPDENLPModel(x0, MEFT, Ypde, Xpde, lvar, uvar)
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
    nlp = GridapPDENLPModel(x0, NT, Ypde, Ycon, Xpde, Xcon, cter)
    nlp = GridapPDENLPModel(x0, NTf, Ypde, Ycon, Xpde, Xcon, cter)
    nlp = GridapPDENLPModel(x0, EFT, Ypde, Ycon, Xpde, Xcon, cter)
    nlp = GridapPDENLPModel(x0, MEFT, Ypde, Ycon, Xpde, Xcon, cter)
    nlp = GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, cter)
    nlp = GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, cter)
    nlp = GridapPDENLPModel(NT, Ypde, Xpde, cter)
    nlp = GridapPDENLPModel(NTf, Ypde, Xpde, cter)
    nlp = GridapPDENLPModel(EFT, Ypde, Xpde, cter)
    nlp = GridapPDENLPModel(MEFT, Ypde, Xpde, cter)
    nlp = GridapPDENLPModel(x0, NT,Ypde, Xpde, cter)
    nlp = GridapPDENLPModel(x0, NTf, Ypde, Xpde, cter)
    nlp = GridapPDENLPModel(x0, EFT, Ypde, Xpde, cter)
    nlp = GridapPDENLPModel(x0, MEFT, Ypde, Xpde, cter)
    nlp = GridapPDENLPModel(x0, NT, Ypde, Ycon, Xpde, Xcon, cter, lcon, ucon, y0 = y0)
    nlp = GridapPDENLPModel(x0, NTf, Ypde, Ycon, Xpde, Xcon, cter, lcon, ucon, y0 = y0)
    nlp = GridapPDENLPModel(x0, EFT, Ypde, Ycon, Xpde, Xcon, cter, lcon, ucon, y0 = y0)
    nlp = GridapPDENLPModel(x0, MEFT, Ypde, Ycon, Xpde, Xcon, cter, lcon, ucon, y0 = y0)
    nlp = GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, cter, lcon, ucon, y0 = y0)
    nlp = GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, cter, lcon, ucon, y0 = y0)
    nlp = GridapPDENLPModel(x0, NT, Ypde, Ycon, Xpde, Xcon, lvar, uvar, cter)
    nlp = GridapPDENLPModel(x0, NTf, Ypde, Ycon, Xpde, Xcon, lvar, uvar, cter)
    nlp = GridapPDENLPModel(x0, EFT, Ypde, Ycon, Xpde, Xcon, lvar, uvar, cter)
    nlp = GridapPDENLPModel(x0, MEFT, Ypde, Ycon, Xpde, Xcon, lvar, uvar, cter)
    nlp = GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, cter)
    nlp = GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, cter)
    nlp = GridapPDENLPModel(x0, NT, Ypde, Ycon, Xpde, Xcon, lvar, uvar, cter, lcon, ucon, y0 = y0)
    nlp = GridapPDENLPModel(x0, NTf, Ypde, Ycon, Xpde, Xcon, lvar, uvar, cter, lcon, ucon, y0 = y0)
    nlp = GridapPDENLPModel(x0, EFT, Ypde, Ycon, Xpde, Xcon, lvar, uvar, cter, lcon, ucon, y0 = y0)
    nlp = GridapPDENLPModel(x0, MEFT, Ypde, Ycon, Xpde, Xcon, lvar, uvar, cter, lcon, ucon, y0 = y0)
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

#Test util_functions.jl
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

@testset "Tests of energy FETerm" begin
    f0(k) = dot(k,k)
    function f1(yu)
        y,u = yu
        y*u
    end
    f1s(y) = y
    f2(yu,k) = f1(yu) + f0(k)
    f2s(y,k) = f1s(y) + f0(k)
    #Test MultiField yu
    x1   = zeros(Gridap.FESpaces.num_free_dofs(Y))
    yu1  = FEFunction(Y, x1)
    cel1 = Gridap.FESpaces.get_cell_values(yu1)
    yuh1 = CellField(Y, cel1)
    #Test SingleField yu
    x2   = rand(Gridap.FESpaces.num_free_dofs(Ypde))
    yu2  = FEFunction(Ypde, x2)
    cel2 = Gridap.FESpaces.get_cell_values(yu2)
    yuh2 = CellField(Ypde, cel2)

    tnrj = NoFETerm()
    @test _obj_integral(tnrj, yu1, []) == 0.
    @test _obj_cell_integral(tnrj, yuh1, []) == 0.
    g = []
    _compute_gradient_k!(g, tnrj, yu1, []) == []
    @test g == []
    @test _obj_integral(tnrj, yu2, []) == 0.
    @test _obj_cell_integral(tnrj, yuh2, ones(2)) == 0.
    g = similar(ones(2))
    _compute_gradient_k!(g, tnrj, yu2, ones(2))
    @test g == zeros(2)

    tnrj = NoFETerm(f0)
    @test _obj_integral(tnrj, yu1, ones(2)) == 2.
    @test _obj_cell_integral(tnrj, yuh1, ones(2)) == 2.
    g = similar(ones(2))
    _compute_gradient_k!(g, tnrj, yu1, ones(2))
    @test g == 2*ones(2)
    @test _obj_integral(tnrj, yu2, ones(2)) == 2.
    @test _obj_cell_integral(tnrj, yuh2, ones(2)) == 2.

    @test_throws AssertionError tnrj = MixteEnergyFETerm(f2, trian, quad, 0)
    @test_throws AssertionError tnrj = MixteEnergyFETerm(f2s, trian, quad, 0)

    tnrj = MixteEnergyFETerm(f2, trian, quad, 3)
    @test sum(_obj_integral(tnrj, yu1, ones(3))) == 3.
    @test _obj_cell_integral(tnrj, yuh1, ones(3)) == 0.75 * ones(4)
    g = similar(ones(3))
    _compute_gradient_k!(g, tnrj, yu1, ones(3))
    @test g == 2*ones(3)
    tnrj = MixteEnergyFETerm(f2s, trian, quad, 3)
    @test typeof(sum(_obj_integral(tnrj, yu2, ones(3)))) <: Number
    @test typeof(sum(_obj_cell_integral(tnrj, yuh2, ones(3)))) <: Number
    g = similar(ones(3))
    _compute_gradient_k!(g, tnrj, yu2, ones(3))
    @test g == 2*ones(3)

    @test_throws DimensionError _obj_cell_integral(tnrj, yuh1, ones(2))
    @test_throws DimensionError _obj_integral(tnrj, yu1, ones(2))
    g = similar(ones(3))
    @test_throws DimensionError _compute_gradient_k!(g, tnrj, yu1, ones(2))

    tnrj = EnergyFETerm(f1, trian, quad)
    @test sum(_obj_integral(tnrj, yu1, [])) == 0.
    @test sum(_obj_cell_integral(tnrj, yuh1, [])) == 0.
    g = []
    _compute_gradient_k!(g, tnrj, yu1, []) == []
    @test g == []
    tnrj = EnergyFETerm(f1s, trian, quad)
    @test typeof(sum(_obj_integral(tnrj, yu2, []))) <: Number
    typeof(sum(_obj_cell_integral(tnrj, yuh2, []))) <: Number

    @test_throws DimensionError _obj_integral(tnrj, yu1, ones(2))
    @test_throws DimensionError _obj_integral(tnrj, yu2, ones(2))
    @test_throws DimensionError _obj_cell_integral(tnrj, yuh1, ones(2))
    @test_throws DimensionError _obj_cell_integral(tnrj, yuh2, ones(2))
    g = similar(ones(2))
    @test_throws DimensionError _compute_gradient_k!(g, tnrj, yu2, ones(2))
end
