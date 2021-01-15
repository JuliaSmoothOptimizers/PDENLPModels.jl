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
X = MultiFieldFESpace([Xpde, Xcon])

nY = num_free_dofs(Y)
nYpde = num_free_dofs(Ypde)


@testset "Constructors for GridapPDENLPModel" begin
    x0   = zeros(3)
    lvar, uvar = -ones(3), ones(3)
    badlvar, baduvar = -ones(4), ones(4)
    badx0 = ones(2)

    NT   = NoFETerm()
    NTf  = NoFETerm(f)
    EFT  = EnergyFETerm(f, trian, quad)
    MEFT = MixedEnergyFETerm(f, trian, quad, 1)

    nlp = GridapPDENLPModel(x0, NT, Ypde, Xpde)
    nlp = GridapPDENLPModel(x0, NTf, Ypde, Xpde)
    nlp = GridapPDENLPModel(x0, EFT, Ypde, Xpde)
    nlp = GridapPDENLPModel(x0, MEFT, Ypde, Xpde)
    nlp = GridapPDENLPModel(x0, f, trian, quad, Ypde, Xpde)
    #nlp = GridapPDENLPModel(f, trian, quad, Ypde, Xpde)
    nlp = GridapPDENLPModel(x0, NT, Ypde, Xpde, lvar = lvar, uvar = uvar)
    nlp = GridapPDENLPModel(x0, NTf, Ypde, Xpde, lvar = lvar, uvar = uvar)
    nlp = GridapPDENLPModel(x0, EFT, Ypde, Xpde, lvar = lvar, uvar = uvar)
    nlp = GridapPDENLPModel(x0, MEFT, Ypde, Xpde,  lvar = lvar, uvar = uvar)
    nlp = GridapPDENLPModel(x0, f, trian, quad, Ypde, Xpde, lvar = lvar, uvar = uvar)
    #nlp = GridapPDENLPModel(f, trian, quad, Ypde, Xpde, lvar = lvar, uvar = uvar)

    @test_throws DimensionError GridapPDENLPModel(x0, f, trian, quad, Ypde, Xpde, lvar = badlvar, uvar = baduvar)
    #@test_throws DimensionError GridapPDENLPModel(f, trian, quad, Ypde, Xpde, lvar = badlvar, uvar = baduvar)
    @test_throws DimensionError GridapPDENLPModel(badx0, f, trian, quad, Ypde, Xpde)
    @test_throws DimensionError GridapPDENLPModel(badx0, f, trian, quad, Ypde, Xpde, lvar = lvar, uvar = uvar)

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
    #nlp = GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, cter)
    #nlp = GridapPDENLPModel(NT, Ypde, Xpde, cter)
    #nlp = GridapPDENLPModel(NTf, Ypde, Xpde, cter)
    #nlp = GridapPDENLPModel(EFT, Ypde, Xpde, cter)
    #nlp = GridapPDENLPModel(MEFT, Ypde, Xpde, cter)
    nlp = GridapPDENLPModel(x0, NT, Ypde, Xpde, cter)
    nlp = GridapPDENLPModel(x0, NTf, Ypde, Xpde, cter)
    nlp = GridapPDENLPModel(x0, EFT, Ypde, Xpde, cter)
    nlp = GridapPDENLPModel(x0, MEFT, Ypde, Xpde, cter)
    nlp = GridapPDENLPModel(x0, NT, Ypde, Xpde, cter, lvar = lvar, uvar = uvar)
    nlp = GridapPDENLPModel(x0, NTf, Ypde, Xpde, cter, lvar = lvar, uvar = uvar)
    nlp = GridapPDENLPModel(x0, EFT, Ypde, Xpde, cter, lvar = lvar, uvar = uvar)
    nlp = GridapPDENLPModel(x0, MEFT, Ypde, Xpde, cter, lvar = lvar, uvar = uvar)
    nlp = GridapPDENLPModel(x0, NT, Ypde, Ycon, Xpde, Xcon, cter, y0 = y0, lcon = lcon, ucon = ucon)
    nlp = GridapPDENLPModel(x0, NTf, Ypde, Ycon, Xpde, Xcon, cter, y0 = y0, lcon = lcon, ucon = ucon)
    nlp = GridapPDENLPModel(x0, EFT, Ypde, Ycon, Xpde, Xcon, cter, y0 = y0, lcon = lcon, ucon = ucon)
    nlp = GridapPDENLPModel(x0, MEFT, Ypde, Ycon, Xpde, Xcon, cter, y0 = y0, lcon = lcon, ucon = ucon)
    nlp = GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, cter, y0 = y0, lcon = lcon, ucon = ucon)
    nlp = GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, cter, y0 = y0, lcon = lcon, ucon = ucon)
    nlp = GridapPDENLPModel(x0, NT, Ypde, Ycon, Xpde, Xcon, cter, lvar = lvar, uvar = uvar)
    nlp = GridapPDENLPModel(x0, NTf, Ypde, Ycon, Xpde, Xcon, cter, lvar = lvar, uvar = uvar)
    nlp = GridapPDENLPModel(x0, EFT, Ypde, Ycon, Xpde, Xcon, cter, lvar = lvar, uvar = uvar)
    nlp = GridapPDENLPModel(x0, MEFT, Ypde, Ycon, Xpde, Xcon, cter, lvar = lvar, uvar = uvar)
    nlp = GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, cter, lvar = lvar, uvar = uvar)
    #nlp = GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, cter, lvar = lvar, uvar = uvar)
    nlp = GridapPDENLPModel(x0, NT, Ypde, Ycon, Xpde, Xcon, cter, lvar = lvar, uvar = uvar, y0 = y0, lcon = lcon, ucon = ucon)
    nlp = GridapPDENLPModel(x0, NTf, Ypde, Ycon, Xpde, Xcon, cter, lvar = lvar, uvar = uvar, y0 = y0, lcon = lcon, ucon = ucon)
    nlp = GridapPDENLPModel(x0, EFT, Ypde, Ycon, Xpde, Xcon, cter, lvar = lvar, uvar = uvar, y0 = y0, lcon = lcon, ucon = ucon)
    nlp = GridapPDENLPModel(x0, MEFT, Ypde, Ycon, Xpde, Xcon, cter, lvar = lvar, uvar = uvar, y0 = y0, lcon = lcon, ucon = ucon)
    nlp = GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, cter, lvar = lvar, uvar = uvar, y0 = y0, lcon = lcon, ucon = ucon)
    nlp = GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, cter, lvar = lvar, uvar = uvar, y0 = y0, lcon = lcon, ucon = ucon)

    nlp = GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, caff)
    #nlp = GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, caff)
    nlp = GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, caff, y0 = y0, lcon = lcon, ucon = ucon)
    #nlp = GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, caff, y0 = y0, lcon = lcon, ucon = ucon)
    nlp = GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, caff, lvar = lvar, uvar = uvar)
    #nlp = GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, caff, lvar = lvar, uvar = uvar)
    nlp = GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, caff, lvar = lvar, uvar = uvar, y0 = y0, lcon = lcon, ucon = ucon)
    #nlp = GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, caff, lvar = lvar, uvar = uvar, y0 = y0, lcon = lcon, ucon = ucon)

    @test_throws DimensionError GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, cter, y0 = bady0, lcon = badlcon, ucon = ucon)
    #@test_throws DimensionError GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, cter, y0 = bady0, lcon = lcon, ucon = baducon)
    @test_throws DimensionError GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, caff, y0 = bady0, lcon = lcon, ucon = ucon)
    #@test_throws DimensionError GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, caff, y0 = y0, lcon = badlcon, ucon = baducon)
    @test_throws DimensionError GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, cter, lvar = badlvar, uvar = baduvar)
    #@test_throws DimensionError GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, cter, lvar = badlvar, uvar = baduvar)
    @test_throws DimensionError GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, cter, lvar = badlvar, uvar = baduvar, y0 = y0, lcon = badlcon, ucon = baducon)
    #@test_throws DimensionError GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, cter, lvar = badlvar, uvar = baduvar, y0 = bady0, lcon = badlcon, ucon = ucon)
    @test_throws DimensionError GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, caff, lvar = badlvar, uvar = baduvar, y0 = bady0, lcon = lcon, ucon = baducon)
    #@test_throws DimensionError GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, caff, lvar = badlvar, uvar = baduvar, y0 = bady0, lcon = lcon, ucon = baducon)
    @test_throws DimensionError GridapPDENLPModel(badx0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, cter)
    @test_throws DimensionError GridapPDENLPModel(badx0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, cter, y0 = y0, lcon = lcon, ucon = ucon)
    @test_throws DimensionError GridapPDENLPModel(badx0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, cter, lvar = lvar, uvar = uvar)
    @test_throws DimensionError GridapPDENLPModel(badx0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, cter, lvar = lvar, uvar = uvar, y0 = y0, lcon = lcon, ucon = ucon)
    @test_throws DimensionError GridapPDENLPModel(badx0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, caff)
    @test_throws DimensionError GridapPDENLPModel(badx0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, caff, y0 = y0, lcon = lcon, ucon = ucon)
    @test_throws DimensionError GridapPDENLPModel(badx0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, caff, lvar = lvar, uvar = uvar)
    @test_throws DimensionError GridapPDENLPModel(badx0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, caff, lvar = lvar, uvar = uvar, y0 = y0, lcon = lcon, ucon = ucon)
    
    @test_throws ErrorException("Error: Xcon or Ycon are both nothing or must be specified.") GridapPDENLPModel(x0, NT, Ypde, Ycon, Xpde, nothing, cter)
    @test_throws ErrorException("Error: Xcon or Ycon are both nothing or must be specified.") GridapPDENLPModel(x0, NT, Ypde, nothing, Xpde, Xcon, cter)
    @test_throws ErrorException("Error: Xcon or Ycon are both nothing or must be specified.") GridapPDENLPModel(x0, NT, Ypde, Ycon, Xpde, nothing, cter, lvar = lvar, uvar = uvar)
    @test_throws ErrorException("Error: Xcon or Ycon are both nothing or must be specified.") GridapPDENLPModel(x0, NT, Ypde, nothing, Xpde, Xcon, cter, lvar = lvar, uvar = uvar)
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

#Test additional_obj_terms.jl
@testset "Tests of energy FETerm" begin
    f0(k) = dot(k,k)
    function f1(yu)
        y,u = yu
        y*u
    end
    f1s(y) = y
    f2(k,yu) = f1(yu) + f0(k)
    f2s(k,y) = f1s(y) + f0(k)
    #Test MultiField yu
    x1   = zeros(nY)
    yu1  = FEFunction(Y, x1)
    cel1 = Gridap.FESpaces.get_cell_values(yu1)
    yuh1 = CellField(Y, cel1)
    #Test SingleField yu
    x2   = rand(Gridap.FESpaces.num_free_dofs(Ypde))
    yu2  = FEFunction(Ypde, x2)
    cel2 = Gridap.FESpaces.get_cell_values(yu2)
    yuh2 = CellField(Ypde, cel2)

    tnrj = NoFETerm()
    @test _obj_integral(tnrj, [], yu1) == 0.
    @test _obj_cell_integral(tnrj, [], yuh1) == 0.
    @test _compute_gradient_k( tnrj, [], yu1) == []
    g = similar(x1)
    _compute_gradient!(g, tnrj, [], yu1, Y, X)
    @test g== zeros(nY)
    @test _compute_hess_coo(tnrj, [], yu1, Y, X) == findnz(spzeros(nY, nY))
    @test _obj_integral(tnrj, [], yu2) == 0.
    @test _obj_cell_integral(tnrj, ones(2), yuh2) == 0.
    @test _compute_gradient_k(tnrj, ones(2), yu2) == zeros(2)
    g = similar(vcat(ones(2),x2))
    _compute_gradient!(g, tnrj, ones(2), yu2, Ypde, Xpde)
    @test g == zeros(2 + num_free_dofs(Ypde))
    @test _compute_hess_coo(tnrj, ones(2), yu2, Ypde, Xpde) == findnz(spzeros(nYpde, nYpde))

    tnrj = NoFETerm(f0)
    @test _obj_integral(tnrj, ones(2), yu1) == 2.
    @test _obj_cell_integral(tnrj, ones(2), yuh1) == 2.
    @test _compute_gradient_k(tnrj, ones(2), yu1) == 2*ones(2)
    g = similar(vcat(ones(2),x1))
    _compute_gradient!(g, tnrj, ones(2), yu1, Y, X)
    @test g[1:2] == 2 * ones(2)
    @test g[3:nY+2] == zeros(nY)
    _H = _compute_hess_coo(tnrj, ones(2), yu1, Y, X)
    @test _obj_integral(tnrj, ones(2), yu2) == 2.
    @test _obj_cell_integral(tnrj, ones(2), yuh2) == 2.

    @test_throws AssertionError tnrj = MixedEnergyFETerm(f2, trian, quad, 0)
    @test_throws AssertionError tnrj = MixedEnergyFETerm(f2s, trian, quad, 0)

    tnrj = MixedEnergyFETerm(f2, trian, quad, 3)
    @test sum(_obj_integral(tnrj, ones(3), yu1)) == 3.
    @test _obj_cell_integral(tnrj, ones(3), yuh1) == 0.75 * ones(4)
    @test _compute_gradient_k(tnrj, ones(3), yu1) == 2*ones(3)
    g = similar(vcat(ones(3),x1))
    _compute_gradient!(g, tnrj, ones(3), yu1, Y, X)
    @test g[1:3] == 2*ones(3)
    _H = _compute_hess_coo(tnrj, ones(3), yu1, Y, X)
    tnrj = MixedEnergyFETerm(f2s, trian, quad, 3)
    @test typeof(sum(_obj_integral(tnrj, ones(3), yu2))) <: Number
    @test typeof(sum(_obj_cell_integral(tnrj, ones(3), yuh2))) <: Number
    @test _compute_gradient_k(tnrj, ones(3), yu2) == 2*ones(3)
    g = similar(vcat(ones(3),x2))
    _compute_gradient!(g, tnrj, ones(3), yu2, Ypde, Xpde)
    @test g[1:3] == 2*ones(3)
    _H = _compute_hess_coo(tnrj, ones(3), yu2, Ypde, Xpde)

    @test_throws DimensionError _obj_cell_integral(tnrj, ones(2), yuh1)
    @test_throws DimensionError _obj_integral(tnrj, ones(2), yu1)
    @test_throws DimensionError _compute_gradient_k(tnrj, ones(2), yu1)

    tnrj = EnergyFETerm(f1, trian, quad)
    @test sum(_obj_integral(tnrj, [], yu1)) == 0.
    @test sum(_obj_cell_integral(tnrj, [], yuh1)) == 0.
    @test _compute_gradient_k(tnrj, [], yu1) == []
    g = similar(x1)
    _compute_gradient!(g, tnrj, [], yu1, Y, X)
    @test length(g) == nY
    _H = _compute_hess_coo(tnrj, [], yu1, Y, X)

    tnrj = EnergyFETerm(f1s, trian, quad)
    @test typeof(sum(_obj_integral(tnrj, [], yu2))) <: Number
    typeof(sum(_obj_cell_integral(tnrj, [], yuh2))) <: Number

    @test_throws DimensionError _obj_integral(tnrj, ones(2), yu1)
    @test_throws DimensionError _obj_integral(tnrj, ones(2), yu2)
    @test_throws DimensionError _obj_cell_integral(tnrj, ones(2), yuh1)
    @test_throws DimensionError _obj_cell_integral(tnrj, ones(2), yuh2)
    @test_throws DimensionError _compute_gradient_k(tnrj, ones(2), yu2)
end

@testset "Functions used to create bounds in constructors" begin
    #Domain
    domain = (-1,1,-1,1)
    n = 2
    partition = (n,n)
    model = CartesianDiscreteModel(domain,partition)

    #Definition of the spaces:
    Xpde = TestFESpace(reffe=:Lagrangian, conformity=:H1, valuetype=Float64, model=model, order=2, dirichlet_tags="boundary")

    y0(x) = 0.0
    Ypde    = TrialFESpace(Xpde, y0)

    Xcon = TestFESpace(
            reffe=:Lagrangian, order=1, valuetype=Float64,
            conformity=:H1, model=model)
    Ycon = TrialFESpace(Xcon)
    Y = MultiFieldFESpace([Ypde, Ycon])

    #Integration machinery
    trian = Triangulation(model)
    degree = 1
    quad = CellQuadrature(trian,degree)

    for T in (Float16, Float32, Float64)
        #Example 0:
        lvar, uvar = bounds_functions_to_vectors(Y, Ycon, Ypde, trian, -T(Inf) * ones(T, 9), 
                                                                        T(Inf) * ones(T, 9), 
                                                                       -T(Inf) * ones(T, 9), 
                                                                        T(Inf) * ones(T, 9))
        @test lvar == -Inf * ones(18)
        @test uvar ==  Inf * ones(18)
        @test eltype(lvar) == T
        @test eltype(uvar) == T

        #Example 0bis:
        lvar, uvar = bounds_functions_to_vectors(Y, nothing, Ypde, trian, -T(Inf) * ones(T, 9), 
                                                                           T(Inf) * ones(T, 9), 
                                                                           nothing, 
                                                                           nothing)
        @test lvar == -Inf * ones(9)
        @test uvar ==  Inf * ones(9)
        @test eltype(lvar) == T
        @test eltype(uvar) == T

        #Example 1:
        umin(x) = T(x[1]+x[2])
        umax(x) = T(x[1]^2+x[2]^2)
        lvar, uvar = bounds_functions_to_vectors(Y, Ycon, Ypde, trian, -T(Inf) * ones(T, 9), T(Inf) * ones(T, 9), umin, umax)
        @test lvar == vcat(-T(Inf) * ones(T, 9), [-1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        @test uvar == vcat( T(Inf) * ones(T, 9), 0.5 * ones(T, 9))
        @test eltype(lvar) == T
        @test eltype(uvar) == T

        #Example 1bis:
        umin(x) = T(x[1]+x[2])
        umax(x) = T(x[1]^2+x[2]^2)
        lvar, uvar = bounds_functions_to_vectors(Y, nothing, Ypde, trian, umin, umax, nothing, nothing)
        @test lvar == [1.0, 0.0, 0.0, 1.0, 1.0, -1.0, 0.0, 0.0, 1.0]
        @test uvar == 0.5 * ones(T, 9)
        @test eltype(lvar) == T
        @test eltype(uvar) == T

        #Example 2:
        umin(x) = [T(x[1]+x[2])]
        umax(x) = [T(x[1]^2+x[2]^2)]
        lvar, uvar = bounds_functions_to_vectors(Y, Ycon, Ypde, trian, umin, umax, umin, umax)
        @test lvar == [1.0, 0.0, 0.0, 1.0, 1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
        @test uvar == 0.5 * ones(T, 18)
        @test eltype(lvar) == T
        @test eltype(uvar) == T

    end

end