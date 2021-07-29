domain = (0, 1)
n = 4
partition = (n)
model = CartesianDiscreteModel(domain, partition)
Xpde = TestFESpace(
  model,
  ReferenceFE(lagrangian, Float64, 1),
  conformity = :H1,
  dirichlet_tags = "boundary",
)
u0(x) = 0.0
Ypde = TrialFESpace(Xpde, u0)

f(x) = dot(x, x)
fxh = x -> ∫(f(x)) * dΩ
function fyuh(y, u)
  ∫(y * 1.0) * dΩ
end
fkxh = (k, x) -> ∫(f(x)) * dΩ
function fkyuh(k, y, u)
  ∫(y * 1.0) * dΩ
end

trian = Triangulation(model)
dΩ = Measure(trian, 1)

Xcon = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), conformity = :H1)
Ycon = TrialFESpace(Xcon)
Y = MultiFieldFESpace([Ypde, Ycon])
X = MultiFieldFESpace([Xpde, Xcon])

nY = num_free_dofs(Y)
nYpde = num_free_dofs(Ypde)

@testset "VoidFESpace" begin
  fespace = PDENLPModels.VoidMultiFieldFESpace()
  @test fespace != nothing
  @test PDENLPModels.num_fields(fespace) == 0
  @test PDENLPModels._fespace_to_multifieldfespace(Y) == Y
  @test PDENLPModels._fespace_to_multifieldfespace(fespace) == fespace
  @test PDENLPModels._fespace_to_multifieldfespace(PDENLPModels.VoidFESpace()) != nothing # PDENLPModels.VoidMultiFieldFESpace()
end

@testset "API for unconstrained problems" begin
  x0 = zeros(3)
  nlp = GridapPDENLPModel(x0, NoFETerm(), Ypde, Xpde)
  @test nlp.meta.ncon == 0
  @test nlp.meta.nnzj == 0
  @test cons(nlp, x0) == []
  @test jac_coord(nlp, x0) == []
  @test jac_structure(nlp) == ([], [])
  @test jac(nlp, x0) == zeros(0, 3)
end

@testset "Constructors for GridapPDENLPModel" begin
  x0 = zeros(3)
  xk0 = zeros(4)
  lvar, uvar = -ones(3), ones(3)
  lvark, uvark = -ones(4), ones(4)
  badlvar, baduvar = -ones(5), ones(5)
  badx0 = ones(2)

  NT = NoFETerm()
  NTf = NoFETerm(f)
  EFT = EnergyFETerm(fxh, trian, Ypde)
  MEFT = MixedEnergyFETerm(fkxh, trian, 1, Ypde)

  EFTmixed = EnergyFETerm(fyuh, trian, Ypde, Ycon)
  MEFTmixed = MixedEnergyFETerm(fkyuh, trian, 1, Ypde, Ycon)

  nlp = GridapPDENLPModel(x0, NT, Ypde, Xpde)
  nlp = GridapPDENLPModel(x0, NTf, Ypde, Xpde)
  nlp = GridapPDENLPModel(x0, EFT, Ypde, Xpde)
  nlp = GridapPDENLPModel(xk0, MEFT, Ypde, Xpde)
  nlp = GridapPDENLPModel(x0, fxh, trian, Ypde, Xpde)
  # nlp = GridapPDENLPModel(fxh, trian, Ypde, Xpde)
  nlp = GridapPDENLPModel(x0, NT, Ypde, Xpde, lvar = lvar, uvar = uvar)
  nlp = GridapPDENLPModel(x0, NTf, Ypde, Xpde, lvar = lvar, uvar = uvar)
  nlp = GridapPDENLPModel(x0, EFT, Ypde, Xpde, lvar = lvar, uvar = uvar)
  nlp = GridapPDENLPModel(xk0, MEFT, Ypde, Xpde, lvar = lvark, uvar = uvark)
  nlp = GridapPDENLPModel(x0, fxh, trian, Ypde, Xpde, lvar = lvar, uvar = uvar)
  # nlp = GridapPDENLPModel(fxh, trian, Ypde, Xpde, lvar = lvar, uvar = uvar)

  @test_throws DimensionError GridapPDENLPModel(
    x0,
    fxh,
    trian,
    Ypde,
    Xpde,
    lvar = badlvar,
    uvar = baduvar,
  )
  #@test_throws DimensionError GridapPDENLPModel(fxh, trian, Ypde, Xpde, lvar = badlvar, uvar = baduvar)
  @test_throws DimensionError GridapPDENLPModel(badx0, fxh, trian, Ypde, Xpde)
  @test_throws DimensionError GridapPDENLPModel(
    badx0,
    fxh,
    trian,
    Ypde,
    Xpde,
    lvar = lvar,
    uvar = uvar,
  )

  x0 = zeros(8)
  xk0 = zeros(9)
  x0y = zeros(3)
  xk0y = zeros(4)
  function res(y, u, v)
    ∫(y * v) * dΩ
  end
  function res(k, y, u, v)
    ∫(y * v) * dΩ
  end
  ctermixed = FEOperator(res, Y, Xpde)
  function res(k::AbstractVector, y, v)
    ∫(y * v) * dΩ
  end
  function res(y, v)
    ∫(y * v) * dΩ
  end
  cter = FEOperator(res, Y, Xpde)
  function resff(yu, v)
    y, u = yu
    ∫(y * v) * dΩ
  end
  rhs(v) = ∫(v * 0.0) * dΩ
  caff = AffineFEOperator(resff, rhs, Y, Xpde)

  lvary, uvary, lvaru, uvaru = -ones(3), ones(3), -ones(5), ones(5)
  lvaryk, uvaryk = -ones(4), ones(4)
  lcon, ucon, y0 = -ones(3), ones(3), zeros(3)
  badlvary, baduvary, badlvaru, baduvaru = -ones(4), ones(5), -ones(3), ones(3)
  badlcon, baducon, bady0 = -ones(2), ones(2), zeros(2)
  nlp = GridapPDENLPModel(x0, NT, Ypde, Ycon, Xpde, Xcon, ctermixed)
  nlp = GridapPDENLPModel(x0, NTf, Ypde, Ycon, Xpde, Xcon, ctermixed)
  nlp = GridapPDENLPModel(x0, EFTmixed, Ypde, Ycon, Xpde, Xcon, ctermixed)
  nlp = GridapPDENLPModel(xk0, MEFTmixed, Ypde, Ycon, Xpde, Xcon, ctermixed)
  nlp = GridapPDENLPModel(x0, fyuh, trian, Ypde, Ycon, Xpde, Xcon, ctermixed)
  #nlp = GridapPDENLPModel(fxh, trian, Ypde, Ycon, Xpde, Xcon, cter)
  #nlp = GridapPDENLPModel(NT, Ypde, Xpde, cter)
  #nlp = GridapPDENLPModel(NTf, Ypde, Xpde, cter)
  #nlp = GridapPDENLPModel(EFT, Ypde, Xpde, cter)
  #nlp = GridapPDENLPModel(MEFT, Ypde, Xpde, cter)
  nlp = GridapPDENLPModel(x0y, NT, Ypde, Xpde, cter)
  nlp = GridapPDENLPModel(x0y, NTf, Ypde, Xpde, cter)
  nlp = GridapPDENLPModel(x0y, EFT, Ypde, Xpde, cter)
  nlp = GridapPDENLPModel(xk0y, MEFT, Ypde, Xpde, cter)
  nlp = GridapPDENLPModel(x0y, NT, Ypde, Xpde, cter, lvar = lvary, uvar = uvary)
  nlp = GridapPDENLPModel(x0y, NTf, Ypde, Xpde, cter, lvar = lvary, uvar = uvary)
  nlp = GridapPDENLPModel(x0y, EFT, Ypde, Xpde, cter, lvar = lvary, uvar = uvary)
  nlp = GridapPDENLPModel(xk0y, MEFT, Ypde, Xpde, cter, lvar = lvaryk, uvar = uvaryk)
  nlp =
    GridapPDENLPModel(x0, NT, Ypde, Ycon, Xpde, Xcon, ctermixed, y0 = y0, lcon = lcon, ucon = ucon)
  nlp =
    GridapPDENLPModel(x0, NTf, Ypde, Ycon, Xpde, Xcon, ctermixed, y0 = y0, lcon = lcon, ucon = ucon)
  nlp = GridapPDENLPModel(
    x0,
    EFTmixed,
    Ypde,
    Ycon,
    Xpde,
    Xcon,
    ctermixed,
    y0 = y0,
    lcon = lcon,
    ucon = ucon,
  )
  nlp = GridapPDENLPModel(
    xk0,
    MEFTmixed,
    Ypde,
    Ycon,
    Xpde,
    Xcon,
    ctermixed,
    y0 = y0,
    lcon = lcon,
    ucon = ucon,
  )
  nlp = GridapPDENLPModel(
    x0,
    fyuh,
    trian,
    Ypde,
    Ycon,
    Xpde,
    Xcon,
    ctermixed,
    y0 = y0,
    lcon = lcon,
    ucon = ucon,
  )
  nlp = GridapPDENLPModel(
    x0,
    fyuh,
    trian,
    Ypde,
    Ycon,
    Xpde,
    Xcon,
    ctermixed,
    y0 = y0,
    lcon = lcon,
    ucon = ucon,
  )
  nlp = GridapPDENLPModel(
    x0,
    NT,
    Ypde,
    Ycon,
    Xpde,
    Xcon,
    ctermixed,
    lvary = lvary,
    uvary = uvary,
    lvaru = lvaru,
    uvaru = uvaru,
  )
  nlp = GridapPDENLPModel(
    x0,
    NTf,
    Ypde,
    Ycon,
    Xpde,
    Xcon,
    ctermixed,
    lvary = lvary,
    uvary = uvary,
    lvaru = lvaru,
    uvaru = uvaru,
  )
  nlp = GridapPDENLPModel(
    x0,
    EFTmixed,
    Ypde,
    Ycon,
    Xpde,
    Xcon,
    ctermixed,
    lvary = lvary,
    uvary = uvary,
    lvaru = lvaru,
    uvaru = uvaru,
  )
  nlp = GridapPDENLPModel(
    xk0,
    MEFTmixed,
    Ypde,
    Ycon,
    Xpde,
    Xcon,
    ctermixed,
    lvary = lvary,
    uvary = uvary,
    lvaru = lvaru,
    uvaru = uvaru,
  )
  nlp = GridapPDENLPModel(
    x0,
    fyuh,
    trian,
    Ypde,
    Ycon,
    Xpde,
    Xcon,
    ctermixed,
    lvary = lvary,
    uvary = uvary,
    lvaru = lvaru,
    uvaru = uvaru,
  )
  #nlp = GridapPDENLPModel(f, trian, Ypde, Ycon, Xpde, Xcon, ctermixed, lvary = lvary, uvary = uvary, lvaru = lvaru, uvaru = uvaru)
  nlp = GridapPDENLPModel(
    x0,
    NT,
    Ypde,
    Ycon,
    Xpde,
    Xcon,
    ctermixed,
    lvary = lvary,
    uvary = uvary,
    lvaru = lvaru,
    uvaru = uvaru,
    y0 = y0,
    lcon = lcon,
    ucon = ucon,
  )
  nlp = GridapPDENLPModel(
    x0,
    NTf,
    Ypde,
    Ycon,
    Xpde,
    Xcon,
    ctermixed,
    lvary = lvary,
    uvary = uvary,
    lvaru = lvaru,
    uvaru = uvaru,
    y0 = y0,
    lcon = lcon,
    ucon = ucon,
  )
  nlp = GridapPDENLPModel(
    x0,
    EFTmixed,
    Ypde,
    Ycon,
    Xpde,
    Xcon,
    ctermixed,
    lvary = lvary,
    uvary = uvary,
    lvaru = lvaru,
    uvaru = uvaru,
    y0 = y0,
    lcon = lcon,
    ucon = ucon,
  )
  nlp = GridapPDENLPModel(
    xk0,
    MEFTmixed,
    Ypde,
    Ycon,
    Xpde,
    Xcon,
    ctermixed,
    lvary = lvary,
    uvary = uvary,
    lvaru = lvaru,
    uvaru = uvaru,
    y0 = y0,
    lcon = lcon,
    ucon = ucon,
  )
  nlp = GridapPDENLPModel(
    x0,
    fyuh,
    trian,
    Ypde,
    Ycon,
    Xpde,
    Xcon,
    ctermixed,
    lvary = lvary,
    uvary = uvary,
    lvaru = lvaru,
    uvaru = uvaru,
    y0 = y0,
    lcon = lcon,
    ucon = ucon,
  )
  nlp = GridapPDENLPModel(
    x0,
    fyuh,
    trian,
    Ypde,
    Ycon,
    Xpde,
    Xcon,
    ctermixed,
    lvary = lvary,
    uvary = uvary,
    lvaru = lvaru,
    uvaru = uvaru,
    y0 = y0,
    lcon = lcon,
    ucon = ucon,
  )
  nlp = GridapPDENLPModel(x0, fyuh, trian, Ypde, Ycon, Xpde, Xcon, caff)
  #nlp = GridapPDENLPModel(fyuh, trian, Ypde, Ycon, Xpde, Xcon, caff)
  nlp = GridapPDENLPModel(
    x0,
    fyuh,
    trian,
    Ypde,
    Ycon,
    Xpde,
    Xcon,
    caff,
    y0 = y0,
    lcon = lcon,
    ucon = ucon,
  )
  #nlp = GridapPDENLPModel(fyuh, trian, Ypde, Ycon, Xpde, Xcon, caff, y0 = y0, lcon = lcon, ucon = ucon)
  nlp = GridapPDENLPModel(
    x0,
    fyuh,
    trian,
    Ypde,
    Ycon,
    Xpde,
    Xcon,
    caff,
    lvary = lvary,
    uvary = uvary,
    lvaru = lvaru,
    uvaru = uvaru,
  )
  #nlp = GridapPDENLPModel(fyuh, trian, Ypde, Ycon, Xpde, Xcon, caff, lvary = lvary, uvary = uvary, lvaru = lvaru, uvaru = uvaru)
  nlp = GridapPDENLPModel(
    x0,
    fyuh,
    trian,
    Ypde,
    Ycon,
    Xpde,
    Xcon,
    caff,
    lvary = lvary,
    uvary = uvary,
    lvaru = lvaru,
    uvaru = uvaru,
    y0 = y0,
    lcon = lcon,
    ucon = ucon,
  )
  #nlp = GridapPDENLPModel(fxh, trian, Ypde, Ycon, Xpde, Xcon, caff, lvary = lvary, uvary = uvary, lvaru = lvaru, uvaru = uvaru, y0 = y0, lcon = lcon, ucon = ucon)
  @test_throws DimensionError GridapPDENLPModel(
    x0,
    fyuh,
    trian,
    Ypde,
    Ycon,
    Xpde,
    Xcon,
    ctermixed,
    y0 = bady0,
    lcon = badlcon,
    ucon = ucon,
  )
  #@test_throws DimensionError GridapPDENLPModel(fyuh, trian, Ypde, Ycon, Xpde, Xcon, ctermixed, y0 = bady0, lcon = lcon, ucon = baducon)
  @test_throws DimensionError GridapPDENLPModel(
    x0,
    fyuh,
    trian,
    Ypde,
    Ycon,
    Xpde,
    Xcon,
    caff,
    y0 = bady0,
    lcon = lcon,
    ucon = ucon,
  )
  #@test_throws DimensionError GridapPDENLPModel(fyuh, trian, Ypde, Ycon, Xpde, Xcon, caff, y0 = y0, lcon = badlcon, ucon = baducon)
  @test_throws DimensionError GridapPDENLPModel(
    x0,
    fyuh,
    trian,
    Ypde,
    Ycon,
    Xpde,
    Xcon,
    ctermixed,
    lvary = badlvary,
    uvary = baduvary,
  )
  #@test_throws DimensionError GridapPDENLPModel(fyuh, trian, Ypde, Ycon, Xpde, Xcon, ctermixed, lvary = badlvary, uvary = baduvary)
  @test_throws DimensionError GridapPDENLPModel(
    x0,
    fyuh,
    trian,
    Ypde,
    Ycon,
    Xpde,
    Xcon,
    ctermixed,
    lvaru = badlvaru,
    uvaru = baduvaru,
  )
  #@test_throws DimensionError GridapPDENLPModel(fyuh, trian, Ypde, Ycon, Xpde, Xcon, ctermixed, lvaru = badlvaru, uvaru = baduvaru)
  @test_throws DimensionError GridapPDENLPModel(
    x0,
    fyuh,
    trian,
    Ypde,
    Ycon,
    Xpde,
    Xcon,
    ctermixed,
    lvary = badlvary,
    uvary = baduvary,
    y0 = y0,
    lcon = badlcon,
    ucon = baducon,
  )
  #@test_throws DimensionError GridapPDENLPModel(fyuh, trian, Ypde, Ycon, Xpde, Xcon, ctermixed, lvary = badlvary, uvary = baduvary, y0 = bady0, lcon = badlcon, ucon = ucon)
  @test_throws DimensionError GridapPDENLPModel(
    x0,
    fyuh,
    trian,
    Ypde,
    Ycon,
    Xpde,
    Xcon,
    caff,
    lvary = badlvary,
    uvary = baduvary,
    y0 = bady0,
    lcon = lcon,
    ucon = baducon,
  )
  #@test_throws DimensionError GridapPDENLPModel(fyuh, trian, Ypde, Ycon, Xpde, Xcon, caff, lvary = badlvary, uvary = baduvary, y0 = bady0, lcon = lcon, ucon = baducon)
  @test_throws DimensionError GridapPDENLPModel(badx0, fyuh, trian, Ypde, Ycon, Xpde, Xcon, cter)
  @test_throws DimensionError GridapPDENLPModel(
    badx0,
    fyuh,
    trian,
    Ypde,
    Ycon,
    Xpde,
    Xcon,
    ctermixed,
    y0 = y0,
    lcon = lcon,
    ucon = ucon,
  )
  @test_throws DimensionError GridapPDENLPModel(
    badx0,
    fyuh,
    trian,
    Ypde,
    Ycon,
    Xpde,
    Xcon,
    ctermixed,
    lvary = lvary,
    uvary = uvary,
    lvaru = lvaru,
    uvaru = uvaru,
  )
  @test_throws DimensionError GridapPDENLPModel(
    badx0,
    fyuh,
    trian,
    Ypde,
    Ycon,
    Xpde,
    Xcon,
    ctermixed,
    lvary = lvary,
    uvary = uvary,
    lvaru = lvaru,
    uvaru = uvaru,
    y0 = y0,
    lcon = lcon,
    ucon = ucon,
  )
  @test_throws DimensionError GridapPDENLPModel(badx0, fyuh, trian, Ypde, Ycon, Xpde, Xcon, caff)
  @test_throws DimensionError GridapPDENLPModel(
    badx0,
    fyuh,
    trian,
    Ypde,
    Ycon,
    Xpde,
    Xcon,
    caff,
    y0 = y0,
    lcon = lcon,
    ucon = ucon,
  )
  @test_throws DimensionError GridapPDENLPModel(
    badx0,
    fyuh,
    trian,
    Ypde,
    Ycon,
    Xpde,
    Xcon,
    caff,
    lvary = lvary,
    uvary = uvary,
    lvaru = lvaru,
    uvaru = uvaru,
  )
  @test_throws DimensionError GridapPDENLPModel(
    badx0,
    fyuh,
    trian,
    Ypde,
    Ycon,
    Xpde,
    Xcon,
    caff,
    lvary = lvary,
    uvary = uvary,
    lvaru = lvaru,
    uvaru = uvaru,
    y0 = y0,
    lcon = lcon,
    ucon = ucon,
  )

  @test_throws ErrorException("Error: Xcon or Ycon are both nothing or must be specified.") GridapPDENLPModel(
    x0,
    NT,
    Ypde,
    Ycon,
    Xpde,
    PDENLPModels.VoidFESpace(),
    cter,
  )
  @test_throws ErrorException("Error: Xcon or Ycon are both nothing or must be specified.") GridapPDENLPModel(
    x0,
    NT,
    Ypde,
    PDENLPModels.VoidFESpace(),
    Xpde,
    Xcon,
    cter,
  )
  @test_throws ErrorException("Error: Xcon or Ycon are both nothing or must be specified.") GridapPDENLPModel(
    x0,
    NT,
    Ypde,
    Ycon,
    Xpde,
    PDENLPModels.VoidFESpace(),
    cter,
    lvary = lvary,
    uvary = uvary,
    lvaru = lvaru,
    uvaru = uvaru,
  )
  @test_throws ErrorException("Error: Xcon or Ycon are both nothing or must be specified.") GridapPDENLPModel(
    x0,
    NT,
    Ypde,
    PDENLPModels.VoidFESpace(),
    Xpde,
    Xcon,
    cter,
    lvary = lvary,
    uvary = uvary,
    lvaru = lvaru,
    uvaru = uvaru,
  )

  io = IOBuffer()
  show(io, nlp)
end

#Test util_functions.jl
x0 = ones(8)
yh, uh = _split_FEFunction(x0, Ypde, Ycon)
@test typeof(yh) <: FEFunctionType && typeof(uh) <: FEFunctionType
x0 = ones(3)
yh, uh = _split_FEFunction(x0, Ypde, PDENLPModels.VoidFESpace())
@test typeof(yh) <: FEFunctionType && typeof(uh) == Nothing
x0 = ones(8)
y, u, k = _split_vector(x0, Ypde, Ycon)
@test y == x0[1:3] && u == x0[4:8]
x0 = ones(3)
y, u, k = _split_vector(x0, Ypde, PDENLPModels.VoidFESpace())
@test y == x0 && u == [] && k == []
x0 = ones(11)
y, u, k = _split_vector(x0, Ypde, Ycon)
@test y == x0[1:3] && u == x0[4:8] && k == x0[9:11]
y, u, k = _split_vector(x0, Ypde, PDENLPModels.VoidFESpace())
@test y == x0[1:3] && u == [] && k == x0[4:11]

#= Gridap v15
#Test additional_obj_terms.jl
@testset "Tests of energy FETerm" begin
  f0(k) = dot(k, k)
  function f1(yu)
    y, u = yu
    y * u
  end
  f1s(y) = y
  f2(k, yu) = f1(yu) + f0(k)
  f2s(k, y) = f1s(y) + f0(k)
  #Test MultiField yu
  x1 = zeros(nY)
  yu1 = FEFunction(Y, x1)
  cel1 = Gridap.FESpaces.get_cell_dof_values(yu1)
  yuh1 = CellField(Y, cel1)
  #Test SingleField yu
  x2 = rand(Gridap.FESpaces.num_free_dofs(Ypde))
  yu2 = FEFunction(Ypde, x2)
  cel2 = Gridap.FESpaces.get_cell_dof_values(yu2)
  yuh2 = CellField(Ypde, cel2)

  tnrj = NoFETerm()
  @test _obj_integral(tnrj, [], yu1) == 0.0
  @test _obj_cell_integral(tnrj, [], yuh1) == 0.0
  @test _compute_gradient_k(tnrj, [], yu1) == []
  g = similar(x1)
  _compute_gradient!(g, tnrj, [], yu1, Y, X)
  @test g == zeros(nY)
  @test _compute_hess_coo(tnrj, [], yu1, Y, X) == findnz(spzeros(nY, nY))
  @test _obj_integral(tnrj, [], yu2) == 0.0
  @test _obj_cell_integral(tnrj, ones(2), yuh2) == 0.0
  @test _compute_gradient_k(tnrj, ones(2), yu2) == zeros(2)
  g = similar(vcat(ones(2), x2))
  _compute_gradient!(g, tnrj, ones(2), yu2, Ypde, Xpde)
  @test g == zeros(2 + num_free_dofs(Ypde))
  @test _compute_hess_coo(tnrj, ones(2), yu2, Ypde, Xpde) == findnz(spzeros(nYpde, nYpde))

  tnrj = NoFETerm(f0)
  @test _obj_integral(tnrj, ones(2), yu1) == 2.0
  @test _obj_cell_integral(tnrj, ones(2), yuh1) == 2.0
  @test _compute_gradient_k(tnrj, ones(2), yu1) == 2 * ones(2)
  g = similar(vcat(ones(2), x1))
  _compute_gradient!(g, tnrj, ones(2), yu1, Y, X)
  @test g[1:2] == 2 * ones(2)
  @test g[3:(nY + 2)] == zeros(nY)
  _H = _compute_hess_coo(tnrj, ones(2), yu1, Y, X)
  @test _obj_integral(tnrj, ones(2), yu2) == 2.0
  @test _obj_cell_integral(tnrj, ones(2), yuh2) == 2.0

  @test_throws AssertionError tnrj = MixedEnergyFETerm(f2, trian, 0)
  @test_throws AssertionError tnrj = MixedEnergyFETerm(f2s, trian, 0)

  tnrj = MixedEnergyFETerm(f2, trian, 3)
  @test sum(_obj_integral(tnrj, ones(3), yu1)) == 3.0
  @test _obj_cell_integral(tnrj, ones(3), yuh1) == 0.75 * ones(4)
  @test _compute_gradient_k(tnrj, ones(3), yu1) == 2 * ones(3)
  g = similar(vcat(ones(3), x1))
  _compute_gradient!(g, tnrj, ones(3), yu1, Y, X)
  @test g[1:3] == 2 * ones(3)
  _H = _compute_hess_coo(tnrj, ones(3), yu1, Y, X)
  tnrj = MixedEnergyFETerm(f2s, trian, 3)
  @test typeof(sum(_obj_integral(tnrj, ones(3), yu2))) <: Number
  @test typeof(sum(_obj_cell_integral(tnrj, ones(3), yuh2))) <: Number
  @test _compute_gradient_k(tnrj, ones(3), yu2) == 2 * ones(3)
  g = similar(vcat(ones(3), x2))
  _compute_gradient!(g, tnrj, ones(3), yu2, Ypde, Xpde)
  @test g[1:3] == 2 * ones(3)
  _H = _compute_hess_coo(tnrj, ones(3), yu2, Ypde, Xpde)

  @test_throws DimensionError _obj_cell_integral(tnrj, ones(2), yuh1)
  @test_throws DimensionError _obj_integral(tnrj, ones(2), yu1)
  @test_throws DimensionError _compute_gradient_k(tnrj, ones(2), yu1)

  tnrj = EnergyFETerm(f1, trian)
  @test sum(_obj_integral(tnrj, [], yu1)) == 0.0
  @test sum(_obj_cell_integral(tnrj, [], yuh1)) == 0.0
  @test _compute_gradient_k(tnrj, [], yu1) == []
  g = similar(x1)
  _compute_gradient!(g, tnrj, [], yu1, Y, X)
  @test length(g) == nY
  _H = _compute_hess_coo(tnrj, [], yu1, Y, X)

  tnrj = EnergyFETerm(f1s, trian)
  @test typeof(sum(_obj_integral(tnrj, [], yu2))) <: Number
  typeof(sum(_obj_cell_integral(tnrj, [], yuh2))) <: Number

  @test_throws DimensionError _obj_integral(tnrj, ones(2), yu1)
  @test_throws DimensionError _obj_integral(tnrj, ones(2), yu2)
  @test_throws DimensionError _obj_cell_integral(tnrj, ones(2), yuh1)
  @test_throws DimensionError _obj_cell_integral(tnrj, ones(2), yuh2)
  @test_throws DimensionError _compute_gradient_k(tnrj, ones(2), yu2)
end
=#

@testset "Functions used to create bounds in constructors" begin
  #Domain
  domain = (-1, 1, -1, 1)
  n = 2
  partition = (n, n)
  model = CartesianDiscreteModel(domain, partition)

  #Definition of the spaces:
  Xpde = TestFESpace(
    model,
    ReferenceFE(lagrangian, Float64, 2),
    conformity = :H1,
    dirichlet_tags = "boundary",
  )

  y0(x) = 0.0
  Ypde = TrialFESpace(Xpde, y0)

  Xcon = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), conformity = :H1)
  Ycon = TrialFESpace(Xcon)
  Y = MultiFieldFESpace([Ypde, Ycon])

  #Integration machinery
  trian = Triangulation(model)
  degree = 1
  dΩ = Measure(trian, degree)

  for T in (Float16, Float32, Float64)
    #Example 0:
    lvar, uvar = bounds_functions_to_vectors(
      Y,
      Ycon,
      Ypde,
      trian,
      -T(Inf) * ones(T, 9),
      T(Inf) * ones(T, 9),
      -T(Inf) * ones(T, 9),
      T(Inf) * ones(T, 9),
    )
    @test lvar == -Inf * ones(18)
    @test uvar == Inf * ones(18)
    @test eltype(lvar) == T
    @test eltype(uvar) == T

    #Example 0bis:
    lvar, uvar = bounds_functions_to_vectors(
      Y,
      PDENLPModels.VoidFESpace(),
      Ypde,
      trian,
      -T(Inf) * ones(T, 9),
      T(Inf) * ones(T, 9),
      T[],
      T[],
    )
    @test lvar == -Inf * ones(9)
    @test uvar == Inf * ones(9)
    @test eltype(lvar) == T
    @test eltype(uvar) == T

    #Example 1:
    umin(x) = T(x[1] + x[2])
    umax(x) = T(x[1]^2 + x[2]^2)
    lvar, uvar = bounds_functions_to_vectors(
      Y,
      Ycon,
      Ypde,
      trian,
      -T(Inf) * ones(T, 9),
      T(Inf) * ones(T, 9),
      umin,
      umax,
    )
    @test lvar == vcat(-T(Inf) * ones(T, 9), [-1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
    @test uvar == vcat(T(Inf) * ones(T, 9), 0.5 * ones(T, 9))
    @test eltype(lvar) == T
    @test eltype(uvar) == T

    #Example 1bis:
    umin(x) = T(x[1] + x[2])
    umax(x) = T(x[1]^2 + x[2]^2)
    lvar, uvar = bounds_functions_to_vectors(
      Y,
      PDENLPModels.VoidFESpace(),
      Ypde,
      trian,
      umin,
      umax,
      nothing,
      nothing,
    )
    @test lvar == [1.0, 0.0, 0.0, 1.0, 1.0, -1.0, 0.0, 0.0, 1.0]
    @test uvar == 0.5 * ones(T, 9)
    @test eltype(lvar) == T
    @test eltype(uvar) == T

    #Example 2:
    umin(x) = [T(x[1] + x[2])]
    umax(x) = [T(x[1]^2 + x[2]^2)]
    lvar, uvar = bounds_functions_to_vectors(Y, Ycon, Ypde, trian, umin, umax, umin, umax)
    @test lvar == [
      1.0,
      0.0,
      0.0,
      1.0,
      1.0,
      -1.0,
      0.0,
      0.0,
      1.0,
      -1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      1.0,
      0.0,
      1.0,
      1.0,
    ]
    @test uvar == 0.5 * ones(T, 18)
    @test eltype(lvar) == T
    @test eltype(uvar) == T
  end
end

function check_counters(nlp::AbstractNLPModel)
  #the real list of functions:
  #[:obj, :grad, :cons, :jcon, :jgrad, :jac, :jprod, :jtprod, :hess, :hprod, :jhprod]
  for s in [:obj, :grad, :cons, :jac, :hess]
    eval(Meta.parse("$(s)"))(nlp, nlp.meta.x0)
    @test eval(Meta.parse("neval_$(s)"))(nlp) == 1
  end
  for s in [:jprod, :hprod]
    eval(Meta.parse("$(s)"))(nlp, nlp.meta.x0, nlp.meta.x0)
    @test eval(Meta.parse("neval_$(s)"))(nlp) == 1
  end
  jtprod(nlp, nlp.meta.x0, nlp.meta.y0)
  @test nlp.counters.neval_jtprod == 1

  reset!(nlp) #we trust reset!

  #test Lagrangian Hessian
  hess(nlp, nlp.meta.x0, nlp.meta.y0)
  @test nlp.counters.neval_hess == 1
  hprod(nlp, nlp.meta.x0, nlp.meta.y0, nlp.meta.x0)
  @test nlp.counters.neval_hprod == 1

  reset!(nlp)
end

@testset "Check counters" begin
  EFT = EnergyFETerm((y, u) -> ∫(1.0)dΩ, trian, Ypde, Ycon)
  function res(y, u, v)
    ∫(y * v)dΩ
  end
  cter = FEOperator(res, Y, Xpde)
  function resff(yu, v)
    y, u = yu
    ∫(y * v)dΩ
  end
  rhs(v) = ∫(v * 0.0) * dΩ
  caff = AffineFEOperator(resff, rhs, Y, Xpde)
  x0 = zeros(8)

  nlp = GridapPDENLPModel(x0, EFT, Ypde, Ycon, Xpde, Xcon, cter)
  check_counters(nlp)

  nlp2 = GridapPDENLPModel(x0, EFT, Ypde, Ycon, Xpde, Xcon, caff)
  check_counters(nlp2)

  # increase coverage
  get_tnrj(nlp)
  get_Ypde(nlp)
  get_Ycon(nlp)
  get_Xpde(nlp)
  get_Xcon(nlp)
  get_Y(nlp)
  get_X(nlp)
  get_op(nlp)
  get_nvar_pde(nlp)
  get_nvar_con(nlp)
  get_nparam(nlp)
  get_nnzh_obj(nlp)
  get_Hrows(nlp)
  get_Hcols(nlp)
  get_Jrows(nlp)
  get_Jcols(nlp)
end
