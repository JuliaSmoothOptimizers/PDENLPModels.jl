using Gridap, PDENLPModels, LinearAlgebra, SparseArrays, NLPModels, NLPModelsTest, Test

function basicunconstrainedtyped(args...; n = 2^4, T = Float64, kwargs...)
  ubis(x) = x[1]^2 + x[2]^2
  function f(yu)
    y, u = yu
    T(0.5) * (ubis - u) * (ubis - u) + T(0.5) * y * y
  end
  
  domain = (0, 1, 0, 1)
  partition = (n, n)
  model = CartesianDiscreteModel(domain, partition)
  
  order = 1
  V0 = TestFESpace(
    reffe = :Lagrangian,
    order = order,
    valuetype = T,
    conformity = :H1,
    model = model,
    dirichlet_tags = "boundary",
  )
  U = TrialFESpace(V0, x -> zero(T))
  
  Ypde = U
  Xpde = V0
  Xcon = TestFESpace(
    reffe = :Lagrangian,
    order = order,
    valuetype = T,
    conformity = :H1,
    model = model,
  )
  Ucon = TrialFESpace(Xcon)
  Ycon = Ucon
  trian = Triangulation(model)
  degree = 2
  quad = CellQuadrature(trian, degree)
  
  Y = MultiFieldFESpace([U, Ucon])
  X = MultiFieldFESpace([V0, Xcon])
  xin = zeros(T, Gridap.FESpaces.num_free_dofs(Y))
  return GridapPDENLPModel(xin, f, trian, quad, Y, X)
end

nlp64 = basicunconstrainedtyped() # T = Float64
x64 = nlp64.meta.x0
@test eltype(x64) == Float64
@test typeof(obj(nlp64, x64)) == Float64
nlp32 = basicunconstrainedtyped(T = Float32)
x32 = nlp32.meta.x0
@test eltype(nlp32.meta.x0) == Float32
@test typeof(obj(nlp32, x32)) == Float32
nlp16 = basicunconstrainedtyped(T = Float16)
x16 = nlp16.meta.x0
@test eltype(nlp16.meta.x0) == Float16
@test typeof(obj(nlp16, x16)) == Float16