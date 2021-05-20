#using Gridap, PDENLPModels, LinearAlgebra, SparseArrays, NLPModels, NLPModelsTest, Test

function torebrachistochrone(args...; n = 3, kwargs...)
  # n est la taille de la discrétisation (entier)
  # le domain au format (t₀, T)
  domain = (0, 1)
  # x0 est le vecteur des données initiales
  x0 = zeros(2)
  # xf est le vecteur des données finales
  xf = π * ones(2)
  # xmin et xmax sont des nombres qui représentent les bornes:
  xmin = 0
  xmax = 2 * π
  #La fonction objectif f:
  a = 1
  c = 3
  #Pour utiliser la fonction cos: `operate(cos, x)` vaut cos(x)
  #Pas de carré disponible, donc: `x*x` vaut pour x^2, et `∇(φ) ⊙ ∇(φ)` vaut `φ'^2` (la dérivée au carré)
  function f(x)
    φ, θ = x
    a * a * ∇(φ) ⊙ ∇(φ) + (c + a * operate(cos, φ)) * (c + a * operate(cos, φ)) * ∇(θ) ⊙ ∇(θ)
  end

  model = CartesianDiscreteModel(domain, n)

  labels = get_face_labeling(model)
  add_tag_from_tags!(labels, "diri1", [2])
  add_tag_from_tags!(labels, "diri0", [1])

  V0 = TestFESpace(
    reffe = :Lagrangian,
    order = 1,
    valuetype = Float64,
    conformity = :H1,
    model = model,
    dirichlet_tags = ["diri0", "diri1"],
  )
  V1 = TestFESpace(
    reffe = :Lagrangian,
    order = 1,
    valuetype = Float64,
    conformity = :H1,
    model = model,
    dirichlet_tags = ["diri0", "diri1"],
  )

  U0 = TrialFESpace(V0, [x0[1], xf[1]])
  U1 = TrialFESpace(V0, [x0[2], xf[2]])

  V = MultiFieldFESpace([V0, V1])
  U = MultiFieldFESpace([U0, U1])
  nU0 = Gridap.FESpaces.num_free_dofs(U0)
  nU1 = Gridap.FESpaces.num_free_dofs(U1)

  trian = Triangulation(model)
  degree = 1
  quad = Measure(trian, degree)

  return GridapPDENLPModel(
    zeros(nU0 + nU1),
    f,
    trian,
    quad,
    U,
    V,
    lvar = xmin * ones(nU0 + nU1),
    uvar = xmax * ones(nU0 + nU1),
  )
end

function torebrachistochrone_test()
  nlp = torebrachistochrone()
  stats = ipopt(nlp, print_level = 0)

  nn = Int(nlp.nvar_pde / 2)
  φs = stats.solution[1:nn]
  θs = stats.solution[(nn + 1):(2 * nn)]

  xs = (c .+ a * cos.(φs)) .* cos.(θs)
  ys = (c .+ a * cos.(φs)) .* sin.(θs)
  zs = a * sin.(φs)

  #=
  plotlyjs()
  linspace(from, to, npoints) = range(from, stop=to, length=npoints)

  #plot a torus
  M = 100
  αs = linspace(0, 2π, M)
  βs = linspace(0, 2π, M)
  Xs = (c .+ a * cos.(αs)) * cos.(βs)'
  Ys = (c .+ a * cos.(αs)) * sin.(βs)'
  Zs = (a * sin.(αs)) * ones(M)'
  plot3d(Xs, Ys, Zs, st=:surface, grid=false, c=:grays, axis=false, colorbar=false)
  plot3d!(xs, ys, zs, linewidth=4, color=:red, title=@sprintf("Geodesic on a Torus (length=%4.f)", L), legend=false)
  =#

end
