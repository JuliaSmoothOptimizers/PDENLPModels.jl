"""
`controlelasticmembrane1(; n :: Int = 10, args...)`
Let Ω = (-1,1)^2, we solve the following
distributed Poisson control problem with Dirichlet boundary:
 min_{y ∈ H^1_0,u ∈ H^1}   0.5 ∫_Ω​ |y(x) - yd(x)|^2dx + 0.5 * α * ∫_Ω​ |u|^2
 s.t.         -Δy = h + u,   for    x ∈  Ω
               y  = 0,       for    x ∈ ∂Ω
              umin(x) <=  u(x) <= umax(x)
where yd(x) = -x[1]^2 and α = 1e-2.
The force term is h(x_1,x_2) = - sin( ω x_1)sin( ω x_2) with  ω = π - 1/8.
In this first case, the bound constraints are constants with
umin(x) = 0.0 and umax(x) = 1.0.
"""
function controlelasticmembrane(args...; n = 3, kargs...)

  # Domain
  domain = (-1, 1, -1, 1)
  partition = (n, n)
  model = CartesianDiscreteModel(domain, partition)

  # Definition of the spaces:
  valuetype = Float64
  reffe = ReferenceFE(lagrangian, valuetype, 2)
  Xpde = TestFESpace(model, reffe; conformity = :H1, dirichlet_tags = "boundary")
  y0(x) = 0.0
  Ypde = TrialFESpace(Xpde, y0)

  reffe_con = ReferenceFE(lagrangian, valuetype, 1)
  Xcon = TestFESpace(model, reffe_con; conformity = :H1)
  Ycon = TrialFESpace(Xcon)
  Y = MultiFieldFESpace([Ypde, Ycon])

  # Integration machinery
  trian = Triangulation(model)
  degree = 1
  dΩ = Measure(trian, degree)

  # Objective function:
  yd(x) = -x[1]^2
  α = 1e-2
  function f(yu)
    y, u = yu
    ∫(0.5 * (yd - y) * (yd - y) + 0.5 * α * u * u) * dΩ
  end

  # Definition of the constraint operator
  ω = π - 1 / 8
  h(x) = -sin(ω * x[1]) * sin(ω * x[2])
  function res(yu, v)
    y, u = yu
    ∫(∇(v) ⊙ ∇(y) - v * u) * dΩ #- v * h
  end
  rhs(v) = ∫(v * h) * dΩ
  op = AffineFEOperator(res, rhs, Y, Xpde)

  # It is easy to have a constant bounds
  umin(x) = 0.0
  umax(x) = 1.0
  npde = Gridap.FESpaces.num_free_dofs(Ypde)
  ncon = Gridap.FESpaces.num_free_dofs(Ycon)

  return GridapPDENLPModel(
    zeros(npde + ncon),
    f,
    trian,
    dΩ,
    Ypde,
    Ycon,
    Xpde,
    Xcon,
    op,
    lvaru = zeros(ncon),
    uvaru = ones(ncon),
    name = "controlelasticmembrane1",
  )
end

function controlelasticmembrane_test()
  nlp = controlelasticmembrane()
end
